import torch
import torch.nn as nn
import logging
from models.segformer import WeTr, TransformerBackbone


def create_model(config):
    """Create a model based on configuration.

    Args:
        config (dict): Configuration dictionary with model parameters

    Returns:
        nn.Module: Created model
    """
    model_type = config["model"].get("type", "segformer").lower()
    num_classes = config["model"]["num_classes"]

    # Get modalities from config
    modalities = [k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")]
    num_modalities = len(modalities)

    # Ensure num_modalities is correctly set in the model config
    config["model"]["num_modalities"] = num_modalities

    modality_dims = {modality: config["model"]["modality_dims"].get(modality, 3) for modality in modalities}

    print(f"Creating {model_type} model with {num_modalities} modalities: {modalities}")
    print(f"Modality dimensions: {modality_dims}")

    if model_type in ["segformer", "wetr", "geminifusion"]:
        # Set the number of modalities in modules.py
        from models.modules import set_num_parallel

        set_num_parallel(num_modalities)

        # Get backbone type
        backbone = config["model"]["backbone"]
        fusion = config["model"].get("fusion", {}).get("type", "conditional")
        fusion_sparsity = config["model"].get("fusion", {}).get("sparsity", 0.5)

        # Get drop path rate and other hyperparameters
        dpr = config["model"].get("drop_path_rate", 0.1)
        drop_rate = config["model"].get("drop_rate", 0.0)
        n_heads = config["model"].get("n_heads", 8)

        # 准备融合参数
        fusion_params = {
            "type": fusion,
            "sparsity": fusion_sparsity,
        }

        # 对于Swin backbone，提前设置mmdet日志级别为ERROR
        if backbone.startswith("swin"):
            mmdet_logger = logging.getLogger("mmdet")
            original_level = mmdet_logger.level
            print("Temporarily setting mmdet logger level to ERROR to suppress weight mismatch warning.")
            mmdet_logger.setLevel(logging.ERROR)

        # 如果是Swin backbone，恢复原来的日志级别
        if backbone.startswith("swin"):
            mmdet_logger.setLevel(original_level)

        # Create WeTr with Swin or MiT backbone and PixelCrossAttention fusion
        print("Creating model with PixelCrossAttention fusion")
        model = WeTr(
            backbone=backbone,
            num_classes=num_classes,
            n_heads=n_heads,
            dpr=dpr,
            drop_rate=drop_rate,
            num_parallel=num_modalities,
            fusion_params=fusion_params,
        )

        # Initialize from pretrained weights if available
        if config["model"].get("pretrained", False):
            # For segformer (MiT) backbones
            if not backbone.startswith("swin"):
                pretrained_path = f"pretrained/{backbone}.pth"
                try:
                    state_dict = torch.load(pretrained_path)
                    # Remove head weights that don't match
                    if "head.weight" in state_dict:
                        state_dict.pop("head.weight")
                    if "head.bias" in state_dict:
                        state_dict.pop("head.bias")

                    # Expand state dict for multiple modalities
                    from models.segformer import expand_state_dict

                    state_dict = expand_state_dict(model.state_dict(), state_dict, num_modalities)
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Successfully loaded pretrained weights from {pretrained_path}")
                except Exception as e:
                    print(f"Warning: Failed to load pretrained weights: {e}")
                    print("Training from scratch...")
            else:
                # For Swin backbone, weights are loaded in TransformerBackbone initialization
                print(f"Using Swin backbone '{backbone}' - weights loaded during initialization")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Print model summary
    print(f"Created {model_type} model with {num_modalities} modalities")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # 融合信息
    print("✓ Using PixelCrossAttention fusion (no CSCE/HMCB)")

    return model
