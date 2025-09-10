import os
import datetime
import yaml
import torch
import numpy as np
from argparse import ArgumentParser

# from tqdm import tqdm

from torch.utils.data import DataLoader

from models.model_factory import create_model
from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations_traditional import get_traditional_val_augmentation
from utils.meter import confusion_matrix


def parse_args():
    parser = ArgumentParser(description="Evaluate model on validation set (IoU, F1, Recall, Precision, ECE)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to checkpoint (e.g., checkpoints/xxx/model_best.pth)"
    )
    parser.add_argument("--output", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--bins", type=int, default=15, help="Number of bins for ECE computation")
    return parser.parse_args()


def compute_binary_metrics_from_confmat(conf_mat):
    """
    Compute binary metrics for foreground class (class index 1) from confusion matrix.

    Args:
        conf_mat (np.ndarray): shape [2, 2]

    Returns:
        dict with keys: iou, precision, recall, f1
    """
    eps = 1e-10
    tp = float(conf_mat[1, 1])
    fp = float(conf_mat[0, 1])
    fn = float(conf_mat[1, 0])

    denom_iou = tp + fp + fn
    iou = tp / (denom_iou + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def update_ece_bins(confidences, correct, bin_totals, bin_conf_sums, bin_correct_sums, num_bins):
    """
    Update ECE accumulators given confidences and correctness booleans for a batch of pixels.
    """
    if confidences.size == 0:
        return
    # Clamp to [0, 1]
    confidences = np.clip(confidences, 0.0, 1.0)
    # Bin indices in [0, num_bins-1]
    bin_indices = np.floor(confidences * num_bins).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Accumulate per-bin totals
    for b in range(num_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        bin_totals[b] += mask.sum()
        bin_conf_sums[b] += confidences[mask].sum()
        bin_correct_sums[b] += correct[mask].sum()


def finalize_ece(bin_totals, bin_conf_sums, bin_correct_sums):
    """
    Compute ECE from accumulators.
    """
    N = float(bin_totals.sum())
    if N <= 0:
        return 0.0
    ece = 0.0
    for tot, conf_sum, corr_sum in zip(bin_totals, bin_conf_sums, bin_correct_sums):
        if tot == 0:
            continue
        acc_b = float(corr_sum) / float(tot)
        conf_b = float(conf_sum) / float(tot)
        ece += (float(tot) / N) * abs(acc_b - conf_b)
    return float(ece)


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve output directory
    if args.output is None:
        base = os.path.join("work_dir", "eval")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base, f"{config['training']['name']}_{ts}")
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = create_model(config)
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(args.model, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # Dataset & loader (val split)
    modalities = [k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")]
    norm_config = config["model"]["modality_norms"]
    root_dir = config["data"]["root_dir"]
    val_list = os.path.join(root_dir, config["data"]["val_list"])

    val_dataset = MultiModalRSDataset(
        root_dir=root_dir,
        file_list=val_list,
        modalities=modalities,
        transform=get_traditional_val_augmentation(norm_config),
        stage="val",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Model type check
    model_type = config["model"].get("type", "segformer").lower()
    is_haefnet = model_type == "haefnet" and hasattr(model, "analyze_modalities")

    # Accumulators
    num_classes = int(config["model"].get("num_classes", 2))
    ignore_index = int(config.get("training", {}).get("loss", {}).get("ignore_index", 255))
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ECE accumulators
    M = int(args.bins)
    bin_totals = np.zeros(M, dtype=np.int64)
    bin_conf_sums = np.zeros(M, dtype=np.float64)
    bin_correct_sums = np.zeros(M, dtype=np.float64)

    print("Starting evaluation loop...")
    with torch.no_grad():
        for i, (images, labels, meta) in enumerate(val_loader):
            print(f"Processing sample {i+1}/{len(val_loader)}")
            images = [img.to(device) for img in images]
            labels = labels.to(device)

            if is_haefnet:
                report = model.analyze_modalities(images, foreground_class=1, compute_loo=False)
                final_prob = report["final_prob"]  # [B,K,H,W]
                logits = torch.log(final_prob.clamp_min(1e-12))
            else:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if isinstance(outputs, list):
                    logits = outputs[-1]
                else:
                    logits = outputs
                # Ensure size matches labels for probability computation
                if logits.shape[2:] != labels.shape[1:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=labels.shape[1:], mode="bilinear", align_corners=False
                    )
                final_prob = torch.softmax(logits, dim=1)

            # Predictions and confidences
            pred = final_prob.argmax(dim=1)  # [B,H,W]
            conf = final_prob.max(dim=1).values  # [B,H,W]

            # Move to CPU numpy
            pred_np = pred.cpu().numpy()
            conf_np = conf.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Update confusion matrix (handle ignore_index)
            conf_mat += confusion_matrix(labels_np.flatten(), pred_np.flatten(), num_classes, ignore_label=ignore_index)

            # ECE: only for valid pixels
            valid = labels_np != ignore_index
            if np.any(valid):
                conf_valid = conf_np[valid]
                correct_valid = (pred_np[valid] == labels_np[valid]).astype(np.float32)
                update_ece_bins(conf_valid, correct_valid, bin_totals, bin_conf_sums, bin_correct_sums, M)

    # Metrics
    if num_classes != 2:
        # For non-binary, report IoU/F1/Recall/Precision for foreground class if exists; else macro on class 1-like
        # Here we fallback to class index 1 if available; otherwise use argmax IoU class
        if num_classes > 1:
            fg_idx = 1
        else:
            fg_idx = 0
        # Build a binary 2x2 confusion from chosen class
        tp = float(conf_mat[fg_idx, fg_idx])
        fp = float(conf_mat[:, fg_idx].sum() - tp)
        fn = float(conf_mat[fg_idx, :].sum() - tp)
        denom_iou = tp + fp + fn
        iou = tp / (denom_iou + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
        metrics = {"IoU": iou, "Precision": precision, "Recall": recall, "F1": f1}
    else:
        m = compute_binary_metrics_from_confmat(conf_mat)
        metrics = {"IoU": m["iou"], "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"]}

    ece = finalize_ece(bin_totals, bin_conf_sums, bin_correct_sums)
    metrics["ECE"] = ece

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.safe_dump({k: float(v) for k, v in metrics.items()}, f)

    # Pretty print
    print("\nEvaluation results (val):")
    for k, v in metrics.items():
        if k in ["IoU", "F1", "Recall", "Precision"]:
            print(f"  {k}: {v * 100.0:.2f}%")
        else:
            print(f"  {k}: {v:.6f}")
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

"""
python eval.py --config configs/exp_haefnet_test.yml\
    --model checkpoints/HAEFNet_test_20250909/model_best.pth\
    --output work_dir/HAEFNet_test_20250909_eval_best
"""
