import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

"""
# 示例1: 标准交叉熵损失
training:
  # ... 其他训练参数 ...
  loss:
    type: cross_entropy  # 或使用 nll
    ignore_index: 255
    class_weights: [1.0, 2.0]  # 可选: 对类别进行加权 (第二类权重更大)

# 示例2: Dice损失
training:
  # ... 其他训练参数 ...
  loss:
    type: dice
    ignore_index: 255

# 示例3: Tversky损失 (Dice损失的一个泛化版本)
training:
  # ... 其他训练参数 ...
  loss:
    type: tversky
    alpha: 0.3  # 控制假阳性的权重
    beta: 0.7   # 控制假阴性的权重
    ignore_index: 255

# 示例4: Focal损失
training:
  # ... 其他训练参数 ...
  loss:
    type: focal
    gamma: 2.0  # 聚焦参数，值越大对易分类样本的惩罚越小
    ignore_index: 255
    class_weights: [1.0, 1.5]  # 可选

# 示例5: 边界感知损失
training:
  # ... 其他训练参数 ...
  loss:
    type: boundary
    theta: 3.0  # 控制权重随距离衰减的速度
    ignore_index: 255

# 示例6: Hausdorff距离损失
training:
  # ... 其他训练参数 ...
  loss:
    type: hausdorff
    ignore_index: 255

# 示例7: 对比学习损失
training:
  # ... 其他训练参数 ...
  loss:
    type: contrastive
    temperature: 0.1  # 温度参数，控制相似度分布的平滑程度
    ignore_index: 255

# 示例8: 多损失函数组合 (Dice + 交叉熵)
training:
  # ... 其他训练参数 ...
  loss:
    type: combined
    ignore_index: 255
    losses:
      - name: cross_entropy
        weight: 1.0
      - name: dice
        weight: 0.5

# 示例9: 多损失函数组合 (边界感知损失 + 交叉熵)
training:
  # ... 其他训练参数 ...
  loss:
    type: combined
    ignore_index: 255
    losses:
      - name: cross_entropy
        weight: 0.7
      - name: boundary
        weight: 0.3
        theta: 3.0

# 示例10: 复杂组合 (交叉熵 + Dice + 边界感知)
training:
  # ... 其他训练参数 ...
  loss:
    type: combined
    ignore_index: 255
    losses:
      - name: cross_entropy
        weight: 0.5
      - name: dice
        weight: 0.3
      - name: boundary
        weight: 0.2
        theta: 5.0
"""


class LossFactory:
    """Factory class for creating loss functions based on configuration."""

    @staticmethod
    def create_loss(config):
        """
        Create a loss function from configuration.

        Args:
            config (dict): Loss configuration dictionary

        Returns:
            callable: Loss function that takes (outputs, labels) and returns loss tensor
        """
        loss_config = config.get("training", {}).get("loss", {})
        loss_type = loss_config.get("type", "nll").lower()

        # Default parameters
        ignore_index = loss_config.get("ignore_index", 255)
        alpha = loss_config.get("alpha", 0.5)  # For combined losses
        weight = None

        # Handle class weights if specified
        if "class_weights" in loss_config:
            weight = torch.tensor(loss_config["class_weights"]).float().cuda()

        # Create loss function based on type
        if loss_type == "nll" or loss_type == "cross_entropy":
            # Standard NLL Loss (with LogSoftmax)
            return LossFactory._create_nll_loss(ignore_index, weight)

        elif loss_type == "dice":
            # Dice Loss
            return LossFactory._create_dice_loss(ignore_index)

        elif loss_type == "focal":
            # Focal Loss
            gamma = loss_config.get("gamma", 2.0)
            return LossFactory._create_focal_loss(gamma, ignore_index, weight)

        elif loss_type == "lovasz":
            # Lovasz Loss for IoU optimization
            return LossFactory._create_lovasz_loss(ignore_index)

        elif loss_type == "boundary":
            # Boundary-aware Loss
            theta = loss_config.get("theta", 3.0)  # Distance parameter
            return LossFactory._create_boundary_loss(theta, ignore_index)

        elif loss_type == "tversky":
            # Tversky Loss (generalization of Dice loss)
            alpha = loss_config.get("alpha", 0.3)
            beta = loss_config.get("beta", 0.7)
            return LossFactory._create_tversky_loss(alpha, beta, ignore_index)

        elif loss_type == "hausdorff":
            # Hausdorff Distance Loss
            return LossFactory._create_hausdorff_loss(ignore_index)

        elif loss_type == "contrastive":
            # Pixel-wise Contrastive Loss
            temperature = loss_config.get("temperature", 0.1)
            return LossFactory._create_contrastive_loss(temperature, ignore_index)

        elif loss_type == "combined":
            # Combined loss (weighted sum of multiple losses)
            losses = []
            weights = []

            for loss_item in loss_config.get("losses", []):
                loss_name = loss_item["name"].lower()
                loss_weight = loss_item.get("weight", 1.0)

                # Create individual loss functions
                if loss_name == "nll" or loss_name == "cross_entropy":
                    losses.append(LossFactory._create_nll_loss(ignore_index, weight))
                elif loss_name == "dice":
                    losses.append(LossFactory._create_dice_loss(ignore_index))
                elif loss_name == "focal":
                    gamma = loss_item.get("gamma", 2.0)
                    losses.append(LossFactory._create_focal_loss(gamma, ignore_index, weight))
                elif loss_name == "lovasz":
                    losses.append(LossFactory._create_lovasz_loss(ignore_index))
                elif loss_name == "boundary":
                    theta = loss_item.get("theta", 3.0)
                    losses.append(LossFactory._create_boundary_loss(theta, ignore_index))
                elif loss_name == "tversky":
                    alpha = loss_item.get("alpha", 0.3)
                    beta = loss_item.get("beta", 0.7)
                    losses.append(LossFactory._create_tversky_loss(alpha, beta, ignore_index))
                elif loss_name == "hausdorff":
                    losses.append(LossFactory._create_hausdorff_loss(ignore_index))
                elif loss_name == "contrastive":
                    temperature = loss_item.get("temperature", 0.1)
                    losses.append(LossFactory._create_contrastive_loss(temperature, ignore_index))

                weights.append(loss_weight)

            # Create combined loss function
            return LossFactory._create_combined_loss(losses, weights, ignore_index)

        else:
            print(f"Warning: Unknown loss type '{loss_type}', defaulting to NLL loss")
            return LossFactory._create_nll_loss(ignore_index, weight)

    @staticmethod
    def _create_nll_loss(ignore_index=255, weight=None):
        """Create NLL loss function."""
        criterion = nn.NLLLoss(weight=weight, ignore_index=ignore_index).cuda()

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)
                soft_output = F.log_softmax(output, dim=1)
                loss += criterion(soft_output, labels)
            return loss

        return loss_fn

    @staticmethod
    def _create_dice_loss(ignore_index=255):
        """Create Dice loss function."""

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Get probabilities with softmax
                probs = F.softmax(output, dim=1)

                # Create one-hot encoded target
                batch_size, num_classes = output.shape[0], output.shape[1]
                one_hot = torch.zeros_like(probs)

                # Handle ignore index
                valid_mask = labels != ignore_index

                # Fill one-hot tensor only for valid pixels
                for i in range(batch_size):
                    for c in range(num_classes):
                        one_hot[i, c, valid_mask[i]] = (labels[i, valid_mask[i]] == c).float()

                # Calculate Dice coefficient for each class and average
                intersection = torch.sum(probs * one_hot, dim=(0, 2, 3))
                union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(one_hot, dim=(0, 2, 3))

                # Dice loss
                dice = (2.0 * intersection) / (union + 1e-6)
                dice_loss = 1.0 - torch.mean(dice)

                loss += dice_loss

            return loss

        return loss_fn

    @staticmethod
    def _create_tversky_loss(alpha=0.3, beta=0.7, ignore_index=255):
        """
        Create Tversky loss function.
        Tversky is a generalization of Dice loss that allows weighting false positives
        and false negatives differently using alpha and beta parameters.

        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            ignore_index: Pixel value to ignore in ground truth
        """

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Get probabilities with softmax
                probs = F.softmax(output, dim=1)

                # Create one-hot encoded target
                batch_size, num_classes = output.shape[0], output.shape[1]
                one_hot = torch.zeros_like(probs)

                # Handle ignore index
                valid_mask = labels != ignore_index

                # Fill one-hot tensor only for valid pixels
                for i in range(batch_size):
                    for c in range(num_classes):
                        one_hot[i, c, valid_mask[i]] = (labels[i, valid_mask[i]] == c).float()

                # Calculate Tversky coefficient for each class
                tp = torch.sum(probs * one_hot, dim=(0, 2, 3))
                fp = torch.sum(probs * (1 - one_hot), dim=(0, 2, 3))
                fn = torch.sum((1 - probs) * one_hot, dim=(0, 2, 3))

                # Tversky index
                tversky = tp / (tp + alpha * fp + beta * fn + 1e-6)
                tversky_loss = 1.0 - torch.mean(tversky)

                loss += tversky_loss

            return loss

        return loss_fn

    @staticmethod
    def _create_focal_loss(gamma=2.0, ignore_index=255, weight=None):
        """Create Focal loss function."""

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Get probabilities with softmax
                probs = F.softmax(output, dim=1)

                # Create one-hot encoded target
                batch_size, num_classes = output.shape[0], output.shape[1]
                one_hot = torch.zeros_like(probs)

                # Handle ignore index
                valid_mask = labels != ignore_index

                # Fill one-hot tensor only for valid pixels
                for i in range(batch_size):
                    for c in range(num_classes):
                        one_hot[i, c, valid_mask[i]] = (labels[i, valid_mask[i]] == c).float()

                # Apply class weights if provided
                if weight is not None:
                    weight_tensor = weight.view(1, -1, 1, 1)
                    one_hot = one_hot * weight_tensor

                # Calculate focal loss: -(1-pt)^gamma * log(pt)
                pt = torch.sum(probs * one_hot, dim=1) + 1e-10
                focal_weight = (1 - pt) ** gamma

                # Cross entropy loss
                ce_loss = -torch.log(pt)

                # Apply focal weight to cross entropy
                focal_loss = focal_weight * ce_loss

                # Average over valid pixels
                focal_loss = torch.sum(focal_loss * valid_mask.float()) / (torch.sum(valid_mask.float()) + 1e-6)

                loss += focal_loss

            return loss

        return loss_fn

    @staticmethod
    def _create_lovasz_loss(ignore_index=255):
        """Create Lovasz loss function for IoU optimization."""

        def lovasz_grad(gt_sorted):
            """Compute gradient of the Lovasz extension w.r.t sorted errors."""
            p = len(gt_sorted)
            gts = gt_sorted.sum()
            intersection = gts - gt_sorted.float().cumsum(0)
            union = gts + (1 - gt_sorted).float().cumsum(0)
            jaccard = 1.0 - intersection / union
            if p > 1:  # cover 1-pixel case
                jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            return jaccard

        def mean_lovasz_loss(logits, labels, ignore=None):
            """Multi-class Lovasz softmax loss."""
            loss = 0
            for c in range(logits.shape[1]):
                # Skip background class if desired
                if c == 0 and ignore is not None and ignore[0]:
                    continue

                # Prepare binary class mask
                labels_c = (labels == c).float()
                if ignore is not None:
                    valid = labels != ignore_index
                    labels_c = labels_c * valid.float()

                # Process class predictions
                logits_c = logits[:, c]

                # Lovasz loss for this class
                class_loss = lovasz_hinge(logits_c, labels_c)
                loss += class_loss

            return loss / logits.shape[1]

        def lovasz_hinge(logits, labels):
            """Lovasz hinge loss for binary classification."""
            # Binary case
            signs = 2.0 * labels - 1.0
            errors = 1.0 - logits * signs

            # Sort errors
            errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
            perm = perm.detach()
            fg_sorted = torch.gather(labels, 1, perm)

            # Gradient of the Lovasz extension
            grad = lovasz_grad(fg_sorted)

            # Compute loss
            loss = torch.mean(torch.sum(errors_sorted * grad, dim=1))
            return loss

        def loss_fn(outputs, labels):
            loss = 0
            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Apply Lovasz loss
                loss += mean_lovasz_loss(output, labels)

            return loss

        return loss_fn

    @staticmethod
    def _create_boundary_loss(theta=3.0, ignore_index=255):
        """
        Create boundary loss function.
        Assigns higher loss weights to pixels near boundaries using distance transforms.

        Args:
            theta: Parameter controlling how quickly weight decays with distance
            ignore_index: Pixel value to ignore in ground truth
        """

        def compute_boundary_weights(masks, theta=3.0):
            """
            Compute boundary weights using distance transform.

            Args:
                masks: One-hot encoded segmentation masks [B, C, H, W]
                theta: Parameter controlling how quickly weight decays with distance

            Returns:
                Weights tensor [B, H, W]
            """
            batch_size, num_classes, height, width = masks.shape
            weights = torch.ones((batch_size, height, width), device=masks.device)

            # Process each sample in the batch
            for b in range(batch_size):
                distance_maps = []

                # Compute for each class
                for c in range(num_classes):
                    # Get binary mask for this class
                    mask = masks[b, c].cpu().numpy().astype(np.uint8)

                    # Compute distance transform (distance to nearest boundary)
                    # For background pixels (mask=0), distance to nearest foreground (mask=1)
                    dt_bg = distance_transform_edt(1 - mask)
                    # For foreground pixels (mask=1), distance to nearest background (mask=0)
                    dt_fg = distance_transform_edt(mask)

                    # Combine distances (one will be 0)
                    distance = dt_bg + dt_fg

                    # Normalize and convert back to tensor
                    distance = torch.from_numpy(distance).float().to(masks.device)
                    distance_maps.append(distance)

                # Combine distance maps (min distance to any boundary)
                if num_classes > 1:
                    distance_map = torch.stack(distance_maps, dim=0)
                    distance_map, _ = torch.min(distance_map, dim=0)
                else:
                    distance_map = distance_maps[0]

                # Apply exponential weighting function
                weights[b] = torch.exp(-distance_map / theta)

            return weights

        def loss_fn(outputs, labels):
            loss = 0
            criterion = nn.NLLLoss(reduction="none", ignore_index=ignore_index).cuda()

            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Get softmax probabilities
                probs = F.softmax(output, dim=1)
                log_probs = F.log_softmax(output, dim=1)

                # Create one-hot encoded target
                batch_size, num_classes = output.shape[0], output.shape[1]
                one_hot = torch.zeros_like(probs)

                # Handle ignore index
                valid_mask = labels != ignore_index

                # Fill one-hot tensor
                for i in range(batch_size):
                    for c in range(num_classes):
                        one_hot[i, c, labels[i] == c] = 1.0

                # Compute boundary weights
                boundary_weights = compute_boundary_weights(one_hot, theta)

                # Compute standard cross-entropy loss
                ce_loss = criterion(log_probs, labels)

                # Apply boundary weights to loss
                weighted_loss = ce_loss * boundary_weights

                # Average over valid pixels
                boundary_loss = torch.sum(weighted_loss * valid_mask.float()) / (torch.sum(valid_mask.float()) + 1e-6)

                loss += boundary_loss

            return loss

        return loss_fn

    @staticmethod
    def _create_hausdorff_loss(ignore_index=255):
        """
        Create Hausdorff Distance loss function.
        Approximates the Hausdorff distance between predicted and ground truth boundaries.

        Args:
            ignore_index: Pixel value to ignore in ground truth
        """

        def find_boundaries(mask):
            """Find boundaries in a binary mask using morphological operations."""
            # Convert to numpy
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            from skimage.morphology import binary_dilation, binary_erosion

            # Compute boundary via erosion/dilation difference
            dilated = binary_dilation(mask)
            eroded = binary_erosion(mask)
            boundary = np.logical_xor(dilated, eroded)

            return torch.from_numpy(boundary).float().to(mask.device if isinstance(mask, torch.Tensor) else "cpu")

        def compute_hausdorff_loss(pred, target):
            """Compute modified Hausdorff loss between prediction and target."""
            # For each prediction, find closest target pixel
            pred_boundary = find_boundaries(pred > 0.5)
            target_boundary = find_boundaries(target > 0.5)

            # Handle empty boundaries
            if torch.sum(pred_boundary) == 0 or torch.sum(target_boundary) == 0:
                if torch.sum(pred_boundary) == 0 and torch.sum(target_boundary) == 0:
                    return torch.tensor(0.0, device=pred.device)
                else:
                    return torch.tensor(1.0, device=pred.device)

            # Compute distance maps (brute force for simplicity)
            h, w = pred_boundary.shape
            pred_points = torch.nonzero(pred_boundary)
            target_points = torch.nonzero(target_boundary)

            # For each pred boundary point, find closest target boundary point
            pred_to_target = torch.zeros(len(pred_points), device=pred.device)
            for i, p in enumerate(pred_points):
                distances = torch.sum((target_points.float() - p.float().unsqueeze(0)) ** 2, dim=1)
                pred_to_target[i] = torch.sqrt(torch.min(distances))

            # For each target boundary point, find closest pred boundary point
            target_to_pred = torch.zeros(len(target_points), device=pred.device)
            for i, p in enumerate(target_points):
                distances = torch.sum((pred_points.float() - p.float().unsqueeze(0)) ** 2, dim=1)
                target_to_pred[i] = torch.sqrt(torch.min(distances))

            # Modified Hausdorff distance
            return (torch.mean(pred_to_target) + torch.mean(target_to_pred)) / 2.0

        def loss_fn(outputs, labels):
            loss = 0

            for output in outputs:
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Get softmax probabilities
                probs = F.softmax(output, dim=1)

                # Process each sample and class
                batch_size, num_classes = output.shape[0], output.shape[1]
                batch_loss = 0

                for b in range(batch_size):
                    class_loss = 0
                    valid_classes = 0

                    for c in range(num_classes):
                        # Skip background class (c=0) optionally
                        if c == 0 and num_classes > 1:
                            continue

                        # Create binary mask for this class
                        pred_mask = probs[b, c]
                        target_mask = (labels[b] == c).float()

                        # Skip if no pixels in ground truth
                        if torch.sum(target_mask) == 0:
                            continue

                        # Compute Hausdorff distance
                        class_loss += compute_hausdorff_loss(pred_mask, target_mask)
                        valid_classes += 1

                    if valid_classes > 0:
                        batch_loss += class_loss / valid_classes

                loss += batch_loss / batch_size

            return loss

        return loss_fn

    @staticmethod
    def _create_contrastive_loss(temperature=0.1, ignore_index=255):
        """
        Create pixel-wise contrastive loss function.
        Pulls together features of pixels from the same class and pushes apart features from different classes.

        Args:
            temperature: Temperature parameter for scaling similarities
            ignore_index: Pixel value to ignore in ground truth
        """

        def loss_fn(outputs, labels):
            # Contrastive loss requires feature maps, not just final outputs
            # Assumes outputs include feature maps as auxiliary outputs

            # If no feature maps available, fall back to NLL loss
            if len(outputs) <= 1 or not hasattr(outputs, "features"):
                print("Warning: No feature maps available for contrastive loss. Falling back to NLL loss.")
                return LossFactory._create_nll_loss(ignore_index)(outputs, labels)

            # Get feature maps and output logits
            features = outputs.features  # Assume model outputs feature maps
            loss = 0

            for feat_map, output in zip(features, outputs):
                # Ensure output is interpolated to the correct size
                output = F.interpolate(output, size=labels.shape[1:], mode="bilinear", align_corners=False)

                # Ensure feature map is interpolated to the same size
                feat_map = F.interpolate(feat_map, size=labels.shape[1:], mode="bilinear", align_corners=False)

                batch_size, num_features, height, width = feat_map.shape

                # Normalize features
                feat_map = F.normalize(feat_map, p=2, dim=1)

                # Reshape feature map and labels for processing
                feat_map = feat_map.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_features)
                labels_flat = labels.view(batch_size, -1)

                # Handle ignore index
                valid_mask = labels_flat != ignore_index

                # Process each sample
                batch_loss = 0

                for b in range(batch_size):
                    # Get valid pixels
                    b_valid = valid_mask[b]
                    if torch.sum(b_valid) == 0:
                        continue

                    # Get features and labels for valid pixels
                    b_feat = feat_map[b, b_valid]
                    b_labels = labels_flat[b, b_valid]

                    # Compute similarity matrix
                    similarity = torch.matmul(b_feat, b_feat.transpose(0, 1)) / temperature

                    # Create mask for positive pairs (same class)
                    pos_mask = b_labels.unsqueeze(1) == b_labels.unsqueeze(0)
                    neg_mask = ~pos_mask

                    # Remove self-similarities from positive mask
                    pos_mask.fill_diagonal_(False)

                    # InfoNCE loss
                    # For each anchor, compute loss against all positives and negatives
                    exp_sim = torch.exp(similarity)

                    # Sum of exp similarities for positives
                    pos_exp_sum = torch.sum(exp_sim * pos_mask, dim=1)

                    # Sum of all exp similarities (except self)
                    all_exp_sum = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

                    # Compute loss: -log(sum_pos / sum_all)
                    # Add small epsilon to avoid log(0)
                    loss_per_anchor = -torch.log((pos_exp_sum + 1e-10) / (all_exp_sum + 1e-10))

                    # Average loss across all valid anchors
                    b_loss = torch.mean(loss_per_anchor)
                    batch_loss += b_loss

                # Average loss across batch
                loss += batch_loss / batch_size

            return loss

        return loss_fn

    @staticmethod
    def _create_combined_loss(losses, weights, ignore_index=255):
        """Create a combined loss function from multiple loss functions."""

        def loss_fn(outputs, labels):
            total_loss = 0

            for i, (loss_fn, weight) in enumerate(zip(losses, weights)):
                loss_value = loss_fn(outputs, labels)
                total_loss += weight * loss_value

            return total_loss

        return loss_fn
