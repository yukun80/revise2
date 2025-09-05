import os
import yaml
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio
from PIL import Image
from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations_traditional import get_traditional_val_augmentation
from models.model_factory import create_model
import csv


def read_original_rgb(rgb_path):
    """
    直接从原始文件读取RGB数据，与数据集切片可视化保持一致

    Args:
        rgb_path: RGB文件路径
    Returns:
        处理后的RGB数组，格式为(H, W, 3)，值范围0-1
    """
    try:
        with rasterio.open(rgb_path) as src:
            # 读取RGB三个波段
            rgb = src.read([1, 2, 3])  # 读取前3个波段

            # 转换为标准numpy数组
            rgb = np.array(rgb, dtype=np.float32, copy=True)
            rgb = np.ascontiguousarray(rgb)

            # 转换为(H, W, C)格式 - 与用户代码一致
            rgb_display = np.transpose(rgb, (1, 2, 0))

            # 使用与用户完全相同的归一化方式
            rgb_min, rgb_max = np.percentile(rgb_display, (2, 98))
            rgb_display = np.clip((rgb_display - rgb_min) / (rgb_max - rgb_min), 0, 1)

            return rgb_display

    except Exception as e:
        print(f"读取原始RGB文件失败 {rgb_path}: {e}")
        return None


def get_rgb_path(root_dir, sample_basename):
    """
    获取RGB文件路径（数据结构统一，RGB总是在root_dir/rgb/目录下）

    Args:
        root_dir: 数据根目录
        sample_basename: 样本基础名称（不含扩展名）
    Returns:
        RGB文件路径
    """
    return os.path.join(root_dir, "rgb", f"{sample_basename}.tif")


def create_overlay(background, mask, output_path, alpha=0.5):
    """
    创建背景图像与掩码的叠加图像

    Args:
        background: 背景图像，RGB格式，(H,W,3)
        mask: 掩码，(H,W)，值为0或1
        output_path: 输出路径
        alpha: 透明度
    """
    # 确保掩码是uint8类型
    mask = mask.astype(np.uint8)

    # 确保掩码与背景图像大小一致
    if background.shape[:2] != mask.shape:
        # 调整掩码大小以匹配背景
        mask_resized = np.array(Image.fromarray(mask).resize((background.shape[1], background.shape[0]), Image.NEAREST))
    else:
        mask_resized = mask

    # 创建叠加图像
    overlay = background.copy()
    overlay[mask_resized == 1] = [255, 0, 0]  # 红色

    # 创建带透明度的叠加
    result = background.copy()
    cv = overlay.astype(float) * alpha + background.astype(float) * (1 - alpha)
    result = cv.astype(np.uint8)

    # 保存结果
    Image.fromarray(result).save(output_path)


def save_prediction_visualization(pred, output_path):
    """
    保存预测结果的可视化图像

    Args:
        pred: 预测结果，二值掩码
        output_path: 输出路径
    """
    # 确保数据是numpy数组
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()

    # 对于掩码数据，转换为uint8类型
    pred = pred.astype(np.uint8)

    # 二分类掩码使用红黑配色
    cmap = LinearSegmentedColormap.from_list("binary_red", [(0, 0, 0), (1, 0, 0)], N=2)

    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(pred, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_heatmap(arr2d, output_path, cmap="viridis", vmin=0.0, vmax=1.0):
    """
    保存二维数组的热力图

    Args:
        arr2d: 2D numpy array
        output_path: 输出路径
        cmap: 颜色映射
        vmin/vmax: 正则化区间
    """
    if torch.is_tensor(arr2d):
        arr2d = arr2d.detach().cpu().numpy()
    arr2d = np.nan_to_num(arr2d, nan=0.0, posinf=1.0, neginf=0.0)
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def predict(config_path, model_path=None, output_dir=None):
    """
    使用训练好的模型进行推理并保存结果（简化版本，只输出overlay和prediction）

    Args:
        config_path: 配置文件路径
        model_path: 模型权重路径
        output_dir: 输出目录
    """
    # 加载配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 设置模型路径
    if model_path is None:
        model_path = config["prediction"].get("model_path")  # 优先使用配置文件中的路径
        if model_path is None:  # 如果配置文件中没有设置，则使用默认逻辑
            model_path = os.path.join(
                config["training"]["ckpt_dir"],
                f"{config['training']['current_time']}_{config['training']['name']}",
                "model_best.pth",
            )

    # 设置输出目录
    if output_dir is None:
        base_dir = config["prediction"].get("output_dir", "./work_dir")
        current_time = datetime.datetime.now().strftime("%Y%m%d")
        output_dir = f"{base_dir}_{current_time}"

    print(f"输出目录: {output_dir}")
    print(f"模型路径: {model_path}")

    # 先创建基础输出子目录
    dirs = {
        "pred": os.path.join(output_dir, "pred_vis"),  # 预测结果可视化
        "overlay": os.path.join(output_dir, "overlay_vis"),  # 叠加可视化
    }

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = create_model(config)
    model = model.to(device)

    # 加载权重
    print("加载模型权重...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 判断模型类型
    model_type = config["model"].get("type", "segformer").lower()
    print(f"使用模型类型: {model_type}")
    is_haefnet = model_type == "haefnet" and hasattr(model, "analyze_modalities")

    # HAEF-Net 专属输出子目录
    if is_haefnet:
        dirs.update(
            {
                "prob": os.path.join(output_dir, "prob_vis"),  # 最终前景概率热力图
                "uncertainty": os.path.join(output_dir, "uncertainty_vis"),  # 最终不确定性热力图
                "contrib": os.path.join(output_dir, "modal_contrib_vis"),  # 每模态贡献图（前景）
                "modal_uncertainty": os.path.join(output_dir, "modal_uncertainty_vis"),  # 每模态不确定性
                "loo_prob": os.path.join(output_dir, "loo_prob_vis"),  # 留一法概率
                "loo_pred": os.path.join(output_dir, "loo_pred_vis"),  # 留一法分类掩码
                "reliability": os.path.join(output_dir, "reliability"),  # 可靠性参数可视化与CSV
                "npy": os.path.join(output_dir, "arrays"),  # 保存原始数组
            }
        )

    # 创建目录
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 创建数据加载器
    modalities = [k[4:] for k, v in config["modalities"].items() if v and k.startswith("use_")]
    print(f"使用模态: {modalities}")

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
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    print(f"开始推理，共{len(val_loader)}个样本...")

    # 统计变量
    overlay_success_count = 0
    overlay_missing_rgb_count = 0

    # 推理和保存结果
    with torch.no_grad():
        for i, (images, labels, meta) in enumerate(tqdm(val_loader, desc="推理进度")):
            file_id = meta["file_id"]
            if isinstance(file_id, list):
                file_id = file_id[0]

            # 获取基础文件名
            mod_key = modalities[0]
            modality_path = meta["modalities"][mod_key]
            if isinstance(modality_path, list):
                modality_path = modality_path[0]

            basename = os.path.basename(modality_path).replace(".tif", "")

            # 推理
            images_cuda = [img.to(device) for img in images]

            if is_haefnet:
                report = model.analyze_modalities(images_cuda, foreground_class=1, compute_loo=True)
                final_logits = report["final_logits"]
                final_prob = report["final_prob"]
                U = report.get("U", None)
                C_t = report.get("C_t", None)
                U_t = report.get("U_t", None)
                prob_loo = report.get("prob_loo", None)
                beta = report.get("beta", None)

                pred = final_prob.argmax(dim=1).cpu().numpy()[0]
                # 保存 HAEF-Net 特有输出
                try:
                    # 最终前景概率
                    if isinstance(final_prob, torch.Tensor) and final_prob.size(1) >= 2:
                        prob_fg = final_prob[:, 1:2, :, :].squeeze(0).squeeze(0).detach().cpu().numpy()
                        save_heatmap(
                            prob_fg, os.path.join(dirs["prob"], f"{basename}.png"), cmap="magma", vmin=0.0, vmax=1.0
                        )
                        # 保存为npy
                        np.save(os.path.join(dirs["npy"], f"{basename}_prob_fg.npy"), prob_fg)

                    # 不确定性热力图
                    if isinstance(U, torch.Tensor):
                        U_map = U.squeeze(0).squeeze(0).detach().cpu().numpy()
                        save_heatmap(
                            U_map,
                            os.path.join(dirs["uncertainty"], f"{basename}.png"),
                            cmap="inferno",
                            vmin=0.0,
                            vmax=1.0,
                        )
                        np.save(os.path.join(dirs["npy"], f"{basename}_uncertainty.npy"), U_map)

                    # 每模态贡献与不确定性
                    if isinstance(C_t, torch.Tensor):
                        Ct_np = C_t.detach().cpu().numpy()[0]  # [T,H,W]
                        for t_idx in range(Ct_np.shape[0]):
                            mod_name = modalities[t_idx] if t_idx < len(modalities) else f"m{t_idx}"
                            contrib = np.exp(Ct_np[t_idx])  # 将log贡献转为[0,1]范围
                            contrib = np.clip(contrib, 0.0, 1.0)
                            save_heatmap(
                                contrib,
                                os.path.join(dirs["contrib"], f"{basename}_{mod_name}.png"),
                                cmap="plasma",
                                vmin=0.0,
                                vmax=1.0,
                            )
                            np.save(os.path.join(dirs["npy"], f"{basename}_contrib_{mod_name}.npy"), contrib)

                    if isinstance(U_t, torch.Tensor):
                        Ut_np = U_t.detach().cpu().numpy()[0]  # [T,H,W]
                        for t_idx in range(Ut_np.shape[0]):
                            mod_name = modalities[t_idx] if t_idx < len(modalities) else f"m{t_idx}"
                            u_map = np.clip(Ut_np[t_idx], 0.0, 1.0)
                            save_heatmap(
                                u_map,
                                os.path.join(dirs["modal_uncertainty"], f"{basename}_{mod_name}.png"),
                                cmap="viridis",
                                vmin=0.0,
                                vmax=1.0,
                            )
                            np.save(os.path.join(dirs["npy"], f"{basename}_uncertainty_{mod_name}.npy"), u_map)

                    # 留一法概率与掩码
                    if isinstance(prob_loo, (list, tuple)) and len(prob_loo) > 0:
                        for t_idx, p_loo in enumerate(prob_loo):
                            if p_loo is None:
                                continue
                            mod_name = modalities[t_idx] if t_idx < len(modalities) else f"m{t_idx}"
                            # 概率热力图（前景）
                            if p_loo.size(1) >= 2:
                                prob_fg_loo = p_loo[:, 1:2, :, :].squeeze(0).squeeze(0).detach().cpu().numpy()
                                save_heatmap(
                                    prob_fg_loo,
                                    os.path.join(dirs["loo_prob"], f"{basename}_no-{mod_name}.png"),
                                    cmap="magma",
                                    vmin=0.0,
                                    vmax=1.0,
                                )
                                np.save(
                                    os.path.join(dirs["npy"], f"{basename}_loo_prob_no-{mod_name}.npy"), prob_fg_loo
                                )
                            # 掩码
                            pred_loo = p_loo.argmax(dim=1).detach().cpu().numpy()[0]
                            save_prediction_visualization(
                                pred_loo, os.path.join(dirs["loo_pred"], f"{basename}_no-{mod_name}.png")
                            )
                            np.save(
                                os.path.join(dirs["npy"], f"{basename}_loo_pred_no-{mod_name}.npy"),
                                pred_loo.astype(np.uint8),
                            )

                    # 可靠性参数 beta 可视化与CSV
                    if isinstance(beta, torch.Tensor):
                        beta_np = beta.detach().cpu().numpy()  # [T,K]
                        # 条形图（前景类优先）
                        plt.figure(figsize=(8, 4), dpi=200)
                        x = np.arange(beta_np.shape[0])
                        if beta_np.shape[1] >= 2:
                            plt.bar(x - 0.15, beta_np[:, 0], width=0.3, label="class0")
                            plt.bar(x + 0.15, beta_np[:, 1], width=0.3, label="class1")
                        else:
                            plt.bar(x, beta_np[:, 0], width=0.5, label="class0")
                        plt.xticks(x, [m[:8] for m in modalities])
                        plt.ylim(0.0, 1.0)
                        plt.ylabel("beta (reliability)")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(os.path.join(dirs["reliability"], f"{basename}_beta.png"))
                        plt.close()

                        # CSV
                        csv_path = os.path.join(dirs["reliability"], f"{basename}_beta.csv")
                        with open(csv_path, "w", newline="") as fcsv:
                            writer = csv.writer(fcsv)
                            header = ["modality"] + [f"class{i}" for i in range(beta_np.shape[1])]
                            writer.writerow(header)
                            for t_idx in range(beta_np.shape[0]):
                                mod_name = modalities[t_idx] if t_idx < len(modalities) else f"m{t_idx}"
                                writer.writerow([mod_name] + [f"{v:.6f}" for v in beta_np[t_idx].tolist()])
                except Exception as ee:
                    print(f"保存HAEF-Net特有输出时出错 {basename}: {str(ee)}")
            else:
                outputs = model(images_cuda)

                # 处理不同模型的输出格式
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 有些模型返回(outputs, _)格式

                # 获取预测结果
                if isinstance(outputs, list):
                    pred = outputs[-1].argmax(dim=1).cpu().numpy()[0]
                else:
                    pred = outputs.argmax(dim=1).cpu().numpy()[0]

            # 1. 保存预测结果可视化
            pred_path = os.path.join(dirs["pred"], f"{basename}.png")
            save_prediction_visualization(pred, pred_path)

            # 2. 创建叠加图像（统一使用RGB底图，不管配置中是否包含RGB模态）
            # 直接构造RGB文件路径（数据结构统一）
            rgb_file_path = get_rgb_path(root_dir, basename)

            if os.path.exists(rgb_file_path):
                try:
                    # 直接从原始文件读取RGB数据，与数据集切片可视化保持一致
                    rgb_display = read_original_rgb(rgb_file_path)

                    if rgb_display is not None:
                        # 转换为uint8格式用于图像保存
                        rgb_uint8 = (rgb_display * 255).astype(np.uint8)

                        # 确保预测和RGB图像大小一致
                        if pred.shape != rgb_uint8.shape[:2]:
                            pred_resized = np.array(
                                Image.fromarray(pred.astype(np.uint8)).resize(
                                    (rgb_uint8.shape[1], rgb_uint8.shape[0]), Image.NEAREST
                                )
                            )
                        else:
                            pred_resized = pred.astype(np.uint8)

                        # 创建叠加图像
                        overlay_path = os.path.join(dirs["overlay"], f"{basename}.png")
                        create_overlay(rgb_uint8, pred_resized, overlay_path, alpha=0.5)
                        overlay_success_count += 1
                    else:
                        print(f"无法读取RGB文件内容: {basename}")
                        overlay_missing_rgb_count += 1

                except Exception as e:
                    print(f"创建叠加图像失败 {basename}: {str(e)}")
                    overlay_missing_rgb_count += 1
                    # 使用matplotlib作为备选方案（使用原始RGB数据）
                    try:
                        if "rgb_display" in locals() and rgb_display is not None:
                            plt.figure(figsize=(8, 8))
                            plt.imshow(rgb_display[:, :, [0, 1, 2]])  # 与用户代码一致
                            plt.imshow(
                                pred_resized,
                                alpha=0.5,
                                cmap=LinearSegmentedColormap.from_list("r", [(0, 0, 0, 0), (1, 0, 0, 0.5)], N=2),
                            )
                            plt.axis("off")
                            overlay_path = os.path.join(dirs["overlay"], f"{basename}.png")
                            plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0, dpi=100)
                            plt.close()
                            overlay_success_count += 1
                        else:
                            print(f"备选方案也无法使用，跳过叠加图像: {basename}")
                    except Exception as e2:
                        print(f"备选叠加方法也失败 {basename}: {str(e2)}")
            else:
                overlay_missing_rgb_count += 1
                print(f"RGB文件不存在，跳过叠加图像: {os.path.basename(rgb_file_path)}")

            # 每50个样本打印一次进度
            if (i + 1) % 50 == 0:
                print(f"已处理 {i+1}/{len(val_loader)} 个样本")

    print(f"\n推理完成！结果已保存至: {output_dir}")
    print("输出目录:")
    print(f"  - 预测结果可视化: {dirs['pred']}")
    print(f"  - RGB叠加可视化: {dirs['overlay']}")
    if is_haefnet:
        print(f"  - 概率热力图: {dirs['prob']}")
        print(f"  - 不确定性热力图: {dirs['uncertainty']}")
        print(f"  - 每模态贡献: {dirs['contrib']}")
        print(f"  - 每模态不确定性: {dirs['modal_uncertainty']}")
        print(f"  - 留一法概率: {dirs['loo_prob']}")
        print(f"  - 留一法掩码: {dirs['loo_pred']}")
        print(f"  - 可靠性可视化与CSV: {dirs['reliability']}")

    # 统计生成的文件数量
    pred_count = len([f for f in os.listdir(dirs["pred"]) if f.endswith(".png")])
    overlay_count = len([f for f in os.listdir(dirs["overlay"]) if f.endswith(".png")])
    if is_haefnet:
        prob_count = len([f for f in os.listdir(dirs["prob"]) if f.endswith(".png")])
        uncert_count = len([f for f in os.listdir(dirs["uncertainty"]) if f.endswith(".png")])
        contrib_count = len([f for f in os.listdir(dirs["contrib"]) if f.endswith(".png")])
        muncert_count = len([f for f in os.listdir(dirs["modal_uncertainty"]) if f.endswith(".png")])
        loo_prob_count = len([f for f in os.listdir(dirs["loo_prob"]) if f.endswith(".png")])
        loo_pred_count = len([f for f in os.listdir(dirs["loo_pred"]) if f.endswith(".png")])

    print(f"\n生成文件统计:")
    print(f"  - 预测可视化: {pred_count} 个文件")
    print(f"  - 叠加可视化: {overlay_count} 个文件")
    if is_haefnet:
        print(f"  - 概率热力图: {prob_count} 个文件")
        print(f"  - 不确定性热力图: {uncert_count} 个文件")
        print(f"  - 每模态贡献: {contrib_count} 个文件")
        print(f"  - 每模态不确定性: {muncert_count} 个文件")
        print(f"  - 留一法概率: {loo_prob_count} 个文件")
        print(f"  - 留一法掩码: {loo_pred_count} 个文件")

    # 显示叠加图像的详细统计
    print(f"\n叠加图像生成详情:")
    print(f"  - 成功生成: {overlay_success_count} 个")
    if overlay_missing_rgb_count > 0:
        print(f"  - 缺少RGB底图: {overlay_missing_rgb_count} 个")
    print(f"  - RGB底图路径: {root_dir}/rgb/*.tif")

    # 更新配置
    config["prediction"] = config.get("prediction", {})
    config["prediction"]["output_dir"] = output_dir
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="简化版模型推理脚本，只输出预测和叠加可视化")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    print("=" * 60)
    print("简化版预测脚本")
    print("只输出预测可视化(pred_vis)和RGB叠加可视化(overlay_vis)")
    print("=" * 60)

    predict(args.config, args.model, args.output)

"""
使用示例:
# SWIN-HARMF
# 基本使用
python predict.py --config configs/exp_swin-HARMF_250524.yml

# HARMF 250524
python predict.py --config configs/exp_swin-HARMF_250524.yml \
    --model checkpoints/HARMF_sparsity05_128-64_data2_20250524/model_best.pth \
    --output work_dir/HARMF_sparsity05_128-64_data2_20250524_pred_best

# HARMF 250525
python predict.py --config configs/exp_swin-HARMF_250524.yml \
    --model checkpoints/HARMF_sparsity05_128-64_data2_20250524/model_epoch_70.pth \
    --output work_dir/HARMF_sparsity05_128-64_data2_20250525_pred_epoch_70

# HAEF-Net 20250904
python predict.py \
  --config configs/exp_haefnet_test.yml \
  --model checkpoints/HAEFNet_test_20250904/model_best.pth \
  --output work_dir/HAEFNet_test_20250904_pred_best

"""
