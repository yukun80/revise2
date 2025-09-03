import os
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset


class MultiModalRSDataset(Dataset):
    """Multi-Modal Remote Sensing Dataset.
    支持读取多模态遥感数据，包括 dem, insar_vel, insar_phase, rgb 等。
    """

    def __init__(self, root_dir, file_list, modalities, transform=None, stage="train"):
        """
        Args:
            root_dir (str): Root directory containing modality subfolders (e.g., rgb, dem, label).
            file_list (str): Path to a text file listing sample IDs.
            modalities (list): List of modality names (e.g., ['rgb', 'dem']).
            transform (callable, optional): Transformations to apply.
            stage (str): 'train' or 'val'.
        """
        self.root_dir = root_dir
        self.modalities = modalities
        self.transform = transform
        self.stage = stage

        # Load sample IDs from file_list
        with open(file_list, "r") as f:
            self.datalist = [line.strip() for line in f if line.strip()]

        # Verify dataset integrity
        self._verify_dataset()

    def _verify_dataset(self):
        """Verify that all required files exist and are readable."""
        valid_samples = []
        for file_id in self.datalist:
            is_valid = True
            # 检查所有模态文件
            for modality in self.modalities:
                filepath = os.path.join(self.root_dir, modality, f"{file_id}.tif")
                if not os.path.exists(filepath):
                    print(f"Warning: {filepath} not found, skipping sample {file_id}")
                    is_valid = False
                    break
                try:
                    with rasterio.open(filepath) as src:
                        if modality == "rgb" and src.count < 3:
                            print(f"Warning: {filepath} has {src.count} bands, expected 3")
                            is_valid = False
                except Exception as e:
                    print(f"Warning: Failed to read {filepath}: {e}")
                    is_valid = False
                    break

            # 检查标签文件
            label_path = os.path.join(self.root_dir, "label", f"{file_id}.tif")
            if not os.path.exists(label_path):
                print(f"Warning: {label_path} not found, skipping {file_id}")
                is_valid = False

            if is_valid:
                valid_samples.append(file_id)

        self.datalist = valid_samples
        print(f"Found {len(self.datalist)} valid samples")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        file_id = self.datalist[idx]
        images = []  # 存储所有模态的图像
        meta = {"file_id": file_id, "modalities": {}}

        try:
            # Load modality images
            for modality in self.modalities:
                filepath = os.path.join(self.root_dir, modality, f"{file_id}.tif")
                img = self.read_image(filepath, modality)
                if img is None:
                    raise ValueError(f"Failed to load {modality} for {file_id}")
                images.append(img)
                meta["modalities"][modality] = filepath

            # Load label
            label_path = os.path.join(self.root_dir, "label", f"{file_id}.tif")
            with rasterio.open(label_path) as src:
                label_data = src.read(1)  # Single-band label

                # 确保转换为标准的numpy数组
                label_np = np.array(label_data, dtype=np.int64, copy=True)

                # 确保数组是连续的
                label_np = np.ascontiguousarray(label_np)

                # 使用torch.tensor替代torch.from_numpy
                label = torch.tensor(label_np, dtype=torch.long)
                meta["label"] = label_path

            # 应用数据增强
            if self.transform:
                sample = {"label": label, **{m: img for m, img in zip(self.modalities, images)}}
                sample = self.transform(sample)
                images = [sample[m] for m in self.modalities]
                label = sample["label"]

            # # Verify tensor shapes (assuming model expects 112x112 input)
            # for i, img in enumerate(images):
            #     assert img.shape == (
            #         3,
            #         112,
            #         112,
            #     ), f"{file_id}, {self.modalities[i]}: Expected (3, 112, 112), got {img.shape}"
            # assert label.shape == (112, 112), f"{file_id}: Expected label (112, 112), got {label.shape}"
            # Verify tensor shapes (assuming model expects 128x128 input)
            for i, img in enumerate(images):
                assert img.shape == (
                    3,
                    128,
                    128,
                ), f"{file_id}, {self.modalities[i]}: Expected (3, 128, 128), got {img.shape}"
            assert label.shape == (128, 128), f"{file_id}: Expected label (128, 128), got {label.shape}"

            # return images, label  # Return (List[Tensor], Tensor)
            return images, label, meta
        except Exception as e:
            print(f"Error processing sample {file_id}: {str(e)}")
            raise e

    def read_image(self, filepath, modality):
        """Read TIF image and return as torch.Tensor.
        Args:
            filepath (str): 图像路径
            modality (str): 模态名称
        Returns:
            torch.Tensor: 图像数据
            dict: 元数据
        """
        with rasterio.open(filepath) as src:
            # 检查图像的波段数
            num_bands = src.count

            if modality == "rgb":
                if num_bands >= 3:
                    # 真正的三波段RGB图像
                    img = src.read([1, 2, 3])  # Read 3 bands for RGB
                else:
                    # 单波段图像（灰度图）
                    img = src.read(1)  # 读取单波段
            elif modality == "insar_vel" or modality == "insar_phase" or modality == "dem":
                img = src.read(1)  # Read single band for these modalities
            else:
                img = src.read(1)  # 其他模态读取单通道

            # 确保转换为标准的numpy数组并保证连续内存布局
            img = np.array(img, dtype=np.float32, copy=True)
            img = np.ascontiguousarray(img)
            # img = torch.from_numpy(img).float()

            # 使用torch.tensor替代torch.from_numpy
            img = torch.tensor(img, dtype=torch.float32)

            # Adjust dimensions
            if img.dim() == 2:  # [H, W] -> [1, H, W] -> [3, H, W]
                img = img.unsqueeze(0)
                img = torch.cat([img, img, img], dim=0)
            elif img.dim() == 3 and img.shape[0] == 1:  # [1, H, W] -> [3, H, W]
                img = torch.cat([img, img, img], dim=0)

            return img

    def set_stage(self, stage):
        """设置数据集阶段（训练/验证）"""
        self.stage = stage
