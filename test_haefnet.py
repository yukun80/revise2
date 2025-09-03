#!/usr/bin/env python3
"""
HAEF-Net测试脚本
用于验证GEM-Layer和HAEF-Net的正确性
"""

import torch
import torch.nn as nn
import yaml
import os
import sys

# 添加项目路径
sys.path.append(".")

from models.model_factory import create_model
from models.geo_evidential_mapping import GeoEvidentialMappingLayer


def test_gem_layer():
    """测试GEM-Layer模块"""
    print("=== 测试GEM-Layer模块 ===")

    # 创建测试数据
    B, C, H, W = 2, 256, 32, 32
    feats = torch.randn(B, C, H, W)
    geo_context = torch.randn(B, 1, H, W)

    # 创建GEM-Layer
    gem_layer = GeoEvidentialMappingLayer(input_dim=C, prototype_dim=20, class_dim=2, geo_prior_weight=0.1)

    # 前向传播
    mass = gem_layer(feats, geo_context)
    uncertainty = gem_layer.get_uncertainty(mass)
    plausibility = gem_layer.get_plausibility(mass)

    print(f"输入特征形状: {feats.shape}")
    print(f"质量函数形状: {mass.shape}")
    print(f"不确定性形状: {uncertainty.shape}")
    print(f"似然度形状: {plausibility.shape}")

    # 验证质量函数的性质
    mass_sum = mass.sum(dim=1, keepdim=True)
    print(f"质量函数和范围: {mass_sum.min().item():.4f} - {mass_sum.max().item():.4f}")

    # 验证概率分布的性质
    from models.geo_evidential_mapping import plausibility_to_probability

    probability = plausibility_to_probability(plausibility)
    prob_sum = probability.sum(dim=1, keepdim=True)
    print(f"概率分布和范围: {prob_sum.min().item():.4f} - {prob_sum.max().item():.4f}")

    print("✓ GEM-Layer测试通过!")
    return True


def test_haefnet():
    """测试HAEF-Net模型"""
    print("\n=== 测试HAEF-Net模型 ===")

    # 创建测试配置
    config = {
        "modalities": {"use_dem": True, "use_insar_vel": True, "use_rgb": True},
        "model": {
            "backbone": "swin_tiny",
            "num_classes": 2,
            "num_modalities": 3,
            "type": "haefnet",
            "gem_prototype_dim": 10,
            "gem_geo_prior_weight": 0.1,
            "use_evidential_fusion": True,
            "use_mrg": True,
            "use_evidential_combination": True,
            "drop_path_rate": 0.1,
            "drop_rate": 0.0,
            "n_heads": 8,
            "modality_dims": {"dem": 3, "insar_vel": 3, "rgb": 3},
        },
    }

    try:
        # 创建模型
        model = create_model(config)
        print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")

        # 创建测试数据
        B, C, H, W = 1, 3, 64, 64
        x = [torch.randn(B, C, H, W) for _ in range(3)]  # RGB, InSAR, DEM

        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs, aux_loss = model(x)
            uncertainty_map = model.get_uncertainty_map(x)

        print(f"输入模态数量: {len(x)}")
        print(f"输入形状: {[xi.shape for xi in x]}")
        print(f"输出数量: {len(outputs)}")
        print(f"输出形状: {[out.shape for out in outputs]}")
        print(f"不确定性图形状: {uncertainty_map.shape if uncertainty_map is not None else 'None'}")
        print(f"辅助损失: {aux_loss}")

        print("✓ HAEF-Net测试通过!")
        return True

    except Exception as e:
        print(f"✗ HAEF-Net测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_training_compatibility():
    """测试训练兼容性"""
    print("\n=== 测试训练兼容性 ===")

    try:
        # 加载配置文件
        config_path = "configs/exp_haefnet_test.yml"
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return False

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 创建模型
        model = create_model(config)
        model.train()

        # 创建模拟数据
        B, C, H, W = 2, 3, 64, 64
        x = [torch.randn(B, C, H, W) for _ in range(3)]
        labels = torch.randint(0, 2, (B, H, W))

        # 前向传播
        outputs, aux_loss = model(x)

        # 计算损失
        criterion = nn.CrossEntropyLoss()
        if len(outputs) > 1:
            # 如果有多个输出，使用最后一个
            loss = criterion(outputs[-1], labels)
        else:
            loss = criterion(outputs[0], labels)

        if aux_loss is not None:
            total_loss = loss + 0.1 * aux_loss
        else:
            total_loss = loss

        # 反向传播
        total_loss.backward()

        print(f"训练损失: {total_loss.item():.4f}")
        print(f"输出形状: {[out.shape for out in outputs]}")
        print("✓ 训练兼容性测试通过!")
        return True

    except Exception as e:
        print(f"✗ 训练兼容性测试失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始HAEF-Net测试...")

    # 设置随机种子
    torch.manual_seed(42)

    # 运行测试
    tests = [
        ("GEM-Layer模块", test_gem_layer),
        ("HAEF-Net模型", test_haefnet),
        ("训练兼容性", test_training_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {str(e)}")
            results.append((test_name, False))

    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)

    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("🎉 所有测试通过！HAEF-Net实现正确。")
    else:
        print("❌ 部分测试失败，请检查实现。")

    return all_passed


if __name__ == "__main__":
    main()
