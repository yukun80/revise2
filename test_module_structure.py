#!/usr/bin/env python3
"""
HAEF-Net模块结构测试脚本
不依赖torch，只测试模块导入和基本结构
"""

import sys
import os

# 添加项目路径
sys.path.append(".")


def test_module_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")

    try:
        # 测试GEM-Layer导入
        from models.geo_evidential_mapping import GeoEvidentialMappingLayer

        print("✓ GeoEvidentialMappingLayer 导入成功")

        # 测试HAEF-Net导入
        from models.haef_net import HAEFNet

        print("✓ HAEFNet 导入成功")

        # 测试模型工厂导入
        from models.model_factory import create_model

        print("✓ create_model 导入成功")

        return True

    except Exception as e:
        print(f"✗ 模块导入失败: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件加载 ===")

    try:
        import yaml

        config_path = "configs/exp_haefnet_test.yml"
        if not os.path.exists(config_path):
            print(f"✗ 配置文件不存在: {config_path}")
            return False

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print("✓ 配置文件加载成功")
        print(f"  模型类型: {config['model']['type']}")
        print(f"  骨干网络: {config['model']['backbone']}")
        print(f"  类别数量: {config['model']['num_classes']}")
        print(f"  模态数量: {config['model']['num_modalities']}")
        print(f"  GEM原型维度: {config['model']['gem_prototype_dim']}")
        print(f"  使用证据融合: {config['model']['use_evidential_fusion']}")

        return True

    except Exception as e:
        print(f"✗ 配置文件加载失败: {str(e)}")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n=== 测试文件结构 ===")

    required_files = [
        "models/geo_evidential_mapping.py",
        "models/haef_net.py",
        "models/model_factory.py",
        "configs/exp_haefnet_test.yml",
        "test_haefnet.py",
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False

    return all_exist


def test_code_syntax():
    """测试代码语法"""
    print("\n=== 测试代码语法 ===")

    try:
        # 测试GEM-Layer语法
        with open("models/geo_evidential_mapping.py", "r") as f:
            code = f.read()
        compile(code, "models/geo_evidential_mapping.py", "exec")
        print("✓ geo_evidential_mapping.py 语法正确")

        # 测试HAEF-Net语法
        with open("models/haef_net.py", "r") as f:
            code = f.read()
        compile(code, "models/haef_net.py", "exec")
        print("✓ haef_net.py 语法正确")

        # 测试模型工厂语法
        with open("models/model_factory.py", "r") as f:
            code = f.read()
        compile(code, "models/model_factory.py", "exec")
        print("✓ model_factory.py 语法正确")

        return True

    except SyntaxError as e:
        print(f"✗ 语法错误: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("开始HAEF-Net模块结构测试...")

    # 运行测试
    tests = [
        ("文件结构", test_file_structure),
        ("代码语法", test_code_syntax),
        ("模块导入", test_module_imports),
        ("配置文件", test_config_loading),
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
        print("🎉 所有结构测试通过！")
        print("\n下一步:")
        print("1. 确保PyTorch环境正确安装")
        print("2. 运行 python test_haefnet.py 进行完整测试")
        print("3. 使用配置文件 configs/exp_haefnet_test.yml 训练模型")
    else:
        print("❌ 部分测试失败，请检查实现。")

    return all_passed


if __name__ == "__main__":
    main()
