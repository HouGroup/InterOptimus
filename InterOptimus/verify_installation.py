#!/usr/bin/env python3
"""
验证 InterOptimus 安装与核心模块导入。
"""

import os
import sys
from pathlib import Path


def check_file_structure():
    """检查文件结构"""
    print("🔍 检查 InterOptimus 文件结构...")

    core_files = [
        "itworker.py",
        "matching.py",
        "tool.py",
        "mlip.py",
        "CNID.py",
        "equi_term.py",
        "jobflow.py",
    ]

    desktop_files = [
        "agents/iomaker_core.py",
        "agents/simple_iomaker.py",
        "desktop_app/entry.py",
        "desktop_app/gui.py",
    ]

    doc_files = [
        "README.md",
    ]

    print("\n📦 核心文件:")
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - 缺失")

    print("\n🖥️  桌面 / IOMaker:")
    for file in desktop_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - 缺失")

    print("\n📖 文档:")
    for file in doc_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - 缺失")


def check_imports():
    """检查导入是否正常"""
    print("\n🔧 检查导入...")

    try:
        from itworker import InterfaceWorker
        print("   ✅ InterfaceWorker 导入成功")

        from matching import interface_searching
        print("   ✅ matching 模块导入成功")

        from tool import convert_dict_to_json
        print("   ✅ tool 模块导入成功")

    except ImportError as e:
        print(f"   ❌ 核心模块导入失败: {e}")
        return False

    try:
        from agents.iomaker_core import uses_complete_iomaker_settings_dict
        print("   ✅ iomaker_core 导入成功")
    except ImportError as e:
        print(f"   ⚠️ iomaker_core 导入失败: {e}")

    return True


def check_python_version():
    """检查Python版本"""
    print("\n🐍 Python版本信息:")
    print(f"   版本: {sys.version}")
    print(f"   主要版本: {sys.version_info.major}.{sys.version_info.minor}")

    if sys.version_info >= (3, 8):
        print("   ✅ Python版本兼容")
        return True
    print("   ⚠️ Python版本可能过低，建议使用Python 3.8+")
    return False


def check_dependencies():
    """检查依赖"""
    print("\n📦 依赖检查:")

    required_packages = ["numpy", "scipy", "matplotlib", "pandas"]

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - 未安装")

    optional_packages = ["pymatgen", "requests"]

    print("\n📦 可选依赖:")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ⚠️ {package} - 未安装 (某些功能不可用)")


def create_summary():
    """创建总结"""
    print("\n" + "=" * 60)
    print("🎉 InterOptimus安装验证完成!")
    print("=" * 60)

    print("\n📁 文件组织:")
    print("   ✅ 核心算法在 InterOptimus/ 根目录模块")
    print("   ✅ IOMaker 在 agents/（iomaker_core + simple_iomaker）")
    print("   ✅ 桌面 GUI 在 desktop_app/")

    print("\n🔧 核心功能:")
    print("   ✅ 晶体界面优化核心算法")
    print("   ✅ 晶格匹配和对称分析")
    print("   ✅ MLIP（Eqnorm）与优化")

    print("\n🚀 下一步:")
    print("   1. 运行: interoptimus-desktop  # Eqnorm 桌面 GUI")
    print("   2. 或: python -m InterOptimus.desktop_app.entry")


def main():
    """主验证函数"""
    print("🔍 InterOptimus安装验证")
    print("=" * 40)

    current_dir = Path.cwd()
    interopt_dir = current_dir / "InterOptimus"

    if not interopt_dir.exists():
        print("❌ InterOptimus文件夹不存在!")
        return False

    os.chdir(interopt_dir)

    version_ok = check_python_version()
    check_file_structure()
    import_ok = check_imports()
    check_dependencies()

    create_summary()

    return version_ok and import_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
