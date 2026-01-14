#!/usr/bin/env python3
"""
正确的InterOptimus Interface Agent使用示例

演示如何修复用户遇到的调用错误
"""

from interface_agent import InterfaceGenerationAgent
from pymatgen.core.structure import Structure

def main():
    print("✅ InterOptimus Interface Agent - 正确使用示例")
    print("=" * 55)

    # 创建代理
    agent = InterfaceGenerationAgent()

    print("\n🔧 加载Tutorial中的结构文件...")
    try:
        film_structure = Structure.from_file('Tutorial/film.cif')
        substrate_structure = Structure.from_file('Tutorial/substrate.cif')

        print(f"✅ Film: {film_structure.composition}")
        print(f"✅ Substrate: {substrate_structure.composition}")

    except FileNotFoundError:
        print("⚠️  Tutorial文件不存在，创建模拟结构...")
        # 创建模拟结构用于演示
        from pymatgen.core.lattice import Lattice

        film_lattice = Lattice.cubic(5.431)
        film_coords = [[0, 0, 0], [0.25, 0.25, 0.25]]
        film_structure = Structure(film_lattice, ['Si', 'Si'], film_coords)

        substrate_lattice = Lattice.hexagonal(4.76, 12.99)
        substrate_coords = [[0, 0, 0.352], [0.333, 0.667, 0.082]]
        substrate_structure = Structure(substrate_lattice, ['Al', 'Al'], substrate_coords)

        print(f"✅ Film (模拟): {film_structure.composition}")
        print(f"✅ Substrate (模拟): {substrate_structure.composition}")

    print("\n🎯 正确的调用方式:")

    # 示例1: 修复用户原来的错误调用
    print("\n📝 示例1: 修复用户原来的调用")
    print("原错误调用:")
    print("  result = agent.generate_interface_from_structures(")
    print("      film_structure=film_structure,")
    print("      substrate_structure=substrate_structure,")
    print("      vacuum_type='without_vaccum',  # 拼写错误")
    print("      interface_type='epitaxial')     # 不需要的参数")

    print("\n正确调用:")
    result = agent.generate_interface_from_structures(
        film_structure=film_structure,
        substrate_structure=substrate_structure,
        vacuum_type='without_vacuum',  # 修正拼写
        optimization_level='basic'      # interface_type已移除
    )

    print(f"结果: {'成功' if result['success'] else '失败'}")
    if result['success']:
        print(f"生成界面数: {result['results']['structures_generated']}")

    # 示例2: 带自定义参数的调用
    print("\n📝 示例2: 带自定义匹配参数的调用")
    result2 = agent.generate_interface_from_structures(
        film_structure=film_structure,
        substrate_structure=substrate_structure,
        vacuum_type='with_vacuum',
        optimization_level='basic',
        # 用户自定义匹配限制条件
        max_area=150,          # 最大界面面积
        max_length_tol=0.04,   # 长度匹配容差
        max_angle_tol=0.04,    # 角度匹配容差
        termination_ftol=0.18, # 终止面拟合容差
        output_dir='custom_example'
    )

    print(f"结果: {'成功' if result2['success'] else '失败'}")
    if result2['success']:
        print(f"生成界面数: {result2['results']['structures_generated']}")

    print("\n" + "=" * 55)
    print("📚 关键改进:")
    print("1. ✅ 所有界面统一作为异质结构处理")
    print("2. ✅ interface_type参数已弃用 (向后兼容)")
    print("3. ✅ 自动修正常见拼写错误")
    print("4. ✅ 支持用户自定义匹配限制条件")
    print("5. ✅ 更清晰的错误信息和警告")

    print("\n🎯 记住正确的参数:")
    print("  • vacuum_type: 'with_vacuum' 或 'without_vacuum'")
    print("  • optimization_level: 'basic' 或 'advanced'")
    print("  • 自定义参数: max_area, max_length_tol, max_angle_tol, termination_ftol")

    print("\n📖 更多信息请查看: USAGE_GUIDE.md")

if __name__ == "__main__":
    main()