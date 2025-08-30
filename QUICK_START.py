#!/usr/bin/env python3
"""
VCC GPU加速器 - 5分钟快速入门脚本

运行此脚本来:
1. 检查环境配置
2. 测试GPU加速功能 
3. 对比CPU vs GPU性能
4. 验证安装正确性

使用方法:
    python QUICK_START.py
"""

import os
import sys
import time
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # Python版本
    python_version = sys.version.split()[0]
    print(f"   Python版本: {python_version}")
    if tuple(map(int, python_version.split('.'))) < (3, 10):
        print("   ⚠️  警告: 推荐Python 3.10+")
    
    # PyTorch
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU设备: {gpu_name}")
            print(f"   GPU内存: {gpu_memory:.1f}GB")
        else:
            print("   ℹ️  CUDA不可用，将使用CPU模式")
            
    except ImportError:
        print("   ❌ PyTorch未安装")
        print("   请运行: pip install torch")
        return False
    
    # cell-eval
    try:
        import cell_eval
        print(f"   cell-eval: 已安装")
    except ImportError:
        print("   ❌ cell-eval未安装") 
        print("   请运行: pip install cell-eval")
        return False
    
    # GPU加速模块
    try:
        from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
        print("   ✅ GPU加速模块: 已安装")
    except ImportError:
        print("   ❌ GPU加速模块未找到")
        print("   请确保vcc_gpu_accelerator/文件夹在当前目录下")
        return False
    
    print("   ✅ 环境检查通过\n")
    return True

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能...")
    
    try:
        # 生成测试数据
        from cell_eval.data import build_random_anndata
        
        print("   生成测试数据...")
        test_real = build_random_anndata(
            n_cells=1000, n_genes=5000, n_perts=30,
            pert_col="target", control_var="non-targeting"
        )
        test_pred = build_random_anndata(
            n_cells=1000, n_genes=5000, n_perts=30,
            pert_col="target", control_var="non-targeting"  
        )
        print(f"   数据维度: {test_real.shape}")
        
        # 测试GPU加速evaluator
        print("   创建GPU加速evaluator...")
        from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
        
        evaluator = AcceleratedMetricsEvaluator(
            adata_pred=test_pred,
            adata_real=test_real,
            control_pert="non-targeting",
            pert_col="target",
            enable_gpu_acceleration=True,
            fallback_to_cpu=True
        )
        
        # 获取系统信息
        sys_info = evaluator.get_acceleration_info()
        print(f"   GPU可用: {sys_info['gpu_available']}")
        
        # 运行计算
        print("   运行VCC指标计算...")
        start_time = time.time()
        results, agg = evaluator.compute(profile="vcc")
        compute_time = time.time() - start_time
        
        print(f"   ✅ 计算完成: {compute_time:.3f}秒")
        print(f"   结果维度: {results.shape}")
        print("   ✅ 基本功能测试通过\n")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 基本功能测试失败: {e}")
        return False

def performance_benchmark():
    """性能基准测试"""
    print("⚡ 性能基准测试...")
    
    try:
        from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
        from cell_eval.data import build_random_anndata
        
        # 创建evaluator
        dummy_data = build_random_anndata(n_cells=100, n_genes=100, n_perts=5)
        evaluator = AcceleratedMetricsEvaluator(
            adata_pred=dummy_data,
            adata_real=dummy_data,
            control_pert="control",
            pert_col="perturbation",
            enable_gpu_acceleration=True
        )
        
        # 运行基准测试
        test_cases = [
            {"n_perts": 50, "n_genes": 5000, "desc": "小规模"},
            {"n_perts": 100, "n_genes": 10000, "desc": "中等规模"},
        ]
        
        if evaluator.get_acceleration_info()['gpu_available']:
            print("   GPU基准测试结果:")
            print(f"   {'规模':<8} {'GPU时间':<10} {'CPU时间':<10} {'加速比':<8}")
            print("   " + "-" * 36)
            
            for case in test_cases:
                benchmark = evaluator.benchmark_acceleration(
                    n_perts=case['n_perts'],
                    n_genes=case['n_genes'], 
                    num_runs=2
                )
                
                if "error" not in benchmark:
                    gpu_time = benchmark['gpu_time_avg']
                    cpu_time = benchmark['cpu_time_avg']  
                    speedup = benchmark['speedup']
                    
                    print(f"   {case['desc']:<8} {gpu_time:<10.3f} {cpu_time:<10.3f} {speedup:<8.1f}x")
                else:
                    print(f"   {case['desc']:<8} 测试失败")
        else:
            print("   GPU不可用，跳过性能测试")
            
        print("   ✅ 性能测试完成\n")
        return True
        
    except Exception as e:
        print(f"   ❌ 性能测试失败: {e}")
        return False

def usage_examples():
    """显示使用示例"""
    print("📚 快速使用指南:")
    print()
    
    print("1️⃣  最简单用法 (替换现有代码):")
    print("   # 原始代码:")
    print("   from cell_eval import MetricsEvaluator")
    print()
    print("   # GPU加速版本 (只改这一行!):")
    print("   from vcc_gpu_accelerator import AcceleratedMetricsEvaluator")
    print()
    print("   # 其余代码完全相同，只需添加:")
    print("   evaluator = AcceleratedMetricsEvaluator(")
    print("       ...,  # 所有原有参数")
    print("       enable_gpu_acceleration=True  # 新增此行")
    print("   )")
    print()
    
    print("2️⃣  完整VCC评分流程:")
    print("   evaluator = AcceleratedMetricsEvaluator(")
    print("       adata_pred='your_predictions.h5ad',")
    print("       adata_real='ground_truth.h5ad',") 
    print("       control_pert='non-targeting',")
    print("       pert_col='target',")
    print("       enable_gpu_acceleration=True")
    print("   )")
    print("   results, agg = evaluator.compute(profile='vcc')")
    print()
    
    print("3️⃣  更多示例:")
    print("   运行 COMPLETE_USAGE_EXAMPLES.py 查看完整示例")
    print("   运行 example_gpu_acceleration.py 查看详细演示") 
    print()

def create_template_script():
    """创建模板脚本"""
    template_content = '''"""
VCC GPU加速评分模板
复制此代码到您的脚本中，修改文件路径即可使用
"""

from vcc_gpu_accelerator import AcceleratedMetricsEvaluator

def main():
    # TODO: 修改为您的实际文件路径
    pred_file = "your_model_predictions.h5ad"  # 您的模型预测结果
    real_file = "vcc_test_ground_truth.h5ad"   # VCC测试集真值
    
    # TODO: 根据您的数据调整参数
    control_pert = "non-targeting"  # 对照组名称
    pert_col = "target"            # 扰动列名称
    
    # 创建GPU加速evaluator
    evaluator = AcceleratedMetricsEvaluator(
        adata_pred=pred_file,
        adata_real=real_file,
        control_pert=control_pert,
        pert_col=pert_col,
        enable_gpu_acceleration=True,
        fallback_to_cpu=True,  # 安全回退
        outdir="./vcc_gpu_results"
    )
    
    # 运行VCC评分
    print("开始GPU加速VCC评分...")
    results, agg_results = evaluator.compute(profile="vcc")
    
    print(f"评分完成! 结果保存到: ./vcc_gpu_results/")
    print(f"个体结果: {results.shape}")
    print(f"聚合结果: {agg_results.shape}")

if __name__ == "__main__":
    main()
'''
    
    template_file = "vcc_gpu_template.py"
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"📄 已创建模板脚本: {template_file}")
    print("   复制此模板，修改文件路径即可快速开始使用")
    print()

def main():
    """主函数"""
    print("🚀 VCC GPU加速器 - 5分钟快速入门")
    print("=" * 50)
    print()
    
    # 步骤1: 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请先安装缺失的依赖")
        return
    
    # 步骤2: 功能测试  
    if not test_basic_functionality():
        print("❌ 功能测试失败，请检查安装")
        return
    
    # 步骤3: 性能测试
    performance_benchmark()
    
    # 步骤4: 使用指南
    usage_examples()
    
    # 步骤5: 创建模板
    create_template_script()
    
    print("🎉 快速入门完成！")
    print()
    print("📋 后续步骤:")
    print("1. 使用vcc_gpu_template.py作为起点")
    print("2. 查看STEP_BY_STEP_GUIDE.md了解详细指导") 
    print("3. 运行COMPLETE_USAGE_EXAMPLES.py查看更多示例")
    print("4. 遇到问题时参考vcc_gpu_accelerator/README.md")
    print()
    print("💡 记住：只需要修改2行代码就能获得5-30倍加速！")

if __name__ == "__main__":
    main()