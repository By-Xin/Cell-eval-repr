"""
VCC GPU加速使用示例
演示如何使用异构加速模块提升VCC评分速度
"""

import numpy as np
import torch
import time
from pathlib import Path

# 导入官方cell-eval (不修改)
from cell_eval import MetricsEvaluator
from cell_eval.data import build_random_anndata

# 导入我们的GPU加速模块 (完全独立)
from vcc_gpu_accelerator import AcceleratedMetricsEvaluator, HybridVCCEvaluator
from vcc_gpu_accelerator.utils import format_benchmark_results, memory_usage_estimate


def example_1_basic_usage():
    """示例1: 基础使用 - 直接替换官方evaluator"""
    print("=== 示例1: 基础GPU加速使用 ===\n")
    
    # 生成测试数据
    print("生成测试数据...")
    adata_real = build_random_anndata(n_cells=2000, n_genes=5000, n_perts=50)
    adata_pred = build_random_anndata(n_cells=2000, n_genes=5000, n_perts=50)
    
    # 方式1: 原始CPU版本
    print("1. 使用原始CPU版本:")
    start_time = time.time()
    cpu_evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control", 
        pert_col="perturbation"
    )
    cpu_results, cpu_agg = cpu_evaluator.compute(profile="vcc")
    cpu_time = time.time() - start_time
    print(f"   CPU时间: {cpu_time:.3f}秒")
    
    # 方式2: GPU加速版本 (API完全兼容)
    print("2. 使用GPU加速版本:")
    start_time = time.time()
    gpu_evaluator = AcceleratedMetricsEvaluator(  # 唯一的改动：类名
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
        enable_gpu_acceleration=True  # 启用GPU加速
    )
    gpu_results, gpu_agg = gpu_evaluator.compute(profile="vcc")
    gpu_time = time.time() - start_time
    print(f"   GPU时间: {gpu_time:.3f}秒")
    print(f"   加速比: {cpu_time/gpu_time:.1f}x\n")
    
    # 显示系统信息
    print("GPU加速器信息:")
    info = gpu_evaluator.get_acceleration_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print()


def example_2_benchmark():
    """示例2: 性能基准测试"""
    print("=== 示例2: 性能基准测试 ===\n")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU基准测试")
        return
    
    # 创建加速evaluator
    dummy_data = build_random_anndata(n_cells=100, n_genes=100, n_perts=10)
    evaluator = AcceleratedMetricsEvaluator(
        adata_pred=dummy_data,
        adata_real=dummy_data,
        control_pert="control",
        pert_col="perturbation"
    )
    
    # 运行基准测试 (不同数据规模)
    test_cases = [
        {"n_perts": 50, "n_genes": 10000, "desc": "小规模测试"},
        {"n_perts": 150, "n_genes": 20000, "desc": "VCC真实规模"},
        {"n_perts": 300, "n_genes": 20000, "desc": "大规模测试"}
    ]
    
    for case in test_cases:
        print(f"运行 {case['desc']} ({case['n_perts']} 扰动, {case['n_genes']} 基因)...")
        
        # 内存估算
        memory_info = memory_usage_estimate(case['n_perts'], case['n_genes'])
        print(f"   预计GPU内存需求: {memory_info['total_estimated_mb']:.1f}MB")
        print(f"   {memory_info['recommendation']}")
        
        # 性能测试
        benchmark_results = evaluator.benchmark_acceleration(
            n_perts=case['n_perts'], 
            n_genes=case['n_genes'],
            num_runs=3
        )
        
        if "error" not in benchmark_results:
            speedup = benchmark_results.get('speedup', 0)
            print(f"   加速比: {speedup:.1f}x")
            if speedup > 5:
                print("   ✅ 显著加速!")
            elif speedup > 2:
                print("   ✨ 适度加速")
            else:
                print("   ⚠️  加速效果有限")
        else:
            print(f"   ❌ 测试失败: {benchmark_results['error']}")
        print()


def example_3_hybrid_computing():
    """示例3: 直接使用异构计算器 (高级用法)"""
    print("=== 示例3: 直接异构计算 ===\n")
    
    # 创建模拟的扰动效应数据 
    n_perts, n_genes = 100, 15000
    print(f"生成 {n_perts} 扰动 × {n_genes} 基因的效应矩阵...")
    
    real_effects = np.random.randn(n_perts, n_genes).astype(np.float32)
    pred_effects = real_effects + np.random.randn(n_perts, n_genes).astype(np.float32) * 0.1
    pert_names = np.array([f"PERT_{i:03d}" for i in range(n_perts)])
    
    # 创建异构计算器
    hybrid_evaluator = HybridVCCEvaluator(
        gpu_device='cuda',
        enable_gpu=True,
        fallback_to_cpu=True
    )
    
    print("异构计算系统信息:")
    sys_info = hybrid_evaluator.get_system_info()
    for key, value in sys_info.items():
        print(f"   {key}: {value}")
    print()
    
    # 执行异构计算
    print("执行异构计算 (GPU: MAE+Discrimination, CPU: Overlap)...")
    start_time = time.time()
    
    individual, aggregated = hybrid_evaluator.compute_vcc_metrics(
        real_effects=real_effects,
        pred_effects=pred_effects,
        perturbation_names=pert_names,
        de_real=None,  # 这里简化，实际使用时传入DE数据
        de_pred=None
    )
    
    compute_time = time.time() - start_time
    print(f"计算完成，耗时: {compute_time:.3f}秒\n")
    
    # 显示结果
    print("VCC指标结果:")
    for metric, scores in individual.items():
        if isinstance(scores, dict) and not metric.startswith('_'):
            avg_score = np.mean(list(scores.values()))
            print(f"   {metric}: {avg_score:.4f} (平均)")
    
    if 'vcc_final_score' in aggregated:
        print(f"   VCC最终分数: {aggregated['vcc_final_score']:.4f}")
    
    # 计算性能统计
    if '_computation_stats' in aggregated:
        stats = aggregated['_computation_stats']
        print(f"\n计算时间分解:")
        for component, time_taken in stats.items():
            print(f"   {component}: {time_taken:.3f}秒")
    print()


def example_4_production_workflow():
    """示例4: 生产环境工作流程"""
    print("=== 示例4: 生产环境使用流程 ===\n")
    
    # 模拟真实VCC场景
    print("模拟VCC比赛场景...")
    
    # 1. 创建训练数据 (baseline)
    print("1. 创建训练数据...")
    train_data = build_random_anndata(
        n_cells=5000, n_genes=20000, n_perts=150,
        pert_col="target", control_var="non-targeting"
    )
    
    # 2. 创建预测数据 (您的模型输出)
    print("2. 准备模型预测数据...")  
    pred_data = build_random_anndata(
        n_cells=5000, n_genes=20000, n_perts=150,
        pert_col="target", control_var="non-targeting"
    )
    
    # 3. 设置加速评估器
    print("3. 初始化GPU加速评估器...")
    evaluator = AcceleratedMetricsEvaluator(
        adata_pred=pred_data,
        adata_real=train_data,  # 作为ground truth
        control_pert="non-targeting",
        pert_col="target",
        enable_gpu_acceleration=True,
        fallback_to_cpu=True  # 安全回退
    )
    
    # 4. 计算VCC分数
    print("4. 计算VCC评分 (GPU加速)...")
    start_time = time.time()
    
    results, agg_results = evaluator.compute(
        profile="vcc",
        use_gpu_for_vcc=True
    )
    
    evaluation_time = time.time() - start_time
    print(f"   评分计算完成: {evaluation_time:.2f}秒")
    
    # 5. 分析结果
    print("5. 分析评分结果:")
    print(f"   个体结果维度: {results.shape}")
    print(f"   聚合结果维度: {agg_results.shape}")
    
    # 提取关键指标
    if hasattr(agg_results, 'filter'):  # Polars DataFrame
        try:
            vcc_metrics = ["mae", "discrimination_score_l1", "overlap_at_N"]
            for metric in vcc_metrics:
                if metric in agg_results.columns:
                    mean_row = agg_results.filter(pl.col("statistic") == "mean")
                    if len(mean_row) > 0:
                        score = mean_row[metric][0]
                        print(f"   {metric}: {score:.4f}")
        except:
            pass
    
    # 6. 保存结果 (可选)
    print("6. 保存结果...")
    results.write_csv("vcc_detailed_results.csv")
    agg_results.write_csv("vcc_aggregated_results.csv") 
    print("   结果已保存到CSV文件")
    
    print("\n🎉 VCC评分流程完成!")


def main():
    """运行所有示例"""
    print("VCC GPU加速模块使用示例\n")
    print("="*50)
    
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print("="*50 + "\n")
    
    try:
        # 运行示例
        example_1_basic_usage()
        example_2_benchmark()
        example_3_hybrid_computing()
        example_4_production_workflow()
        
        print("🎉 所有示例运行完成!")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查环境配置和依赖包安装")


if __name__ == "__main__":
    main()