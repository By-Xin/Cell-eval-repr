"""
VCC GPU加速器 - 完整调用示例集合
涵盖各种真实使用场景的具体代码示例
"""

import os
import time
import torch
import numpy as np
import anndata as ad
import polars as pl
from pathlib import Path


# =============================================================================
# 示例1: 最基础的使用 - 直接替换现有代码
# =============================================================================

def example_1_basic_replacement():
    """
    场景：您现在有使用官方cell-eval的代码，想要GPU加速
    """
    print("=== 示例1: 基础替换使用 ===")
    
    # ---------- 原始代码 (CPU版本) ----------
    print("1. 原始CPU版本代码:")
    print("""
    # 您的原始代码可能长这样:
    from cell_eval import MetricsEvaluator
    
    evaluator = MetricsEvaluator(
        adata_pred="predictions.h5ad",
        adata_real="ground_truth.h5ad", 
        control_pert="non-targeting",
        pert_col="target"
    )
    results, agg = evaluator.compute(profile="vcc")
    """)
    
    # ---------- GPU加速版本 (只改2行!) ----------
    print("2. GPU加速版本 (只需修改2行):")
    print("""
    # 修改第1行: 导入GPU版本
    from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
    
    # 修改第2行: 使用GPU evaluator + 添加GPU参数
    evaluator = AcceleratedMetricsEvaluator(
        adata_pred="predictions.h5ad",
        adata_real="ground_truth.h5ad",
        control_pert="non-targeting", 
        pert_col="target",
        enable_gpu_acceleration=True  # 新增这一行
    )
    results, agg = evaluator.compute(profile="vcc")  # 其余完全相同
    """)
    
    # 实际运行示例 (使用测试数据)
    print("3. 实际运行GPU版本:")
    
    # 生成测试数据
    from cell_eval.data import build_random_anndata
    test_pred = build_random_anndata(n_cells=1000, n_genes=5000, n_perts=50, 
                                    pert_col="target", control_var="non-targeting")
    test_real = build_random_anndata(n_cells=1000, n_genes=5000, n_perts=50,
                                    pert_col="target", control_var="non-targeting")
    
    # GPU加速评估
    from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
    
    start_time = time.time()
    evaluator = AcceleratedMetricsEvaluator(
        adata_pred=test_pred,
        adata_real=test_real,
        control_pert="non-targeting",
        pert_col="target", 
        enable_gpu_acceleration=True
    )
    results, agg = evaluator.compute(profile="vcc")
    gpu_time = time.time() - start_time
    
    print(f"   GPU评分完成: {gpu_time:.3f}秒")
    print(f"   结果维度: {results.shape}")
    print()


# =============================================================================
# 示例2: VCC比赛完整流程
# =============================================================================

def example_2_vcc_competition_workflow():
    """
    场景：完整的VCC比赛评分流程
    """
    print("=== 示例2: VCC比赛完整流程 ===")
    
    # 步骤1: 准备数据文件路径
    print("步骤1: 数据准备")
    
    # 模拟您的实际文件路径
    your_prediction_file = "your_model_predictions.h5ad"   # 您的模型输出
    vcc_test_data_file = "vcc_test_ground_truth.h5ad"     # VCC测试集真值
    baseline_predictions = "vcc_baseline_predictions.h5ad" # VCC baseline
    
    # 对于演示，我们生成测试数据
    print("   生成VCC规模测试数据 (150扰动, 20000基因)...")
    vcc_test_data = build_random_anndata(
        n_cells=3000, n_genes=20000, n_perts=150,
        pert_col="target", control_var="non-targeting"
    )
    your_predictions = build_random_anndata(
        n_cells=3000, n_genes=20000, n_perts=150, 
        pert_col="target", control_var="non-targeting"
    )
    
    # 步骤2: 评估您的模型相对于测试集
    print("步骤2: 评估模型性能")
    
    from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
    
    model_evaluator = AcceleratedMetricsEvaluator(
        adata_pred=your_predictions,        # 您的模型预测
        adata_real=vcc_test_data,          # VCC测试集真值  
        control_pert="non-targeting",       # VCC标准对照组
        pert_col="target",                 # VCC标准扰动列
        enable_gpu_acceleration=True,
        outdir="./vcc_model_evaluation"
    )
    
    print("   正在GPU加速计算VCC指标...")
    start_time = time.time()
    model_results, model_agg = model_evaluator.compute(profile="vcc")
    eval_time = time.time() - start_time
    
    print(f"   模型评估完成: {eval_time:.2f}秒")
    
    # 步骤3: 计算baseline分数 (用于标准化)
    print("步骤3: 计算baseline分数")
    
    # 如果您需要计算baseline
    from cell_eval import build_base_mean_adata
    
    baseline_data = build_base_mean_adata(
        adata=vcc_test_data,  # 使用训练数据构建baseline
        control_pert="non-targeting",
        pert_col="target"
    )
    
    baseline_evaluator = AcceleratedMetricsEvaluator(
        adata_pred=baseline_data,
        adata_real=vcc_test_data,
        control_pert="non-targeting", 
        pert_col="target",
        enable_gpu_acceleration=True,
        outdir="./vcc_baseline_evaluation"
    )
    
    baseline_results, baseline_agg = baseline_evaluator.compute(profile="vcc")
    print("   baseline评估完成")
    
    # 步骤4: 计算相对于baseline的标准化分数
    print("步骤4: 计算VCC最终分数")
    
    from cell_eval import score_agg_metrics
    
    # 保存中间结果
    model_agg.write_csv("./vcc_model_agg.csv")
    baseline_agg.write_csv("./vcc_baseline_agg.csv")
    
    # 计算标准化分数
    final_scores = score_agg_metrics(
        results_user="./vcc_model_agg.csv",
        results_base="./vcc_baseline_agg.csv", 
        output="./vcc_final_scores.csv"
    )
    
    # 提取最终VCC分数
    avg_score_row = final_scores.filter(pl.col("metric") == "avg_score")
    if len(avg_score_row) > 0:
        final_vcc_score = avg_score_row["from_baseline"][0]
        print(f"   🏆 您的VCC最终分数: {final_vcc_score:.4f}")
        
        if final_vcc_score > 0.1:
            print("   🚀 优秀！显著超越baseline")
        elif final_vcc_score > 0:
            print("   ✨ 良好！超越baseline")
        else:
            print("   📈 需要改进，未超越baseline")
    
    print()


# =============================================================================
# 示例3: 批量模型对比
# =============================================================================

def example_3_batch_model_comparison():
    """
    场景：您有多个模型版本，想批量对比性能
    """
    print("=== 示例3: 批量模型对比 ===")
    
    # 准备测试数据
    print("准备测试环境...")
    ground_truth = build_random_anndata(
        n_cells=2000, n_genes=10000, n_perts=100,
        pert_col="target", control_var="non-targeting"
    )
    
    # 模拟多个模型的预测结果
    model_names = ["ModelV1", "ModelV2", "ModelV3", "Baseline"]
    model_predictions = {}
    
    for model_name in model_names:
        # 模拟不同质量的预测结果
        noise_level = {"ModelV1": 0.2, "ModelV2": 0.15, "ModelV3": 0.1, "Baseline": 0.3}
        
        pred_data = build_random_anndata(
            n_cells=2000, n_genes=10000, n_perts=100,
            pert_col="target", control_var="non-targeting"
        )
        model_predictions[model_name] = pred_data
    
    # 批量评估所有模型
    print("批量评估模型性能...")
    
    from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
    
    model_scores = {}
    
    for model_name, pred_data in model_predictions.items():
        print(f"   评估 {model_name}...")
        
        evaluator = AcceleratedMetricsEvaluator(
            adata_pred=pred_data,
            adata_real=ground_truth,
            control_pert="non-targeting",
            pert_col="target",
            enable_gpu_acceleration=True,
            outdir=f"./batch_eval_{model_name.lower()}"
        )
        
        start_time = time.time()
        results, agg = evaluator.compute(profile="vcc")
        eval_time = time.time() - start_time
        
        # 提取关键指标 (简化版本)
        model_scores[model_name] = {
            "eval_time": eval_time,
            "results_shape": results.shape,
            "agg_shape": agg.shape
        }
        
        print(f"      完成，耗时: {eval_time:.2f}秒")
    
    # 显示对比结果
    print("\n=== 模型对比结果 ===")
    print(f"{'模型名称':<10} {'评估时间':<10} {'结果维度':<15}")
    print("-" * 35)
    
    for model_name, scores in model_scores.items():
        print(f"{model_name:<10} {scores['eval_time']:<10.2f} {str(scores['results_shape']):<15}")
    
    print()


# =============================================================================
# 示例4: 自定义参数和高级配置
# =============================================================================

def example_4_advanced_configuration():
    """
    场景：需要自定义参数和高级配置
    """
    print("=== 示例4: 高级配置 ===")
    
    # 准备数据
    test_data = build_random_anndata(n_cells=1500, n_genes=8000, n_perts=75)
    pred_data = build_random_anndata(n_cells=1500, n_genes=8000, n_perts=75)
    
    from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
    
    # 1. 完整参数配置
    print("1. 完整参数配置示例:")
    
    evaluator = AcceleratedMetricsEvaluator(
        # 数据参数
        adata_pred=pred_data,
        adata_real=test_data,
        control_pert="control",        # 您的对照组名称
        pert_col="perturbation",       # 您的扰动列名称
        
        # GPU加速参数
        enable_gpu_acceleration=True,  # 启用GPU
        gpu_device="cuda:0",          # 指定GPU设备 
        fallback_to_cpu=True,         # GPU失败时自动回退
        
        # 输出参数
        outdir="./advanced_eval_results",
        
        # 官方cell-eval参数 (透传)
        allow_discrete=False,
        num_threads=8,
        
        # DE计算参数 (如果需要)
        de_pred=None,  # 可以提供预计算的DE结果
        de_real=None   # 可以提供预计算的DE结果
    )
    
    # 2. 获取系统信息
    print("2. 系统配置信息:")
    sys_info = evaluator.get_acceleration_info()
    for key, value in sys_info.items():
        print(f"   {key}: {value}")
    
    # 3. 运行自定义评估
    print("3. 运行自定义评估:")
    results, agg = evaluator.compute(
        profile="vcc",                # VCC指标集
        use_gpu_for_vcc=True,        # 对VCC指标使用GPU
        write_csv=True,              # 保存CSV结果
        break_on_error=False         # 遇到错误继续执行
    )
    
    print(f"   自定义评估完成: {results.shape}")
    
    # 4. 性能基准测试
    print("4. 性能基准测试:")
    benchmark = evaluator.benchmark_acceleration(
        n_perts=100, n_genes=10000, num_runs=3
    )
    
    if "error" not in benchmark:
        print(f"   CPU平均时间: {benchmark['cpu_time_avg']:.3f}s")
        print(f"   GPU平均时间: {benchmark['gpu_time_avg']:.3f}s") 
        print(f"   加速比: {benchmark['speedup']:.1f}x")
    
    print()


# =============================================================================
# 示例5: 错误处理和调试
# =============================================================================

def example_5_error_handling_and_debugging():
    """
    场景：处理各种可能的错误和调试问题
    """
    print("=== 示例5: 错误处理和调试 ===")
    
    # 1. 环境检查
    print("1. 环境检查:")
    
    # 检查PyTorch和CUDA
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"   当前内存使用: {torch.cuda.memory_allocated() / 1e6:.1f}MB")
    
    # 检查模块导入
    try:
        from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
        print("   ✅ GPU加速模块导入成功")
    except ImportError as e:
        print(f"   ❌ GPU加速模块导入失败: {e}")
        print("   请检查vcc_gpu_accelerator文件夹是否在正确位置")
        return
    
    # 2. 安全的evaluator创建
    print("2. 安全的evaluator创建:")
    
    try:
        # 生成测试数据
        test_real = build_random_anndata(n_cells=500, n_genes=2000, n_perts=20)
        test_pred = build_random_anndata(n_cells=500, n_genes=2000, n_perts=20)
        
        # 创建evaluator with错误处理
        evaluator = AcceleratedMetricsEvaluator(
            adata_pred=test_pred,
            adata_real=test_real,
            control_pert="control", 
            pert_col="perturbation",
            enable_gpu_acceleration=True,
            fallback_to_cpu=True  # 重要：启用自动回退
        )
        
        print("   ✅ Evaluator创建成功")
        
        # 获取详细配置信息
        config = evaluator.get_acceleration_info()
        print(f"   GPU启用: {config['gpu_available']}")
        
    except Exception as e:
        print(f"   ❌ Evaluator创建失败: {e}")
        print("   请检查数据格式和参数配置")
        return
    
    # 3. 安全的计算执行
    print("3. 安全的计算执行:")
    
    try:
        start_time = time.time()
        results, agg = evaluator.compute(profile="vcc")
        compute_time = time.time() - start_time
        
        print(f"   ✅ 计算成功完成: {compute_time:.3f}秒")
        print(f"   结果维度: {results.shape}")
        
    except torch.cuda.OutOfMemoryError:
        print("   ⚠️ GPU内存不足，尝试减少数据规模或使用CPU")
        
        # 自动回退到CPU模式
        evaluator_cpu = AcceleratedMetricsEvaluator(
            adata_pred=test_pred,
            adata_real=test_real,
            control_pert="control",
            pert_col="perturbation", 
            enable_gpu_acceleration=False  # 强制CPU模式
        )
        
        results, agg = evaluator_cpu.compute(profile="vcc")
        print("   ✅ CPU模式计算完成")
        
    except Exception as e:
        print(f"   ❌ 计算失败: {e}")
        print("   请检查数据兼容性和系统配置")
    
    # 4. 内存使用监控
    print("4. 内存使用监控:")
    
    if torch.cuda.is_available():
        # 获取内存使用情况
        allocated = torch.cuda.memory_allocated() / 1e6
        cached = torch.cuda.memory_reserved() / 1e6
        total = torch.cuda.get_device_properties(0).total_memory / 1e6
        
        print(f"   GPU内存已分配: {allocated:.1f}MB")
        print(f"   GPU内存缓存: {cached:.1f}MB") 
        print(f"   GPU内存总计: {total:.1f}MB")
        print(f"   使用率: {(allocated/total)*100:.1f}%")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        print("   🧹 GPU内存已清理")
    
    print()


# =============================================================================
# 示例6: 与现有代码库集成
# =============================================================================

def example_6_integration_with_existing_codebase():
    """
    场景：与现有代码库集成的具体方法
    """
    print("=== 示例6: 代码库集成 ===")
    
    # 1. 现有函数的包装
    print("1. 包装现有评估函数:")
    
    def original_vcc_evaluation_function(pred_file, real_file, output_dir):
        """
        这是您现有的VCC评估函数 (示例)
        """
        # 原始实现 (CPU版本)
        from cell_eval import MetricsEvaluator
        
        evaluator = MetricsEvaluator(
            adata_pred=pred_file,
            adata_real=real_file,
            control_pert="non-targeting",
            pert_col="target",
            outdir=output_dir
        )
        
        return evaluator.compute(profile="vcc")
    
    def gpu_accelerated_vcc_evaluation_function(pred_file, real_file, output_dir, use_gpu=True):
        """
        GPU加速版本 - 包装您的现有函数
        """
        if use_gpu:
            from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
            
            evaluator = AcceleratedMetricsEvaluator(
                adata_pred=pred_file,
                adata_real=real_file,
                control_pert="non-targeting",
                pert_col="target", 
                outdir=output_dir,
                enable_gpu_acceleration=True,
                fallback_to_cpu=True
            )
        else:
            # 回退到原始实现
            from cell_eval import MetricsEvaluator
            
            evaluator = MetricsEvaluator(
                adata_pred=pred_file,
                adata_real=real_file,
                control_pert="non-targeting",
                pert_col="target",
                outdir=output_dir
            )
        
        return evaluator.compute(profile="vcc")
    
    # 2. 类的继承扩展
    print("2. 类的继承扩展:")
    
    class MyVCCEvaluator:
        """您现有的评估类"""
        
        def __init__(self, config):
            self.config = config
            
        def evaluate_model(self, model_predictions):
            # 您的现有逻辑
            pass
    
    class GPUAcceleratedVCCEvaluator(MyVCCEvaluator):
        """GPU加速扩展版本"""
        
        def __init__(self, config, enable_gpu=True):
            super().__init__(config)
            self.enable_gpu = enable_gpu
            
        def evaluate_model(self, model_predictions):
            if self.enable_gpu:
                return self._gpu_evaluate(model_predictions)
            else:
                return super().evaluate_model(model_predictions)
                
        def _gpu_evaluate(self, model_predictions):
            from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
            
            evaluator = AcceleratedMetricsEvaluator(
                adata_pred=model_predictions,
                adata_real=self.config['ground_truth'],
                control_pert=self.config['control_pert'],
                pert_col=self.config['pert_col'],
                enable_gpu_acceleration=True
            )
            
            return evaluator.compute(profile="vcc")
    
    # 3. 配置文件驱动的集成
    print("3. 配置驱动集成:")
    
    def load_evaluation_config():
        """加载评估配置"""
        return {
            "data": {
                "pred_file": "model_predictions.h5ad",
                "real_file": "ground_truth.h5ad", 
                "control_pert": "non-targeting",
                "pert_col": "target"
            },
            "compute": {
                "use_gpu": True,
                "gpu_device": "cuda:0", 
                "fallback_to_cpu": True,
                "profile": "vcc"
            },
            "output": {
                "dir": "./gpu_eval_results",
                "save_csv": True
            }
        }
    
    def run_configured_evaluation(config):
        """基于配置运行评估"""
        
        if config["compute"]["use_gpu"]:
            from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
            
            evaluator = AcceleratedMetricsEvaluator(
                adata_pred=config["data"]["pred_file"],
                adata_real=config["data"]["real_file"],
                control_pert=config["data"]["control_pert"],
                pert_col=config["data"]["pert_col"],
                enable_gpu_acceleration=config["compute"]["use_gpu"],
                gpu_device=config["compute"]["gpu_device"],
                fallback_to_cpu=config["compute"]["fallback_to_cpu"],
                outdir=config["output"]["dir"]
            )
        else:
            from cell_eval import MetricsEvaluator
            
            evaluator = MetricsEvaluator(
                adata_pred=config["data"]["pred_file"],
                adata_real=config["data"]["real_file"],
                control_pert=config["data"]["control_pert"],
                pert_col=config["data"]["pert_col"],
                outdir=config["output"]["dir"]
            )
        
        return evaluator.compute(
            profile=config["compute"]["profile"],
            write_csv=config["output"]["save_csv"]
        )
    
    # 演示配置驱动的使用
    print("   配置驱动评估示例:")
    config = load_evaluation_config()
    print(f"   GPU模式: {config['compute']['use_gpu']}")
    print(f"   输出目录: {config['output']['dir']}")
    
    print()


# =============================================================================
# 主函数 - 运行所有示例
# =============================================================================

def main():
    """运行所有完整示例"""
    print("VCC GPU加速器 - 完整使用示例集合")
    print("=" * 60)
    
    # 检查基础环境
    print(f"环境检查:")
    print(f"   Python版本: {os.sys.version.split()[0]}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 导入检查
    try:
        from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
        print("✅ GPU加速模块导入成功")
    except ImportError:
        print("❌ GPU加速模块导入失败")
        print("请确保vcc_gpu_accelerator/文件夹在正确位置")
        return
    
    from cell_eval.data import build_random_anndata
    print("✅ 测试数据生成模块可用")
    print()
    
    # 运行所有示例
    try:
        example_1_basic_replacement()
        example_2_vcc_competition_workflow() 
        example_3_batch_model_comparison()
        example_4_advanced_configuration()
        example_5_error_handling_and_debugging()
        example_6_integration_with_existing_codebase()
        
        print("🎉 所有示例运行完成！")
        print("\n💡 使用提示:")
        print("1. 对于日常使用，推荐示例1的简单替换方法")
        print("2. 对于VCC比赛，参考示例2的完整流程")  
        print("3. 对于批量评估，参考示例3的批处理方法")
        print("4. 遇到问题时，参考示例5的调试方法")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        print("请检查环境配置和依赖安装")


if __name__ == "__main__":
    # 设置日志级别以查看详细信息
    import logging
    logging.basicConfig(level=logging.INFO)
    
    main()