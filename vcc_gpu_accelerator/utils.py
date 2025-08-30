"""
辅助工具函数
"""

import numpy as np
from typing import Tuple
from scipy.sparse import issparse
import logging

from cell_eval._types import PerturbationAnndataPair

logger = logging.getLogger(__name__)


def extract_perturbation_effects(
    anndata_pair: PerturbationAnndataPair,
    control_pert: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从官方PerturbationAnndataPair提取扰动效应矩阵
    
    Args:
        anndata_pair: 官方的AnnData配对对象
        control_pert: 对照组标识
        
    Returns:
        (real_effects, pred_effects, perturbation_names)
        - real_effects: (n_perts, n_genes) 真实扰动效应
        - pred_effects: (n_perts, n_genes) 预测扰动效应  
        - perturbation_names: (n_perts,) 扰动名称
    """
    logger.info("Extracting perturbation effects from AnnData pair")
    
    # 获取扰动列表 (排除control)
    perturbations = anndata_pair.perts  # 已经排除了control
    n_perts = len(perturbations)
    n_genes = anndata_pair.real.shape[1]
    
    logger.info(f"Found {n_perts} perturbations and {n_genes} genes")
    
    # 初始化效应矩阵
    real_effects = np.zeros((n_perts, n_genes), dtype=np.float32)
    pred_effects = np.zeros((n_perts, n_genes), dtype=np.float32)
    
    # 计算对照组的平均表达
    real_control_mask = anndata_pair.real.obs[anndata_pair.pert_col] == control_pert
    pred_control_mask = anndata_pair.pred.obs[anndata_pair.pert_col] == control_pert
    
    real_control_mean = _compute_mean_expression(
        anndata_pair.real[real_control_mask].X
    )
    pred_control_mean = _compute_mean_expression(
        anndata_pair.pred[pred_control_mask].X
    )
    
    logger.info(f"Computed control means (shape: {real_control_mean.shape})")
    
    # 为每个扰动计算效应 (扰动均值 - 对照均值)
    for i, pert in enumerate(perturbations):
        # 真实数据
        real_pert_mask = anndata_pair.real.obs[anndata_pair.pert_col] == pert
        real_pert_mean = _compute_mean_expression(
            anndata_pair.real[real_pert_mask].X
        )
        real_effects[i] = real_pert_mean - real_control_mean
        
        # 预测数据  
        pred_pert_mask = anndata_pair.pred.obs[anndata_pair.pert_col] == pert
        pred_pert_mean = _compute_mean_expression(
            anndata_pair.pred[pred_pert_mask].X
        )
        pred_effects[i] = pred_pert_mean - pred_control_mean
    
    logger.info(f"Extracted effects matrices: real {real_effects.shape}, pred {pred_effects.shape}")
    
    return real_effects, pred_effects, perturbations


def _compute_mean_expression(expression_matrix) -> np.ndarray:
    """
    计算表达矩阵的平均值，处理稠密和稀疏矩阵
    
    Args:
        expression_matrix: (n_cells, n_genes) 表达矩阵
        
    Returns:
        (n_genes,) 平均表达向量
    """
    if issparse(expression_matrix):
        return np.array(expression_matrix.mean(axis=0)).flatten().astype(np.float32)
    else:
        return expression_matrix.mean(axis=0).astype(np.float32)


def validate_perturbation_data(
    real_effects: np.ndarray,
    pred_effects: np.ndarray, 
    perturbation_names: np.ndarray
) -> bool:
    """
    验证扰动数据的有效性
    
    Args:
        real_effects: 真实效应矩阵
        pred_effects: 预测效应矩阵
        perturbation_names: 扰动名称
        
    Returns:
        是否通过验证
    """
    checks = []
    
    # 形状检查
    if real_effects.shape != pred_effects.shape:
        logger.error(f"Shape mismatch: real {real_effects.shape} vs pred {pred_effects.shape}")
        checks.append(False)
    else:
        checks.append(True)
    
    # 扰动数量检查
    if len(perturbation_names) != real_effects.shape[0]:
        logger.error(f"Perturbation count mismatch: {len(perturbation_names)} vs {real_effects.shape[0]}")
        checks.append(False)
    else:
        checks.append(True)
    
    # 数据类型检查
    if not (np.isfinite(real_effects).all() and np.isfinite(pred_effects).all()):
        logger.error("Found non-finite values in effects matrices")
        checks.append(False)
    else:
        checks.append(True)
    
    # 数据范围检查 (log-normalized数据通常在合理范围内)
    real_range = (real_effects.min(), real_effects.max())
    pred_range = (pred_effects.min(), pred_effects.max())
    
    if abs(real_range[0]) > 50 or abs(real_range[1]) > 50:
        logger.warning(f"Real effects range seems unusual: {real_range}")
        
    if abs(pred_range[0]) > 50 or abs(pred_range[1]) > 50:
        logger.warning(f"Pred effects range seems unusual: {pred_range}")
    
    all_passed = all(checks)
    if all_passed:
        logger.info("Perturbation data validation passed")
    else:
        logger.error("Perturbation data validation failed")
        
    return all_passed


def format_benchmark_results(results: dict) -> str:
    """
    格式化基准测试结果为可读字符串
    """
    if "error" in results:
        return f"Benchmark Error: {results['error']}"
    
    lines = []
    lines.append("=== VCC GPU Acceleration Benchmark ===")
    lines.append(f"Data Shape: {results.get('data_shape', 'N/A')}")
    lines.append(f"Device: {results.get('device', 'N/A')}")
    lines.append("")
    lines.append(f"GPU Time: {results.get('gpu_time_avg', 0):.4f}s")
    lines.append(f"CPU Time: {results.get('cpu_time_avg', 0):.4f}s") 
    lines.append(f"Speedup: {results.get('speedup', 0):.1f}x")
    lines.append("")
    
    if results.get('speedup', 0) > 2:
        lines.append("✅ Significant acceleration achieved!")
    elif results.get('speedup', 0) > 1:
        lines.append("✨ Moderate acceleration achieved")
    else:
        lines.append("⚠️  GPU slower than CPU (may need optimization)")
    
    return "\n".join(lines)


def memory_usage_estimate(n_perts: int, n_genes: int) -> dict:
    """
    估算GPU内存使用量
    
    Args:
        n_perts: 扰动数量
        n_genes: 基因数量
        
    Returns:
        内存使用估算信息
    """
    # 假设使用float32 (4 bytes per element)
    bytes_per_element = 4
    
    # 主要矩阵
    real_effects_mb = (n_perts * n_genes * bytes_per_element) / (1024 ** 2)
    pred_effects_mb = (n_perts * n_genes * bytes_per_element) / (1024 ** 2)
    
    # 中间计算矩阵 (distance matrices etc.)
    temp_matrices_mb = (n_perts * n_perts * bytes_per_element) / (1024 ** 2)
    
    # 总内存 + 20% 缓冲
    total_mb = (real_effects_mb + pred_effects_mb + temp_matrices_mb) * 1.2
    
    return {
        "real_effects_mb": real_effects_mb,
        "pred_effects_mb": pred_effects_mb, 
        "temp_matrices_mb": temp_matrices_mb,
        "total_estimated_mb": total_mb,
        "total_estimated_gb": total_mb / 1024,
        "data_shape": f"{n_perts}x{n_genes}",
        "recommendation": (
            "Should fit comfortably in GPU memory" if total_mb < 4000 
            else "May require large GPU memory" if total_mb < 8000
            else "Requires high-end GPU with >8GB memory"
        )
    }