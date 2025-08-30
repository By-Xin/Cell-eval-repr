"""
异构计算协调器
GPU计算: MAE + Discrimination Score L1
CPU计算: Overlap at N (使用官方实现)
"""

import numpy as np
import polars as pl
from typing import Dict, Any, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# 导入官方cell-eval模块 (不修改，只使用)
from cell_eval.metrics._de import de_overlap_metric
from cell_eval._types import DEComparison, initialize_de_comparison

# 导入我们的GPU模块
from .torch_metrics import TorchVCCMetrics

logger = logging.getLogger(__name__)


class HybridVCCEvaluator:
    """
    异构VCC评估器：智能分配GPU和CPU任务
    - GPU (PyTorch): MAE + Discrimination Score L1  
    - CPU (官方实现): Overlap at N
    """
    
    def __init__(
        self, 
        gpu_device: Optional[str] = None,
        enable_gpu: bool = True,
        fallback_to_cpu: bool = True
    ):
        """
        初始化异构评估器
        
        Args:
            gpu_device: GPU设备名 ('cuda', 'cuda:0', etc.)
            enable_gpu: 是否启用GPU加速
            fallback_to_cpu: GPU不可用时是否回退到CPU
        """
        self.enable_gpu = enable_gpu
        self.fallback_to_cpu = fallback_to_cpu
        
        # 初始化GPU计算器
        if enable_gpu:
            try:
                self.gpu_metrics = TorchVCCMetrics(device=gpu_device)
                self.gpu_available = True
                logger.info("GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}")
                if fallback_to_cpu:
                    self.gpu_available = False
                    logger.info("Falling back to CPU-only mode")
                else:
                    raise e
        else:
            self.gpu_available = False
            logger.info("GPU acceleration disabled")
    
    def compute_vcc_metrics(
        self,
        real_effects: np.ndarray,
        pred_effects: np.ndarray, 
        perturbation_names: np.ndarray,
        de_real: Optional[pl.DataFrame] = None,
        de_pred: Optional[pl.DataFrame] = None,
        overlap_k: Optional[int] = None,
        exclude_target_genes: bool = True
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        异构计算VCC的三个核心指标
        
        Args:
            real_effects: (n_perts, n_genes) 真实扰动效应
            pred_effects: (n_perts, n_genes) 预测扰动效应  
            perturbation_names: 扰动名称数组
            de_real: 真实DE结果 (用于overlap计算)
            de_pred: 预测DE结果 (用于overlap计算) 
            overlap_k: overlap计算的top-k参数
            exclude_target_genes: 是否在discrimination中排除目标基因
            
        Returns:
            (个体指标字典, 聚合指标字典)
        """
        logger.info("Starting hybrid VCC metrics computation")
        start_time = time.time()
        
        # 存储所有结果
        individual_results = {}
        computation_stats = {}
        
        # === GPU任务：并行计算MAE和Discrimination Score ===
        if self.gpu_available:
            logger.info("Computing MAE + Discrimination Score on GPU...")
            gpu_start = time.time()
            
            # 准备目标基因排除列表
            target_gene_indices = None
            if exclude_target_genes:
                target_gene_indices = self._get_target_gene_indices(perturbation_names)
            
            # GPU批量计算
            gpu_results = self.gpu_metrics.batch_vcc_metrics(
                real_effects=real_effects,
                pred_effects=pred_effects, 
                perturbation_names=perturbation_names,
                exclude_target_genes=target_gene_indices
            )
            
            individual_results.update(gpu_results)
            computation_stats['gpu_time'] = time.time() - gpu_start
            
        else:
            # CPU回退实现
            logger.info("Computing MAE + Discrimination Score on CPU (fallback)...")
            cpu_start = time.time()
            individual_results.update(self._cpu_fallback_anndata_metrics(
                real_effects, pred_effects, perturbation_names
            ))
            computation_stats['cpu_fallback_time'] = time.time() - cpu_start
        
        # === CPU任务：Overlap计算 (使用官方实现) ===
        if de_real is not None and de_pred is not None:
            logger.info("Computing Overlap metrics on CPU...")
            cpu_start = time.time()
            
            overlap_results = self._compute_overlap_cpu(
                de_real=de_real, 
                de_pred=de_pred,
                k=overlap_k
            )
            individual_results['overlap_at_N'] = overlap_results
            computation_stats['cpu_overlap_time'] = time.time() - cpu_start
        
        # === 计算聚合指标 ===
        aggregated = self._aggregate_results(individual_results)
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid computation completed in {total_time:.3f}s")
        
        # 添加性能统计
        aggregated['_computation_stats'] = computation_stats
        aggregated['_total_time'] = total_time
        
        return individual_results, aggregated
    
    def _compute_overlap_cpu(
        self, 
        de_real: pl.DataFrame, 
        de_pred: pl.DataFrame,
        k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        使用官方cell-eval实现计算overlap指标
        """
        try:
            # 使用官方的DE比较初始化
            de_comparison = initialize_de_comparison(real=de_real, pred=de_pred)
            
            # 使用官方的overlap计算函数
            overlap_results = de_overlap_metric(
                data=de_comparison,
                k=k,  # None表示使用所有基因
                metric="overlap"
            )
            
            return overlap_results
            
        except Exception as e:
            logger.error(f"CPU Overlap computation failed: {e}")
            return {}
    
    def _get_target_gene_indices(self, perturbation_names: np.ndarray) -> np.ndarray:
        """
        获取需要在discrimination score中排除的目标基因索引
        这里简化实现，实际使用时需要根据基因名称匹配
        """
        # 简化版本：假设扰动名就是基因名
        # 实际实现时需要根据您的数据格式进行适配
        target_indices = []
        for i, pert_name in enumerate(perturbation_names):
            # 这里需要根据实际情况将扰动名映射到基因索引
            # target_indices.append(gene_name_to_index_mapping.get(pert_name, -1))
            pass
        return np.array(target_indices) if target_indices else None
    
    def _cpu_fallback_anndata_metrics(
        self,
        real_effects: np.ndarray,
        pred_effects: np.ndarray, 
        perturbation_names: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        CPU回退实现：当GPU不可用时的backup计算
        """
        logger.warning("Using CPU fallback for anndata metrics")
        
        results = {}
        
        # CPU MAE实现
        mae_scores = {}
        for i, pert in enumerate(perturbation_names):
            mae_scores[str(pert)] = float(np.mean(np.abs(pred_effects[i] - real_effects[i])))
        results['mae'] = mae_scores
        
        # CPU Discrimination Score实现 (简化版)
        disc_scores = {}
        n_perts = len(perturbation_names)
        
        for p_idx in range(n_perts):
            # 计算与所有真实效应的L1距离
            distances = []
            for r_idx in range(n_perts):
                dist = np.sum(np.abs(real_effects[r_idx] - pred_effects[p_idx]))
                distances.append(dist)
            
            # 排序并找到排名
            sorted_indices = np.argsort(distances)
            rank = np.where(sorted_indices == p_idx)[0][0]
            norm_rank = rank / n_perts
            
            disc_scores[str(perturbation_names[p_idx])] = 1.0 - norm_rank
            
        results['discrimination_score_l1'] = disc_scores
        
        return results
    
    def _aggregate_results(self, individual_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        聚合个体指标得到总体分数
        """
        aggregated = {}
        
        for metric_name, pert_scores in individual_results.items():
            if metric_name.startswith('_'):  # 跳过元数据
                continue
                
            if isinstance(pert_scores, dict) and pert_scores:
                # 计算平均值
                scores = list(pert_scores.values())
                aggregated[f"{metric_name}_mean"] = float(np.mean(scores))
                aggregated[f"{metric_name}_std"] = float(np.std(scores))
                aggregated[f"{metric_name}_min"] = float(np.min(scores))
                aggregated[f"{metric_name}_max"] = float(np.max(scores))
        
        # VCC最终分数 (三个指标的平均)
        vcc_components = []
        if 'mae_mean' in aggregated:
            # MAE需要转换：越小越好 -> 越大越好
            mae_normalized = 1.0 / (1.0 + aggregated['mae_mean'])  # 简单归一化
            vcc_components.append(mae_normalized)
        
        if 'discrimination_score_l1_mean' in aggregated:
            vcc_components.append(aggregated['discrimination_score_l1_mean'])
            
        if 'overlap_at_N_mean' in aggregated:
            vcc_components.append(aggregated['overlap_at_N_mean'])
        
        if vcc_components:
            aggregated['vcc_final_score'] = float(np.mean(vcc_components))
        
        return aggregated
    
    def benchmark_performance(
        self, 
        n_perts: int = 150,
        n_genes: int = 20000,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        性能基准测试：评估异构计算的加速效果
        """
        if not self.gpu_available:
            logger.warning("GPU not available, cannot run benchmark")
            return {"error": "GPU not available"}
            
        return self.gpu_metrics.benchmark_vs_cpu(n_perts, n_genes, num_runs)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息和配置
        """
        info = {
            "gpu_available": self.gpu_available,
            "enable_gpu": self.enable_gpu,
            "fallback_to_cpu": self.fallback_to_cpu
        }
        
        if self.gpu_available:
            info.update(self.gpu_metrics.get_device_info())
        
        return info