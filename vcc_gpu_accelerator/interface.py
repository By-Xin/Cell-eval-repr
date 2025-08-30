"""
非侵入式接口层
提供与官方cell-eval完全兼容的API，但内部使用GPU加速
"""

import numpy as np
import polars as pl
import anndata as ad
from typing import Dict, Any, Optional, Tuple
import logging

# 导入官方cell-eval类 (只使用，不修改)
from cell_eval import MetricsEvaluator
from cell_eval._types import PerturbationAnndataPair
from cell_eval._evaluator import _build_anndata_pair, _build_de_comparison

# 导入我们的异构计算器
from .hybrid_evaluator import HybridVCCEvaluator
from .utils import extract_perturbation_effects

logger = logging.getLogger(__name__)


class AcceleratedMetricsEvaluator:
    """
    GPU加速的MetricsEvaluator包装器
    
    完全兼容官方API，但内部使用异构计算加速VCC指标
    对于其他指标，透明地调用官方实现
    """
    
    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        # GPU加速相关参数
        enable_gpu_acceleration: bool = True,
        gpu_device: Optional[str] = None,
        fallback_to_cpu: bool = True,
        # 其他官方参数
        **kwargs
    ):
        """
        初始化加速评估器
        
        Args:
            adata_pred, adata_real: 与官方MetricsEvaluator相同
            de_pred, de_real: 与官方MetricsEvaluator相同  
            control_pert, pert_col: 与官方MetricsEvaluator相同
            enable_gpu_acceleration: 是否启用GPU加速
            gpu_device: 指定GPU设备
            fallback_to_cpu: GPU失败时是否回退
            **kwargs: 传递给官方MetricsEvaluator的其他参数
        """
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.control_pert = control_pert
        self.pert_col = pert_col
        
        # 初始化官方评估器 (用于非VCC指标)
        self.official_evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real, 
            de_pred=de_pred,
            de_real=de_real,
            control_pert=control_pert,
            pert_col=pert_col,
            **kwargs
        )
        
        # 初始化GPU加速器
        if enable_gpu_acceleration:
            try:
                self.gpu_evaluator = HybridVCCEvaluator(
                    gpu_device=gpu_device,
                    enable_gpu=True,
                    fallback_to_cpu=fallback_to_cpu
                )
                self.gpu_enabled = True
                logger.info("GPU acceleration initialized successfully")
            except Exception as e:
                logger.warning(f"GPU acceleration initialization failed: {e}")
                self.gpu_enabled = False
        else:
            self.gpu_enabled = False
            
        # 缓存数据以避免重复计算
        self._cached_effects = None
        self._cached_perturbations = None
    
    def compute(
        self, 
        profile: str = "vcc",
        use_gpu_for_vcc: bool = True,
        **kwargs
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        计算指标 - 智能选择GPU或CPU实现
        
        Args:
            profile: 指标配置文件
            use_gpu_for_vcc: 对VCC指标是否使用GPU加速
            **kwargs: 传递给官方compute方法的参数
            
        Returns:
            (个体结果DataFrame, 聚合结果DataFrame) - 与官方API完全兼容
        """
        logger.info(f"Computing metrics with profile: {profile}")
        
        # 如果是VCC profile且GPU可用，使用GPU加速
        if (profile == "vcc" and 
            self.gpu_enabled and 
            use_gpu_for_vcc):
            
            return self._compute_vcc_accelerated(**kwargs)
        else:
            # 使用官方实现
            logger.info("Using official CPU implementation")
            return self.official_evaluator.compute(profile=profile, **kwargs)
    
    def _compute_vcc_accelerated(self, **kwargs) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        GPU加速的VCC指标计算
        """
        logger.info("Computing VCC metrics with GPU acceleration")
        
        try:
            # 提取扰动效应矩阵
            real_effects, pred_effects, pert_names = self._get_perturbation_effects()
            
            # 获取DE数据 (用于overlap计算)
            de_real = getattr(self.official_evaluator, 'de_comparison', None)
            de_pred = None
            if hasattr(self.official_evaluator, 'de_comparison'):
                de_real = self.official_evaluator.de_comparison.real.data
                de_pred = self.official_evaluator.de_comparison.pred.data
            
            # 异构计算VCC指标
            individual_results, aggregated_results = self.gpu_evaluator.compute_vcc_metrics(
                real_effects=real_effects,
                pred_effects=pred_effects,
                perturbation_names=pert_names,
                de_real=de_real,
                de_pred=de_pred,
                overlap_k=None  # 使用所有基因
            )
            
            # 转换为与官方兼容的DataFrame格式
            results_df = self._format_results_as_dataframe(individual_results, pert_names)
            agg_df = self._format_aggregated_as_dataframe(aggregated_results)
            
            return results_df, agg_df
            
        except Exception as e:
            logger.error(f"GPU acceleration failed: {e}, falling back to CPU")
            return self.official_evaluator.compute(profile="vcc", **kwargs)
    
    def _get_perturbation_effects(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从AnnData对象中提取扰动效应矩阵
        
        Returns:
            (real_effects, pred_effects, perturbation_names)
            shapes: (n_perts, n_genes), (n_perts, n_genes), (n_perts,)
        """
        if self._cached_effects is not None:
            return self._cached_effects
        
        logger.info("Extracting perturbation effects from AnnData objects")
        
        # 使用工具函数提取效应矩阵
        real_effects, pred_effects, pert_names = extract_perturbation_effects(
            anndata_pair=self.official_evaluator.anndata_pair,
            control_pert=self.control_pert
        )
        
        # 缓存结果
        self._cached_effects = (real_effects, pred_effects, pert_names)
        return self._cached_effects
    
    def _format_results_as_dataframe(
        self,
        individual_results: Dict[str, Dict[str, float]], 
        perturbation_names: np.ndarray
    ) -> pl.DataFrame:
        """
        将GPU计算结果格式化为与官方兼容的DataFrame格式
        """
        rows = []
        
        for pert_name in perturbation_names:
            row = {"perturbation": str(pert_name)}
            
            for metric_name, pert_scores in individual_results.items():
                if metric_name.startswith('_'):
                    continue
                if str(pert_name) in pert_scores:
                    row[metric_name] = pert_scores[str(pert_name)]
            
            rows.append(row)
        
        return pl.DataFrame(rows)
    
    def _format_aggregated_as_dataframe(
        self, 
        aggregated_results: Dict[str, float]
    ) -> pl.DataFrame:
        """
        将聚合结果格式化为与官方兼容的DataFrame格式
        """
        # 模拟官方的聚合格式 (类似pl.describe的输出)
        data = []
        
        # 添加统计行
        for stat in ["mean", "std", "min", "max"]:
            row = {"statistic": stat}
            
            # VCC指标
            for metric in ["mae", "discrimination_score_l1", "overlap_at_N"]:
                key = f"{metric}_{stat}"
                if key in aggregated_results:
                    row[metric] = aggregated_results[key]
                else:
                    row[metric] = None
            
            data.append(row)
        
        # 添加最终VCC分数行
        if 'vcc_final_score' in aggregated_results:
            final_row = {
                "statistic": "vcc_final_score",
                "value": aggregated_results['vcc_final_score']
            }
            data.append(final_row)
        
        return pl.DataFrame(data)
    
    def benchmark_acceleration(self, **kwargs) -> Dict[str, Any]:
        """
        基准测试：比较GPU vs CPU的性能
        """
        if not self.gpu_enabled:
            return {"error": "GPU acceleration not available"}
            
        return self.gpu_evaluator.benchmark_performance(**kwargs)
    
    def get_acceleration_info(self) -> Dict[str, Any]:
        """
        获取加速系统信息
        """
        info = {
            "gpu_acceleration_enabled": self.gpu_enabled,
            "enable_gpu_acceleration": self.enable_gpu_acceleration
        }
        
        if self.gpu_enabled:
            info.update(self.gpu_evaluator.get_system_info())
        
        return info
    
    # 代理其他方法到官方实现
    def __getattr__(self, name):
        """
        代理模式：未定义的方法自动转发到官方evaluator
        """
        return getattr(self.official_evaluator, name)