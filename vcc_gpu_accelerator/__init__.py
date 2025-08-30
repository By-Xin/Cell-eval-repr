"""
VCC GPU加速器 - 独立模块
提供PyTorch GPU加速的VCC评分计算，完全不污染官方cell-eval代码
"""

from .hybrid_evaluator import HybridVCCEvaluator
from .interface import AcceleratedMetricsEvaluator
from .torch_metrics import TorchVCCMetrics

__version__ = "1.0.0"
__all__ = [
    "HybridVCCEvaluator", 
    "AcceleratedMetricsEvaluator",
    "TorchVCCMetrics"
]