"""
PyTorch GPU加速的VCC指标计算
专门针对MAE和Discrimination Score进行优化
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TorchVCCMetrics:
    """PyTorch GPU加速的VCC指标计算器"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化GPU计算器
        
        Args:
            device: 指定设备 ('cuda', 'cpu' 或 None自动选择)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"TorchVCCMetrics initialized on device: {self.device}")
        
    def mae_batch(
        self, 
        pred_effects: np.ndarray, 
        real_effects: np.ndarray,
        perturbation_names: np.ndarray
    ) -> Dict[str, float]:
        """
        批量计算MAE - GPU加速版本
        
        Args:
            pred_effects: (n_perts, n_genes) 预测的扰动效应
            real_effects: (n_perts, n_genes) 真实的扰动效应  
            perturbation_names: 扰动名称数组
            
        Returns:
            Dict[pert_name, mae_score] 每个扰动的MAE分数
        """
        # 转换为PyTorch tensors并移到GPU
        pred_tensor = torch.from_numpy(pred_effects).float().to(self.device)
        real_tensor = torch.from_numpy(real_effects).float().to(self.device)
        
        with torch.no_grad():  # 节省GPU内存
            # 批量计算MAE: |pred - real|
            abs_diff = torch.abs(pred_tensor - real_tensor)  # (n_perts, n_genes)
            mae_values = torch.mean(abs_diff, dim=1)  # (n_perts,) 对基因维度求平均
            
            # 转换回CPU numpy
            mae_cpu = mae_values.cpu().numpy()
        
        # 构建结果字典
        return {str(perturbation_names[i]): float(mae_cpu[i]) for i in range(len(perturbation_names))}
    
    def discrimination_score_l1_batch(
        self,
        real_effects: np.ndarray,
        pred_effects: np.ndarray, 
        perturbation_names: np.ndarray,
        exclude_target_genes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        批量计算Discrimination Score L1 - GPU加速版本
        
        Args:
            real_effects: (n_perts, n_genes) 真实的扰动效应矩阵
            pred_effects: (n_perts, n_genes) 预测的扰动效应矩阵
            perturbation_names: 扰动名称数组
            exclude_target_genes: 可选，需要排除的目标基因索引
            
        Returns:
            Dict[pert_name, discrimination_score] 每个扰动的区分度分数
        """
        # 转换为PyTorch tensors
        real_tensor = torch.from_numpy(real_effects).float().to(self.device)
        pred_tensor = torch.from_numpy(pred_effects).float().to(self.device)
        
        n_perts, n_genes = real_tensor.shape
        norm_ranks = {}
        
        with torch.no_grad():
            # 如果需要排除目标基因
            if exclude_target_genes is not None:
                include_mask = torch.ones(n_genes, dtype=torch.bool, device=self.device)
                exclude_indices = torch.from_numpy(exclude_target_genes).long().to(self.device)
                include_mask[exclude_indices] = False
                real_tensor = real_tensor[:, include_mask]
                pred_tensor = pred_tensor[:, include_mask]
                
            # 对每个扰动计算discrimination score
            for p_idx in range(n_perts):
                # 当前预测效应
                pred_single = pred_tensor[p_idx].unsqueeze(0)  # (1, n_genes)
                
                # 计算与所有真实效应的L1距离
                distances = torch.sum(torch.abs(real_tensor - pred_single), dim=1)  # (n_perts,)
                
                # 排序获得排名
                sorted_indices = torch.argsort(distances)
                
                # 找到当前扰动在排序中的位置
                rank_position = torch.where(sorted_indices == p_idx)[0][0].item()
                
                # 标准化排名 (越小的rank越好，所以用1减去标准化的rank)
                norm_rank = rank_position / n_perts
                discrimination_score = 1.0 - norm_rank
                
                norm_ranks[str(perturbation_names[p_idx])] = float(discrimination_score)
        
        return norm_ranks
    
    def batch_vcc_metrics(
        self,
        real_effects: np.ndarray,
        pred_effects: np.ndarray,
        perturbation_names: np.ndarray,
        exclude_target_genes: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        一次性计算所有GPU适合的VCC指标
        
        Returns:
            {
                'mae': {pert_name: mae_score, ...},
                'discrimination_score_l1': {pert_name: disc_score, ...}
            }
        """
        logger.info(f"Computing VCC metrics on GPU for {len(perturbation_names)} perturbations")
        
        results = {}
        
        # 计算MAE
        mae_scores = self.mae_batch(pred_effects, real_effects, perturbation_names)
        results['mae'] = mae_scores
        
        # 计算Discrimination Score L1
        disc_scores = self.discrimination_score_l1_batch(
            real_effects, pred_effects, perturbation_names, exclude_target_genes
        )
        results['discrimination_score_l1'] = disc_scores
        
        logger.info("GPU metrics computation completed")
        return results
    
    def get_device_info(self) -> Dict[str, str]:
        """获取当前设备信息"""
        info = {"device": str(self.device)}
        
        if self.device.type == 'cuda':
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
                "allocated_memory": f"{torch.cuda.memory_allocated() / 1e9:.3f}GB"
            })
        
        return info
    
    def benchmark_vs_cpu(
        self, 
        n_perts: int = 150, 
        n_genes: int = 20000,
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        性能基准测试：比较GPU vs CPU的计算时间
        
        Args:
            n_perts: 扰动数量
            n_genes: 基因数量  
            num_runs: 测试运行次数
            
        Returns:
            性能统计信息
        """
        import time
        
        # 生成测试数据
        real_effects = np.random.randn(n_perts, n_genes).astype(np.float32)
        pred_effects = np.random.randn(n_perts, n_genes).astype(np.float32)
        pert_names = np.array([f"PERT_{i:03d}" for i in range(n_perts)])
        
        # GPU测试
        gpu_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.batch_vcc_metrics(real_effects, pred_effects, pert_names)
            gpu_times.append(time.time() - start_time)
        
        # CPU测试 (简单实现作为对比)
        cpu_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self._cpu_baseline(real_effects, pred_effects, pert_names)
            cpu_times.append(time.time() - start_time)
        
        gpu_avg = np.mean(gpu_times)
        cpu_avg = np.mean(cpu_times)
        speedup = cpu_avg / gpu_avg
        
        return {
            "gpu_time_avg": gpu_avg,
            "cpu_time_avg": cpu_avg, 
            "speedup": speedup,
            "data_shape": f"{n_perts}x{n_genes}",
            "device": str(self.device)
        }
    
    def _cpu_baseline(self, real_effects, pred_effects, pert_names):
        """CPU基准实现 - 用于性能对比"""
        results = {}
        
        # 简单的CPU MAE实现
        mae_scores = {}
        for i, pert in enumerate(pert_names):
            mae_scores[str(pert)] = np.mean(np.abs(pred_effects[i] - real_effects[i]))
        results['mae'] = mae_scores
        
        # 简单的CPU discrimination score实现
        disc_scores = {}
        for p_idx in range(len(pert_names)):
            distances = []
            for r_idx in range(len(pert_names)):
                dist = np.sum(np.abs(real_effects[r_idx] - pred_effects[p_idx]))
                distances.append(dist)
            
            sorted_indices = np.argsort(distances)
            rank = np.where(sorted_indices == p_idx)[0][0]
            norm_rank = rank / len(pert_names)
            disc_scores[str(pert_names[p_idx])] = 1.0 - norm_rank
            
        results['discrimination_score_l1'] = disc_scores
        return results