# VCC Baseline 模型详解与复现指南

本文档详细介绍了Virtual Cell Challenge (VCC) 比赛中的baseline模型机制，包括实现原理、评分系统和完整的复现代码。

## 目录
- [1. VCC Baseline 模型原理](#1-vcc-baseline-模型原理)
- [2. VCC 评分机制详解](#2-vcc-评分机制详解)
- [3. 完整的Baseline复现代码](#3-完整的baseline复现代码)
- [4. 关键注意事项](#4-关键注意事项)

## 1. VCC Baseline 模型原理

### 1.1 核心思想

**VCC的baseline模型是一个极简的"平均预测"模型**，其核心理念是：
- 对于任何新的扰动，预测结果就是**所有训练集扰动的平均效应**
- 不考虑具体的基因靶点或扰动类型
- 代表了"最朴素"的预测方法，用作性能比较的底线

### 1.2 算法流程

基于 `src/cell_eval/_baseline.py:153-196` 的实现：

```python
def _build_pert_baseline(adata: ad.AnnData, as_delta: bool = False) -> NDArray[np.float64]:
    # 1. 计算每个扰动的平均表达谱
    pert_means = compute_perturbation_means(adata)  # 按扰动分组求均值
    
    # 2. 排除control组，只使用扰动数据
    pert_mask = names != control_pert
    pert_matrix = pert_means[pert_mask]  # 所有扰动的均值矩阵
    
    # 3. 计算全局平均
    if as_delta:
        # Delta模式：先算每个扰动相对control的变化，再平均
        delta = pert_matrix - control_mean
        return delta.mean(axis=0)  # 平均delta向量
    else:
        # 直接模式：直接对所有扰动均值求平均
        return pert_matrix.mean(axis=0)  # 平均表达向量
```

## 2. VCC 评分机制详解

### 2.1 VCC 官方指标

基于 `src/cell_eval/_pipeline/_runner.py:21-25`：

```python
VCC_METRICS = [
    "mae",                    # Mean Absolute Error (越小越好)
    "discrimination_score_l1", # 扰动区分能力 (越大越好)  
    "overlap_at_N",           # 差异表达基因重叠度 (越大越好)
]
```

### 2.2 Scaling算法

基于 `src/cell_eval/_score.py:110-121`，VCC使用两种标准化策略：

**对于"越小越好"的指标 (如MAE)**:
```python
def _calc_norm_by_zero(user_score, baseline_score):
    return 1.0 - (user_score / baseline_score)
```
- 如果用户模型完美(score=0)，得分=1.0
- 如果用户模型等于baseline，得分=0.0
- 如果用户模型更差，得分<0（会被clip到0）

**对于"越大越好"的指标 (如discrimination_score, overlap)**:
```python
def _calc_norm_by_one(user_score, baseline_score):
    return (user_score - baseline_score) / (1 - baseline_score)
```
- 如果用户模型完美(score=1)，得分=1.0  
- 如果用户模型等于baseline，得分=0.0
- 如果用户模型更差，得分<0（会被clip到0）

### 2.3 最终评分计算

```python
final_score = np.mean([scaled_mae, scaled_discrimination, scaled_overlap])
```

## 3. 完整的Baseline复现代码

### 3.1 使用cell-eval命令行工具

```bash
# 生成baseline模型
cell-eval baseline \
    -a /path/to/training_data.h5ad \
    -o ./vcc_baseline.h5ad \
    -O ./vcc_baseline_de.csv \
    --control-pert "non-targeting" \
    --pert-col "target" \
    --num-threads 8

# 运行评估（获取baseline分数）
cell-eval run \
    -ap ./vcc_baseline.h5ad \
    -ar /path/to/real_test_data.h5ad \
    --profile vcc \
    --num-threads 8

# 计算相对于baseline的标准化分数  
cell-eval score \
    --user-input ./user_model_agg_results.csv \
    --base-input ./vcc_baseline_agg_results.csv
```

### 3.2 Python API 实现

```python
import anndata as ad
import numpy as np
from cell_eval import MetricsEvaluator, build_base_mean_adata, score_agg_metrics

def create_vcc_baseline(training_adata_path: str, output_path: str = "vcc_baseline.h5ad"):
    """
    创建VCC baseline模型
    
    Args:
        training_adata_path: 训练数据路径
        output_path: baseline模型输出路径
    
    Returns:
        baseline_adata: AnnData对象，包含baseline预测
    """
    
    # 读取训练数据
    adata_train = ad.read_h5ad(training_adata_path)
    
    # 生成baseline模型（使用扰动平均）
    baseline_adata = build_base_mean_adata(
        adata=adata_train,
        control_pert="non-targeting",  # 根据您的数据调整
        pert_col="target",             # 根据您的数据调整
        as_delta=False,                # 直接使用均值，不使用delta
        output_path=output_path
    )
    
    print(f"Baseline模型已保存到: {output_path}")
    return baseline_adata

def evaluate_against_baseline(user_adata_path: str, 
                            real_adata_path: str,
                            baseline_adata_path: str):
    """
    评估用户模型相对于baseline的性能
    
    Args:
        user_adata_path: 用户模型预测结果路径
        real_adata_path: 真实数据路径  
        baseline_adata_path: baseline模型路径
    """
    
    # 评估用户模型
    print("评估用户模型...")
    user_evaluator = MetricsEvaluator(
        adata_pred=user_adata_path,
        adata_real=real_adata_path,
        control_pert="non-targeting",
        pert_col="target",
        outdir="./user_eval"
    )
    user_results, user_agg = user_evaluator.compute(profile="vcc")
    
    # 评估baseline模型
    print("评估baseline模型...")
    baseline_evaluator = MetricsEvaluator(
        adata_pred=baseline_adata_path,
        adata_real=real_adata_path,
        control_pert="non-targeting", 
        pert_col="target",
        outdir="./baseline_eval"
    )
    baseline_results, baseline_agg = baseline_evaluator.compute(profile="vcc")
    
    # 计算标准化分数
    print("计算标准化分数...")
    scores = score_agg_metrics(
        results_user=user_agg,
        results_base=baseline_agg,
        output="./vcc_scores.csv"
    )
    
    print("VCC评分结果:")
    print(scores)
    
    # 提取最终分数
    final_score = scores.filter(pl.col("metric") == "avg_score")["from_baseline"][0]
    print(f"\n最终VCC分数: {final_score:.4f}")
    
    return scores

# 使用示例
if __name__ == "__main__":
    # 1. 创建baseline
    baseline_adata = create_vcc_baseline(
        training_adata_path="path/to/training_data.h5ad",
        output_path="vcc_baseline.h5ad"
    )
    
    # 2. 评估您的模型
    scores = evaluate_against_baseline(
        user_adata_path="path/to/your_predictions.h5ad",
        real_adata_path="path/to/test_data.h5ad", 
        baseline_adata_path="vcc_baseline.h5ad"
    )
```

### 3.3 手动实现baseline算法

如果您想手动实现baseline逻辑：

```python
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

def manual_baseline_prediction(training_adata, test_perturbations, 
                             control_pert="non-targeting", pert_col="target"):
    """
    手动实现VCC baseline算法
    """
    
    # 1. 计算训练集中每个扰动的平均表达谱
    pert_means = {}
    unique_perts = training_adata.obs[pert_col].unique()
    
    for pert in unique_perts:
        if pert != control_pert:  # 排除control
            mask = training_adata.obs[pert_col] == pert
            pert_data = training_adata[mask]
            
            # 计算该扰动的平均表达
            if issparse(pert_data.X):
                mean_expr = np.array(pert_data.X.mean(axis=0))[0]
            else:
                mean_expr = pert_data.X.mean(axis=0)
            
            pert_means[pert] = mean_expr
    
    # 2. 计算全局平均（baseline预测）
    all_means = np.stack(list(pert_means.values()))
    baseline_prediction = all_means.mean(axis=0)
    
    # 3. 为测试扰动生成预测
    predictions = {}
    for test_pert in test_perturbations:
        predictions[test_pert] = baseline_prediction.copy()
    
    return predictions, baseline_prediction

# 使用示例
training_data = ad.read_h5ad("training.h5ad")
test_perts = ["GENE1", "GENE2", "GENE3"]  # 您的测试扰动

predictions, baseline = manual_baseline_prediction(training_data, test_perts)
print(f"Baseline向量形状: {baseline.shape}")
print(f"为{len(predictions)}个测试扰动生成了预测")
```

## 4. 关键注意事项

1. **数据格式**：确保您的数据列名与代码中的参数一致（`pert_col`, `control_pert`）

2. **标准化**：Baseline会自动检测并应用log-normalization

3. **评分解释**：
   - 分数>0：您的模型比baseline好
   - 分数=0：您的模型等于baseline  
   - 分数<0：您的模型比baseline差（会被clip到0）

4. **VCC具体指标**：只使用MAE、discrimination_score_l1和overlap_at_N这三个指标

5. **文件路径**：所有路径都应该使用绝对路径以避免错误

6. **计算资源**：差异表达分析可能需要较长时间，建议使用多线程加速

## 总结

通过这个baseline，您可以客观评估自己模型的性能提升程度，这正是VCC比赛评分的核心机制。Baseline模型虽然简单，但为所有参赛者提供了一个公平的比较基准，确保评分的科学性和可靠性。

## 相关文件

- `src/cell_eval/_baseline.py`: Baseline模型核心实现
- `src/cell_eval/_score.py`: 评分和标准化算法
- `src/cell_eval/_pipeline/_runner.py`: VCC指标定义
- `src/cell_eval/_cli/_baseline.py`: 命令行接口

---

*本文档基于cell-eval代码库版本0.5.43编写*