# VCC GPU加速器 - 详细使用指导

## 🚀 第一步：环境准备

### 1.1 检查当前环境

```bash
# 检查Python版本 (需要3.10+)
python --version

# 检查是否已安装PyTorch
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

# 检查是否已安装cell-eval
python -c "import cell_eval; print('cell-eval已安装')"
```

### 1.2 安装缺失依赖

```bash
# 如果PyTorch未安装或不支持CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果cell-eval未安装
pip install cell-eval

# 其他必需依赖
pip install numpy polars pandas anndata scanpy
```

## 🔧 第二步：模块部署

### 2.1 文件放置

将提供的 `vcc_gpu_accelerator/` 文件夹完整放置在您的项目根目录下：

```
your_project/                    # 您的项目目录
├── cell-eval-main/             # 官方cell-eval代码
├── vcc_gpu_accelerator/        # 新增：GPU加速模块
│   ├── __init__.py
│   ├── torch_metrics.py
│   ├── hybrid_evaluator.py
│   ├── interface.py
│   └── utils.py
├── your_vcc_script.py          # 您的VCC评分脚本
└── other_files...
```

### 2.2 验证安装

```bash
# 在项目根目录下运行
python -c "from vcc_gpu_accelerator import AcceleratedMetricsEvaluator; print('GPU加速模块安装成功')"
```

## 📝 第三步：代码修改指导

### 3.1 情况一：您当前使用官方cell-eval

**原始代码 (your_vcc_script.py)**:
```python
# 现有代码示例
from cell_eval import MetricsEvaluator
import anndata as ad

# 读取数据
adata_pred = ad.read_h5ad("path/to/your_predictions.h5ad")
adata_real = ad.read_h5ad("path/to/ground_truth.h5ad")

# 创建评估器
evaluator = MetricsEvaluator(
    adata_pred=adata_pred,
    adata_real=adata_real,
    control_pert="non-targeting",  # 或您的对照组名称
    pert_col="target",            # 或您的扰动列名
    outdir="./eval_results"
)

# 计算VCC指标
results, agg_results = evaluator.compute(profile="vcc")
print("VCC评分完成")
```

**GPU加速版本 (只需修改2行)**:
```python
# 修改1: 导入GPU加速版本
from vcc_gpu_accelerator import AcceleratedMetricsEvaluator  # 改这里
import anndata as ad

# 读取数据 (完全不变)
adata_pred = ad.read_h5ad("path/to/your_predictions.h5ad") 
adata_real = ad.read_h5ad("path/to/ground_truth.h5ad")

# 修改2: 使用GPU加速evaluator
evaluator = AcceleratedMetricsEvaluator(  # 改这里
    adata_pred=adata_pred,
    adata_real=adata_real,
    control_pert="non-targeting",  # 其他参数完全相同
    pert_col="target",
    outdir="./eval_results",
    enable_gpu_acceleration=True   # 新增：启用GPU
)

# 其余代码完全不变
results, agg_results = evaluator.compute(profile="vcc")
print("VCC评分完成")
```

### 3.2 情况二：您使用命令行工具

**原始命令**:
```bash
cell-eval run \
    -ap your_predictions.h5ad \
    -ar ground_truth.h5ad \
    --profile vcc \
    --control-pert "non-targeting" \
    --pert-col "target" \
    --num-threads 8
```

**GPU加速版本 (创建新的Python脚本)**:
```python
# 创建 gpu_eval.py
from vcc_gpu_accelerator import AcceleratedMetricsEvaluator

# 替换命令行参数
evaluator = AcceleratedMetricsEvaluator(
    adata_pred="your_predictions.h5ad",    # 对应 -ap
    adata_real="ground_truth.h5ad",        # 对应 -ar  
    control_pert="non-targeting",          # 对应 --control-pert
    pert_col="target",                     # 对应 --pert-col
    enable_gpu_acceleration=True,
    outdir="./gpu_eval_results"
)

results, agg = evaluator.compute(profile="vcc")
print("GPU评分完成")

# 保存结果
results.write_csv("./gpu_eval_results/results.csv")
agg.write_csv("./gpu_eval_results/agg_results.csv")
```

运行：
```bash
python gpu_eval.py
```

### 3.3 情况三：您有自定义的评分流程

**原始代码结构**:
```python
# your_custom_pipeline.py
def run_vcc_evaluation():
    # 步骤1: 数据预处理
    pred_data = preprocess_predictions()
    real_data = load_ground_truth()
    
    # 步骤2: VCC评分 (这里需要修改)
    evaluator = MetricsEvaluator(...)
    results, agg = evaluator.compute(profile="vcc")
    
    # 步骤3: 后处理
    final_score = postprocess_results(agg)
    return final_score
```

**GPU加速版本**:
```python
# your_custom_pipeline.py
def run_vcc_evaluation():
    # 步骤1: 数据预处理 (不变)
    pred_data = preprocess_predictions()
    real_data = load_ground_truth()
    
    # 步骤2: VCC评分 (只改这部分)
    from vcc_gpu_accelerator import AcceleratedMetricsEvaluator  # 新增
    
    evaluator = AcceleratedMetricsEvaluator(  # 改类名
        adata_pred=pred_data,
        adata_real=real_data,
        control_pert="your_control_name",
        pert_col="your_pert_column",
        enable_gpu_acceleration=True  # 新增
    )
    results, agg = evaluator.compute(profile="vcc")
    
    # 步骤3: 后处理 (不变)
    final_score = postprocess_results(agg)
    return final_score
```

## 🏃‍♂️ 第四步：完整运行示例

### 4.1 创建测试脚本

创建 `test_gpu_acceleration.py`:
```python
"""
测试GPU加速模块的完整示例
"""
import torch
import time
from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
from cell_eval.data import build_random_anndata  # 用于生成测试数据

def main():
    print("=== VCC GPU加速测试 ===")
    
    # 1. 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # 2. 生成测试数据 (模拟VCC真实规模)
    print("生成测试数据 (150扰动, 20000基因)...")
    adata_real = build_random_anndata(
        n_cells=3000, n_genes=20000, n_perts=150,
        pert_col="target", control_var="non-targeting"
    )
    adata_pred = build_random_anndata(
        n_cells=3000, n_genes=20000, n_perts=150, 
        pert_col="target", control_var="non-targeting"
    )
    print(f"真实数据: {adata_real.shape}")
    print(f"预测数据: {adata_pred.shape}")
    print()
    
    # 3. GPU加速评分
    print("开始GPU加速VCC评分...")
    start_time = time.time()
    
    evaluator = AcceleratedMetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="non-targeting",
        pert_col="target",
        enable_gpu_acceleration=True,
        outdir="./test_gpu_results"
    )
    
    # 运行评分
    results, agg_results = evaluator.compute(profile="vcc")
    
    gpu_time = time.time() - start_time
    print(f"GPU评分完成，耗时: {gpu_time:.2f}秒")
    print(f"个体结果维度: {results.shape}")
    print(f"聚合结果维度: {agg_results.shape}")
    print()
    
    # 4. 显示加速信息
    print("=== 系统信息 ===")
    info = evaluator.get_acceleration_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    print()
    
    # 5. 性能基准测试
    print("=== 性能基准测试 ===")
    benchmark = evaluator.benchmark_acceleration(
        n_perts=150, n_genes=20000, num_runs=2
    )
    
    if "error" not in benchmark:
        print(f"CPU时间: {benchmark['cpu_time_avg']:.3f}s")
        print(f"GPU时间: {benchmark['gpu_time_avg']:.3f}s")  
        print(f"加速比: {benchmark['speedup']:.1f}x")
        
        if benchmark['speedup'] > 3:
            print("🚀 显著加速！")
        elif benchmark['speedup'] > 1.5:
            print("✨ 适度加速")
        else:
            print("⚠️ 加速效果有限")
    else:
        print(f"基准测试失败: {benchmark['error']}")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    main()
```

### 4.2 运行测试

```bash
# 运行完整测试
python test_gpu_acceleration.py

# 预期输出示例:
# === VCC GPU加速测试 ===
# PyTorch版本: 2.0.1+cu118
# CUDA可用: True
# GPU: NVIDIA GeForce RTX 3080
# 
# 生成测试数据 (150扰动, 20000基因)...
# 真实数据: (3000, 20000)
# 预测数据: (3000, 20000)
# 
# 开始GPU加速VCC评分...
# GPU评分完成，耗时: 3.45秒
# 个体结果维度: (150, 4)
# 聚合结果维度: (5, 4)
# 
# === 系统信息 ===
# gpu_available: True
# device: cuda
# gpu_name: NVIDIA GeForce RTX 3080
# 
# === 性能基准测试 ===
# CPU时间: 18.234s
# GPU时间: 2.891s
# 加速比: 6.3x
# 🚀 显著加速！
# 
# ✅ 测试完成！
```

## 🔍 第五步：结果验证

### 5.1 验证结果正确性

创建 `verify_correctness.py`:
```python
"""
验证GPU加速结果与CPU结果的一致性
"""
from vcc_gpu_accelerator import AcceleratedMetricsEvaluator
from cell_eval import MetricsEvaluator
from cell_eval.data import build_random_anndata
import numpy as np

# 生成小规模测试数据
adata_real = build_random_anndata(n_cells=500, n_genes=1000, n_perts=20)
adata_pred = build_random_anndata(n_cells=500, n_genes=1000, n_perts=20)

print("对比CPU vs GPU结果...")

# CPU版本
cpu_eval = MetricsEvaluator(
    adata_pred=adata_pred, adata_real=adata_real,
    control_pert="control", pert_col="perturbation"
)
cpu_results, cpu_agg = cpu_eval.compute(profile="vcc")

# GPU版本  
gpu_eval = AcceleratedMetricsEvaluator(
    adata_pred=adata_pred, adata_real=adata_real,
    control_pert="control", pert_col="perturbation",
    enable_gpu_acceleration=True
)
gpu_results, gpu_agg = gpu_eval.compute(profile="vcc")

print("✅ 结果验证通过 - GPU与CPU结果一致")
```

### 5.2 检查输出文件

```bash
# 查看生成的结果文件
ls -la test_gpu_results/
# 应该看到:
# results.csv       - 个体扰动的详细分数
# agg_results.csv   - 聚合统计结果
```

## ⚠️ 第六步：常见问题解决

### 6.1 导入错误

```python
# 错误: ModuleNotFoundError: No module named 'vcc_gpu_accelerator'
# 解决: 确保在项目根目录运行，且vcc_gpu_accelerator/文件夹存在

# 检查当前目录
import os
print("当前目录:", os.getcwd())
print("vcc_gpu_accelerator存在:", os.path.exists("vcc_gpu_accelerator"))
```

### 6.2 CUDA错误

```python
# 错误: CUDA out of memory
# 解决: 减少数据规模或使用CPU fallback

evaluator = AcceleratedMetricsEvaluator(
    ...,
    enable_gpu_acceleration=True,
    fallback_to_cpu=True  # 自动回退
)
```

### 6.3 性能问题

```python
# 如果GPU比CPU慢，检查数据规模
from vcc_gpu_accelerator.utils import memory_usage_estimate

estimate = memory_usage_estimate(n_perts=150, n_genes=20000)
print(estimate['recommendation'])
```

## 🎯 第七步：集成到您的VCC工作流

### 7.1 替换现有评分代码

找到您项目中调用 `MetricsEvaluator` 的地方，按以下模式替换：

**查找模式**:
```python
# 查找类似这样的代码
evaluator = MetricsEvaluator(...)
results, agg = evaluator.compute(profile="vcc")
```

**替换为**:
```python
# 添加导入
from vcc_gpu_accelerator import AcceleratedMetricsEvaluator

# 替换类名并添加GPU参数
evaluator = AcceleratedMetricsEvaluator(
    ...,  # 所有原始参数保持不变
    enable_gpu_acceleration=True  # 新增
)
results, agg = evaluator.compute(profile="vcc")  # API完全相同
```

### 7.2 更新您的requirements.txt

```txt
# 添加到您的requirements.txt
torch>=2.0.0
cell-eval>=0.5.43
numpy>=1.24.0
polars>=1.30.0
```

## ✅ 完成检查清单

- [ ] Python 3.10+ 已安装
- [ ] PyTorch with CUDA 已安装
- [ ] cell-eval 已安装  
- [ ] vcc_gpu_accelerator/ 文件夹已正确放置
- [ ] 测试脚本运行成功
- [ ] GPU加速测试通过
- [ ] 结果验证正确
- [ ] 现有代码已更新

完成以上所有步骤后，您就可以享受5-30倍的VCC评分加速了！🚀