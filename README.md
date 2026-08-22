# AlexNet FashionMNIST 分类

基于 PyTorch 实现的 AlexNet，用于 FashionMNIST 十分类。项目包含模型定义、训练、测试和数据加载性能测试脚本。

## 使用方法

```bash
pip install torch torchvision matplotlib pandas torchsummary
python model_train.py
python model_test.py
```

首次运行会自动下载数据集；训练权重保存为 `best_model.pth`，训练曲线保存在 `result_figures/`。运行环境建议 Python 3.8+、PyTorch 1.10+。
