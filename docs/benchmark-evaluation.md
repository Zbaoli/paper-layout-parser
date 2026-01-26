# DocLayNet 基准评测报告

## 概述

对 DocLayout-YOLO 和 YOLOv8-DocLayNet 两个模型在 DocLayNet 数据集上进行基准评测，比较其在文档布局检测任务上的表现。

---

## 评测模块

### 新增文件

```
src/benchmark.py    # 基准评测模块
```

### 功能特性

- 支持 DocLayNet 完整数据集 (28GB) 和 DocLayNet-small (~50MB)
- 自动检测数据集格式 (COCO / per-image JSON)
- 计算 Precision / Recall / F1 指标
- 生成 Markdown 对比报告

---

## 数据集

### DocLayNet-small

| 属性 | 值 |
|------|-----|
| 来源 | [pierreguillou/DocLayNet-small](https://huggingface.co/datasets/pierreguillou/DocLayNet-small) |
| 大小 | ~50MB |
| 测试集 | 49 张图片 |
| 标注格式 | Per-image JSON |

### 下载命令

```bash
# 下载小数据集 (推荐)
uv run python -m src.benchmark download --dataset doclaynet-small

# 下载完整数据集 (28GB)
uv run python -m src.benchmark download --dataset doclaynet --split test
```

---

## 评测结果

### 测试环境

- 数据集: DocLayNet-small test (49 images)
- IoU 阈值: 0.5
- 评测类别: Picture (Figure), Table

### 结果对比

| 指标 | DocLayout-YOLO | YOLOv8-DocLayNet |
|------|----------------|------------------|
| **Figure F1** | 0.16 | **0.52** |
| **Table F1** | 0.66 | **0.91** |
| **End-to-End F1** | 0.46 | **0.72** |

### 详细指标

#### Figure Detection (Picture)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| DocLayoutDetector | 0.15 | 0.17 | 0.16 |
| YOLOv8LayoutDetector | 0.35 | 1.00 | 0.52 |

#### Table Detection

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| DocLayoutDetector | 0.74 | 0.59 | 0.66 |
| YOLOv8LayoutDetector | 0.89 | 0.94 | 0.91 |

---

## 结果分析

### 为什么 DocLayout-YOLO 表现较差？

**不是代码 Bug，是模型训练数据不同。**

| 模型 | 训练数据集 | 图片类名 |
|------|-----------|---------|
| DocLayout-YOLO | DocStructBench | `Figure` |
| YOLOv8-DocLayNet | DocLayNet | `Picture` |

### 标注风格差异

以一张包含多个小图的页面为例：

```
GT (DocLayNet):     14 个 Picture (每个小图单独标注)
DocLayout-YOLO:      1 个 Figure  (把多个小图合并成一个大区域)
YOLOv8-DocLayNet:   24 个 Picture (每个小图单独识别)
```

### 关键发现

1. **DocLayout-YOLO (DocStructBench 训练)**
   - 倾向于检测"图片区域"作为整体
   - 对小图片/logo 不敏感
   - 适合粗粒度布局分析

2. **YOLOv8-DocLayNet (DocLayNet 训练)**
   - 标注风格与 DocLayNet 一致
   - 能识别每个独立的小图片
   - 适合细粒度布局分析

---

## 使用方法

### 运行评测

```bash
# 下载数据集
uv run python -m src.benchmark download --dataset doclaynet-small

# 评测 DocLayout-YOLO
uv run python -m src.benchmark evaluate \
  --dataset data/benchmark/doclaynet-small \
  --model doclayout \
  --output data/benchmark/results/doclayout_report.json

# 评测 YOLOv8
uv run python -m src.benchmark evaluate \
  --dataset data/benchmark/doclaynet-small \
  --model yolov8 \
  --output data/benchmark/results/yolov8_report.json

# 生成对比报告
uv run python -m src.benchmark compare \
  --inputs data/benchmark/results/doclayout_report.json \
          data/benchmark/results/yolov8_report.json \
  --output data/benchmark/results/comparison.md
```

### 输出文件

```
data/benchmark/
├── doclaynet-small/          # 数据集
│   └── small_dataset/
│       └── test/
│           ├── images/       # 测试图片
│           └── annotations/  # 标注文件
└── results/
    ├── doclayout_report.json # DocLayout 评测结果
    ├── yolov8_report.json    # YOLOv8 评测结果
    └── comparison.md         # 对比报告
```

---

## 模型选择建议

| 使用场景 | 推荐模型 |
|----------|---------|
| DocLayNet 风格标注 (细粒度) | YOLOv8-DocLayNet |
| DocStructBench 风格标注 (粗粒度) | DocLayout-YOLO |
| 学术论文布局分析 | DocLayout-YOLO |
| 通用文档处理 | YOLOv8-DocLayNet |

---

## 代码实现要点

### 数据集格式自动检测

```python
def _detect_format(self) -> str:
    """检测数据集格式"""
    small_path = self.dataset_path / "small_dataset" / self.split
    if small_path.exists() and (small_path / "annotations").exists():
        return "small"  # per-image JSON 格式
    return "coco"       # COCO 格式
```

### 类名映射

```python
# DocLayout-YOLO 使用 "Figure"，DocLayNet 使用 "Picture"
if class_name == "Figure" and "Picture" in target_classes:
    pred_boxes.append(d["bbox"])
```

### IoU 计算

```python
def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0
```

---

## 总结

1. YOLOv8-DocLayNet 在 DocLayNet 数据集上表现显著优于 DocLayout-YOLO
2. 这是由于训练数据集不同导致的标注风格差异，而非代码问题
3. 选择模型时应考虑目标应用场景的标注风格需求
