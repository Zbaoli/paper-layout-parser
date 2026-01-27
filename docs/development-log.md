# PDF 文档布局检测 - 开发日志

## 项目概述

使用 DocLayout-YOLO 对 PDF 文件进行文档布局检测，识别标题、段落、表格、图片、公式等布局元素。

### 环境
- **设备**: Mac M系列芯片 (使用 MPS 加速) / NVIDIA GPU (CUDA)
- **模型**: DocLayout-YOLO
- **包管理**: uv

---

## 项目结构

```
paper-layout-parser/
├── pyproject.toml           # 项目配置和依赖
├── uv.lock                  # 依赖锁定文件
├── config/
│   └── config.yaml          # 运行配置
├── data/
│   ├── papers/              # 输入 PDF
│   ├── images/              # PDF 转换后的图像
│   └── results/
│       ├── json/            # 检测结果 JSON
│       └── visualizations/  # 可视化标注图像
├── models/                  # 预训练模型
├── src/
│   ├── __init__.py          # 包初始化
│   ├── pdf_converter.py     # PDF 转图像模块
│   ├── layout_detector.py   # 布局检测模块 (核心)
│   ├── result_processor.py  # 结果处理模块
│   └── visualizer.py        # 可视化模块
├── main.py                  # 主程序入口
└── docs/                    # 文档目录
```

---

## 实现的模块

### 1. PDF 转图像模块 (`src/pdf_converter.py`)

使用 PyMuPDF 将 PDF 转换为 PNG 图像。

**主要功能**:
- 支持自定义 DPI (默认 200)
- 单页转换和批量转换
- 获取 PDF 元数据

### 2. 布局检测模块 (`src/layout_detector.py`)

核心检测模块，使用 DocLayout-YOLO 模型。

**类结构**:
- `BaseLayoutDetector`: 抽象基类
- `DocLayoutDetector`: DocLayout-YOLO 实现
- `create_detector()`: 工厂函数

### 3. 结果处理模块 (`src/result_processor.py`)

**功能**:
- 保存 JSON 格式检测结果
- 计算统计信息 (按类别、按页面)
- 生成批量处理汇总报告

**JSON 输出结构**:
```json
{
  "pdf_name": "example.pdf",
  "total_pages": 10,
  "model": "doclayout-yolo",
  "pages": [{
    "page_number": 1,
    "detections": [{
      "class_name": "Title",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 50, "x2": 500, "y2": 100}
    }]
  }],
  "statistics": {
    "total_detections": 150,
    "by_class": {"Title": 10, "Plain-Text": 80}
  }
}
```

### 4. 可视化模块 (`src/visualizer.py`)

**功能**:
- 绘制检测框和类别标签
- 不同类别使用不同颜色
- 生成图例图像

---

## 检测类别

### DocLayout-YOLO DocStructBench (10 类)

| ID | 类别 | 说明 |
|----|------|------|
| 0 | Title | 标题 |
| 1 | Plain-Text | 正文文本 |
| 2 | Abandon | 废弃内容 |
| 3 | Figure | 图片 |
| 4 | Figure-Caption | 图片说明 |
| 5 | Table | 表格 |
| 6 | Table-Caption | 表格说明 |
| 7 | Table-Footnote | 表格脚注 |
| 8 | Isolate-Formula | 独立公式 |
| 9 | Formula-Caption | 公式说明 |

---

## 问题修复记录

### 问题: 正文被错误分类为 Footnote

**现象**: 运行检测后，大量正文文本被分类为 "Footnote"

**统计结果 (修复前)**:
```
Footnote: 79
Caption: 24
Formula: 14
```

**原因分析**:

代码中硬编码了 DocLayNet 的类别映射：
```python
CLASS_NAMES = {
    0: "Caption",
    1: "Footnote",  # 错误！模型中 1 是 "plain text"
    ...
}
```

但 DocLayout-YOLO 模型使用不同的类别 ID：
```python
# 模型实际类别
{
    0: "title",
    1: "plain text",  # 被错误映射为 Footnote
    ...
}
```

**解决方案**:

修改 `layout_detector.py`，从模型文件动态读取类别名称：

```python
def load_model(self):
    ...
    self.model = YOLOv10(model_file)
    # 使用模型自带的类别名称
    self.class_names = {
        idx: name.replace("_", "-").title().replace(" ", "-")
        for idx, name in self.model.names.items()
    }
```

**修复后统计**:
```
Plain-Text: 79
Title: 24
Abandon: 14
```

### 问题: Plain-Text 颜色太亮

**现象**: 可视化图像中 Plain-Text 的黄色太亮，难以看清

**解决方案**:

修改 `visualizer.py` 中的颜色定义：
```python
# 修改前
"Plain-Text": (0, 255, 255),    # 亮黄色

# 修改后
"Plain-Text": (0, 180, 0),      # 深绿色
```

---

## 使用方法

```bash
# 安装依赖
uv sync

# 处理所有 PDF
uv run python main.py

# 处理单个 PDF
uv run python main.py --single-pdf data/papers/example.pdf

# 跳过可视化
uv run python main.py --no-visualize

# 指定设备
uv run python main.py --device mps   # Mac M系列
uv run python main.py --device cuda  # NVIDIA GPU
uv run python main.py --device cpu   # CPU
```

---

## Git 提交记录

按照单一职责原则，分 7 次提交：

```
8703986 添加主程序入口
ac612f5 添加可视化模块
d8c3d24 添加结果处理模块
d3eba84 添加布局检测模块
3674537 添加 PDF 转图像模块
dbf9163 添加 src 包初始化模块
84dea8e 初始化项目配置
```

---

## 依赖列表

```toml
dependencies = [
    "pymupdf>=1.24.0",
    "doclayout-yolo>=0.0.2",
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "huggingface-hub>=0.20.0",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
]
```

---

## 测试验证

使用 `YOLOv11.pdf` 进行测试：

```
Processing: YOLOv11.pdf
  Converting PDF to images...
  Detecting layout elements in 9 pages...
  Saved results to: data/results/json/YOLOv11.json
  Generating visualizations...
  Saved 9 visualization images
  Total detections: 125
```

**输出目录**:
- 图像: `data/images/YOLOv11/`
- JSON: `data/results/json/YOLOv11.json`
- 可视化: `data/results/visualizations/YOLOv11/`
