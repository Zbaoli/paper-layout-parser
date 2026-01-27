# PDF Document Layout Detection

使用 DocLayout-YOLO 对 PDF 文件进行文档布局检测，识别标题、段落、表格、图片、公式等布局元素。

## 功能特性

- DocLayout-YOLO 深度学习模型
- Mac M系列芯片 MPS 加速 / NVIDIA CUDA 加速
- 10 类布局元素检测
- JSON 格式结果输出
- 可视化标注图像

## 检测类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | Title | 标题 |
| 1 | Plain-Text | 正文 |
| 2 | Abandon | 废弃内容 |
| 3 | Figure | 图片 |
| 4 | Figure-Caption | 图片说明 |
| 5 | Table | 表格 |
| 6 | Table-Caption | 表格说明 |
| 7 | Table-Footnote | 表格脚注 |
| 8 | Isolate-Formula | 独立公式 |
| 9 | Formula-Caption | 公式说明 |

## 安装

```bash
# 使用 uv 安装依赖
uv sync

# 或安装开发依赖
uv sync --extra dev
```

## 使用方法

```bash
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

## 项目结构

```
paper-layout-parser/
├── pyproject.toml        # 项目配置
├── config/
│   └── config.yaml       # 运行配置
├── data/
│   ├── papers/           # 输入 PDF
│   ├── images/           # 转换后的图像
│   └── results/
│       ├── json/         # 检测结果
│       └── visualizations/  # 可视化图像
├── src/
│   ├── pdf_converter.py     # PDF 转图像
│   ├── layout_detector.py   # 布局检测
│   ├── result_processor.py  # 结果处理
│   └── visualizer.py        # 可视化
└── main.py               # 主程序
```

## 输出示例

JSON 结果格式:

```json
{
  "pdf_name": "example.pdf",
  "total_pages": 10,
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
    "by_class": {"Title": 10, "Plain-Text": 80, "Table": 5}
  }
}
```
