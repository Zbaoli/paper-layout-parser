# YOLOv8 PDF Document Layout Detection

使用 YOLOv8 对 PDF 文件进行文档布局检测，识别标题、段落、表格、图片、公式等布局元素。

## 功能特性

- 支持 DocLayout-YOLO 和 YOLOv8-DocLayNet 双模型
- Mac M系列芯片 MPS 加速
- 11 类布局元素检测
- JSON 格式结果输出
- 可视化标注图像

## 检测类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | Caption | 图表说明 |
| 1 | Footnote | 脚注 |
| 2 | Formula | 公式 |
| 3 | List-item | 列表项 |
| 4 | Page-footer | 页脚 |
| 5 | Page-header | 页眉 |
| 6 | Picture | 图片 |
| 7 | Section-header | 章节标题 |
| 8 | Table | 表格 |
| 9 | Text | 正文 |
| 10 | Title | 标题 |

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
uv run python main.py --single-pdf data/papers/YOLOv11.pdf

# 切换模型
uv run python main.py --model doclayout  # DocLayout-YOLO (默认)
uv run python main.py --model yolov8     # YOLOv8

# 跳过可视化
uv run python main.py --no-visualize

# 指定设备
uv run python main.py --device mps   # Mac M系列
uv run python main.py --device cpu   # CPU
```

## 项目结构

```
yolo_project/
├── pyproject.toml        # 项目配置
├── config/
│   └── config.yaml       # 运行配置
├── data/
│   ├── papers/           # 输入 PDF
│   ├── images/           # 转换后的图像
│   └── results/
│       ├── json/         # 检测结果
│       └── visualizations/  # 可视化图像
├── models/               # 预训练模型
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
    "by_class": {"Title": 10, "Text": 80, "Table": 5}
  }
}
```
