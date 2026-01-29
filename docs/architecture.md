# PDF Layout Parser 架构文档

> 版本: 2.1.0

## 概述

基于 DocLayout-YOLO 的 PDF 文档布局检测系统。将 PDF 页面转换为图像，检测布局元素（标题、段落、表格、图片、公式等）。

## 处理流程

```
PDF → 图像转换 → 布局检测 → JSON 结果 → 可视化 → (可选) 图表提取
     (PyMuPDF)   (YOLO)     (结构化)    (标注)     (裁剪+匹配标题)
```

## 模块结构

```
src/doclayout/               # 核心包 (pip install doclayout)
├── __init__.py              # 公共 API
├── core/                    # 核心处理
│   ├── pdf_converter.py     # PDF → PNG
│   ├── layout_detector.py   # YOLO 检测
│   ├── result_processor.py  # 结果处理
│   └── figure_extractor.py  # 图表提取
├── matching/                # 标题匹配
│   ├── caption_matcher.py   # 空间 proximity 匹配算法
│   └── types.py             # 数据类型定义
└── visualization/           # 可视化
    ├── renderer.py          # 统一渲染器 (策略模式)
    ├── styles.py            # 标签策略、颜色
    └── legend.py            # 图例生成

benchmarks/                  # 评测工具 (不打包)
├── caption_matching/        # 标题匹配评测
│   ├── cli.py               # CLI 入口 (所有命令)
│   ├── dataset.py           # 标注数据集
│   ├── evaluator.py         # 单文档评估器
│   ├── batch.py             # 批量评估器
│   ├── manifest.py          # 数据集清单
│   ├── builder.py           # 数据集构建器
│   ├── metrics.py           # 共享指标计算
│   └── reporter.py          # 报告生成
└── tools/
    └── vlm_annotator/       # VLM 标注工具
        ├── annotator.py     # 标注器
        └── *.py             # API 客户端 (ollama, openai, anthropic)
```

## 公共 API

```python
from doclayout import (
    # 核心
    PDFConverter,              # PDF 转图像
    create_detector,           # 创建检测器 (工厂函数)
    DocLayoutDetector,         # 检测器类
    Detection,                 # 检测结果
    ResultProcessor,           # 结果处理
    FigureTableExtractor,      # 图表提取

    # 匹配
    CaptionMatcher,            # 标题匹配器
    SearchDirection,           # 搜索方向枚举
    ExtractedItem,             # 提取项
    ExtractionResult,          # 提取结果

    # 可视化
    BoundingBoxRenderer,       # 边界框渲染器
    create_visualizer,         # 创建可视化器
    ColorPalette,              # 颜色配置
    LabelStrategy,             # 标签策略接口
    ClassNameLabelStrategy,    # 类名标签策略
    NumberedLabelStrategy,     # 编号标签策略
    LegendRenderer,            # 图例渲染器
    Visualizer,                # 别名 (= BoundingBoxRenderer)
)
```

## 检测类别

DocLayout-YOLO 检测 10 类元素：

| 类别 | 说明 |
|------|------|
| Title | 标题 |
| Plain-Text | 正文 |
| Figure | 图片 |
| Figure-Caption | 图片标题 |
| Table | 表格 |
| Table-Caption | 表格标题 |
| Table-Footnote | 表格脚注 |
| Isolate-Formula | 独立公式 |
| Formula-Caption | 公式标题 |
| Abandon | 弃用区域 |

## 命令行

```bash
# 主程序
uv run python main.py --single-pdf input.pdf --extract

# 评测工具
uv run python -m benchmarks.caption_matching evaluate
uv run python -m benchmarks.caption_matching annotate --input data/output/paper1 --vlm ollama
```

## 配置

`config/config.yaml` 包含：
- 模型设置 (路径、置信度阈值)
- 设备选择 (mps/cuda/cpu)
- 路径映射
- 可视化颜色 (BGR 格式)

## 数据目录

```
data/
├── papers/                  # 输入 PDF
├── output/{pdf_name}/       # 检测输出
│   ├── pages/               # 转换的页面图像
│   ├── annotated/           # 标注可视化
│   ├── extractions/         # 提取的图表
│   │   ├── figures/
│   │   ├── tables/
│   │   └── extraction_metadata.json
│   └── result.json          # 检测结果
└── benchmark/               # 评测数据集
    └── caption-matching/
        ├── dataset.json
        └── annotations/
```
