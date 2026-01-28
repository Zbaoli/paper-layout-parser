# Caption Matching Benchmark 生成记录

**日期**: 2026-01-28

## 概述

本文档记录了使用 `data/papers/` 目录下的所有论文生成图表和注释匹配 benchmark 的完整流程。

## 执行步骤

### 步骤 1: PDF 检测与提取

```bash
uv run python main.py --extract
```

**结果**:
- 处理了 27 个 PDF 文件
- 总页数: 646 页
- 总检测数: 8261 个
- 提取的图表: 295 个 Figure, 230 个 Table

### 步骤 2: VLM 标注 (生成 Ground Truth)

```bash
uv run python -m src.benchmark annotate-batch \
  --input data/output \
  --vlm openai \
  --skip-existing \
  --concurrent-docs 3 \
  --concurrent-pages 5
```

**结果**:
- 成功标注 27 个文档
- 使用 OpenAI GPT-4V (通过兼容 API)
- 并发处理: 3 个文档 × 5 个页面

### 步骤 3: 构建 Benchmark 数据集

```bash
uv run python -m src.benchmark build \
  --input "data/output/*/caption_annotations.json" \
  --output benchmark/caption-matching \
  --name caption-matching-v1 \
  --version 1.0.0
```

**结果**:
- 创建了 `dataset.json` 清单文件
- 复制了 27 个文档的标注文件到 benchmark 目录

### 步骤 4: 验证数据集

```bash
uv run python -m src.benchmark validate --dataset benchmark/caption-matching
```

**结果**: 验证通过，27/27 文档有效

### 步骤 5: 运行评估

```bash
uv run python -m src.benchmark evaluate \
  --dataset benchmark/caption-matching \
  --format both
```

## 评估结果

### 总体指标

| 指标 | 值 |
|------|-----|
| Precision | 72.00% |
| Recall | 9.73% |
| F1 Score | 17.14% |

### 详细计数

| 类型 | 数量 |
|------|------|
| True Positives | 18 |
| False Positives | 7 |
| False Negatives | 167 |

### 按类型分析

| 类型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Figure | 100% | 19.63% | 32.81% |
| Table | 100% | 29.09% | 45.07% |

### 结果分析

1. **高 Precision**: 匹配算法做出的匹配通常是正确的（72%，部分类型 100%）
2. **低 Recall**: 漏掉了大量 VLM 识别出的正确匹配（~10%）
3. **改进方向**:
   - 优化 caption matching 的空间距离阈值
   - 改进匹配策略以提高召回率

## 输出结构

```
benchmark/caption-matching/
├── dataset.json              # 数据集清单
├── annotations/              # Ground truth 标注
│   └── {doc_name}/
│       ├── caption_annotations.json
│       └── extraction_metadata.json
└── results/
    ├── eval_report.json      # JSON 评估报告
    └── eval_report.md        # Markdown 评估报告
```

## 代码改进

在本次工作中，对 VLM 标注模块进行了并发优化：

1. **页面级并发** (`src/vlm_annotator/annotator.py`):
   - 添加 `max_workers` 参数
   - 使用 `ThreadPoolExecutor` 并发处理页面
   - 预提取 caption 文本以避免线程安全问题

2. **文档级并发** (`src/benchmark.py`):
   - 添加 `--concurrent-docs` 和 `--concurrent-pages` CLI 参数
   - 使用 `ThreadPoolExecutor` 并发处理多个文档

## 命令参考

```bash
# 完整工作流
uv run python main.py --extract
uv run python -m src.benchmark annotate-batch --input data/output --vlm openai --skip-existing
uv run python -m src.benchmark build --input "data/output/*/caption_annotations.json" --output benchmark/caption-matching
uv run python -m src.benchmark validate --dataset benchmark/caption-matching
uv run python -m src.benchmark evaluate --dataset benchmark/caption-matching --format both
```
