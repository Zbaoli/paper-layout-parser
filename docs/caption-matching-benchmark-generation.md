# Caption Matching Benchmark 生成记录

**日期**: 2026-01-28

## 概述

本文档记录了使用 `data/papers/` 目录下的所有论文生成图表和注释匹配 benchmark 的完整流程，包括评估器 bug 修复和人工标注工具集成。

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
- 使用 VLM: Qwen/Qwen2.5-VL-72B-Instruct (通过 OpenAI 兼容 API)
- 并发处理: 3 个文档 × 5 个页面

### 步骤 3: 构建 Benchmark 数据集

```bash
uv run python -m src.benchmark build \
  --input "data/output/*/caption_annotations.json" \
  --output benchmark/caption-matching \
  --name caption-matching-v1 \
  --version 1.0.0
```

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

## 评估器 Bug 修复

### 问题发现

初始评估结果显示 Recall 异常低（9.73%），经分析发现评估器存在严重 bug。

### 根本原因

评估器 (`src/caption_matching/evaluator.py`) 使用错误的 ID 推断逻辑：
- 假设 `fig_04_01` → `cap_04_01`（基于序号对应）
- 但实际上 figure 和 caption 是**独立编号**的，不能直接对应

**示例**:
```
Ground Truth: fig_04_01 → cap_04_02, fig_04_03 → cap_04_01
评估器假设: fig_04_01 → cap_04_01 ❌
```

### 修复方案

改用 **bbox IoU 匹配**代替 ID 推断：
1. `_calculate_iou()` - 计算两个 bbox 的交并比
2. `_find_matching_prediction()` - 通过 figure bbox 的 IoU 找到对应的 prediction
3. `_check_caption_match()` - 通过 caption bbox 的 IoU 验证匹配是否正确

### 修复前后对比

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| **F1 Score** | 17.14% | **77.42%** | +351% |
| **Recall** | 9.73% | **71.35%** | +633% |
| **Precision** | 72.00% | **84.62%** | +18% |
| **Figure F1** | 32.81% | **95.10%** | +190% |
| **Table F1** | 45.07% | **68.26%** | +51% |

## 最终评估结果

### 总体指标

| 指标 | 值 |
|------|-----|
| Precision | 84.62% |
| Recall | 71.35% |
| F1 Score | 77.42% |

### 详细计数

| 类型 | 数量 |
|------|------|
| True Positives | 132 |
| False Positives | 24 |
| False Negatives | 53 |

### 按类型分析

| 类型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Figure | 100% | 90.65% | 95.10% |
| Table | 100% | 51.82% | 68.26% |

## VLM 标注的局限性

VLM 生成的 ground truth 并非完美：
- VLM 本身会出错（视觉理解有误、幻觉问题）
- 复杂布局识别困难（多栏、嵌套图表）
- 标注不一致

**更可靠的方案**: 使用 VLM 预标注 + 人工校正

## 人工标注工具

### 导出到 Label Studio

```bash
python scripts/export_to_label_studio.py \
  --input data/output \
  --output label_studio_export
```

**输出**:
- `label_studio_export/tasks.json` - 341 个标注任务
- `label_studio_export/labeling_config.xml` - 标注界面配置
- `label_studio_export/images/` - 页面图像

### 使用 Label Studio 修正标注

```bash
# 安装
pip install label-studio

# 启动
label-studio start
```

1. 创建项目
2. Settings → Labeling Interface → 粘贴 `labeling_config.xml` 内容
3. Settings → Cloud Storage → 添加本地存储指向 `images/` 目录
4. Import → 导入 `tasks.json`
5. 在界面中修正 figure-caption 匹配关系

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

label_studio_export/          # Label Studio 导出
├── tasks.json                # 标注任务
├── labeling_config.xml       # 界面配置
└── images/                   # 页面图像
```

## 代码改进总结

1. **VLM 标注并发优化** (`src/vlm_annotator/annotator.py`, `src/benchmark.py`):
   - 页面级并发 + 文档级并发
   - 添加 `--concurrent-docs` 和 `--concurrent-pages` CLI 参数

2. **评估器 Bug 修复** (`src/caption_matching/evaluator.py`):
   - 改用 bbox IoU 匹配代替 ID 推断
   - F1 从 17% 提升到 77%

3. **Label Studio 导出工具** (`scripts/export_to_label_studio.py`):
   - 支持将 VLM 标注导出为 Label Studio 格式
   - 便于人工校正

## 命令参考

```bash
# 完整工作流
uv run python main.py --extract
uv run python -m src.benchmark annotate-batch --input data/output --vlm openai --skip-existing
uv run python -m src.benchmark build --input "data/output/*/caption_annotations.json" --output benchmark/caption-matching
uv run python -m src.benchmark validate --dataset benchmark/caption-matching
uv run python -m src.benchmark evaluate --dataset benchmark/caption-matching --format both

# 人工标注修正
python scripts/export_to_label_studio.py --input data/output --output label_studio_export
label-studio start
```
