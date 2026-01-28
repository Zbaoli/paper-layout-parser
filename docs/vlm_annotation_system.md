# VLM 辅助标注图表-标注对应关系系统

## 概述

本系统使用视觉语言模型（VLM）自动标注图表与图注/表注的对应关系，生成 ground truth 数据集，用于评估现有 `CaptionMatcher` 算法的准确性。

## 系统架构

### 新增模块

```
src/
├── vlm_annotator/                    # VLM 标注模块
│   ├── __init__.py                   # 模块导出
│   ├── base.py                       # VLM 客户端抽象基类
│   ├── openai_client.py              # OpenAI GPT-4o 实现
│   ├── anthropic_client.py           # Anthropic Claude 实现
│   ├── ollama_client.py              # 本地 Ollama 实现（免费）
│   ├── annotator.py                  # 标注器核心逻辑
│   ├── image_renderer.py             # 图像渲染（绘制编号标注框）
│   └── prompts.py                    # VLM 提示词模板
│
├── caption_matching/                 # 标注匹配评估模块
│   ├── __init__.py                   # 模块导出
│   ├── dataset.py                    # 标注数据集管理
│   └── evaluator.py                  # 匹配算法评估器
```

### 工作流程

```
1. 运行检测
   uv run python main.py --single-pdf paper.pdf --extract
                ↓
2. VLM 标注（生成 ground truth）
   uv run python -m src.benchmark annotate --input data/output/paper --vlm openai
                ↓
3. 评估 CaptionMatcher
   uv run python -m src.benchmark evaluate-caption --ground-truth ... --detection ...
```

## CLI 命令

### 安装依赖

```bash
# 安装 VLM 依赖
uv sync --extra vlm
```

### 配置 API 密钥

```bash
# 复制模板
cp .env.example .env

# 编辑 .env 填入 API 密钥
# OPENAI_API_KEY=sk-xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
```

### 生成 VLM 标注

```bash
# 使用 OpenAI
uv run python -m src.benchmark annotate \
    --input data/output/paper_name \
    --vlm openai \
    --pdf data/papers/paper.pdf  # 可选，用于提取图注文本

# 使用本地 Ollama（免费）
uv run python -m src.benchmark annotate \
    --input data/output/paper_name \
    --vlm ollama \
    --model llava:13b

# 使用 Anthropic Claude
uv run python -m src.benchmark annotate \
    --input data/output/paper_name \
    --vlm anthropic
```

### 评估 CaptionMatcher

```bash
uv run python -m src.benchmark evaluate-caption \
    --ground-truth data/output/paper_name/caption_annotations.json \
    --detection data/output/paper_name/extractions/extraction_metadata.json \
    --confidence 0.7  # 可选，ground truth 置信度阈值
```

## VLM 标注流程详解

### Step 1: 读取检测结果

从 `result.json` 中读取 YOLO 检测到的 Figure/Table/Caption。

### Step 2: 渲染带编号的标注图像

在页面图像上绘制带编号的边界框：
- 绿色框 F1, F2... = Figure
- 蓝色框 T1, T2... = Table
- 橙色框 C1, C2... = Caption

标注图像保存在 `vlm_annotated/` 目录。

### Step 3: 调用 VLM 分析

发送带标注的图像和提示词给 VLM，获取匹配关系。

### Step 4: 生成标注数据集

输出格式示例：
```json
{
  "pdf_name": "paper.pdf",
  "annotator": "OpenAIClient:gpt-4o",
  "pages": [{
    "page_number": 1,
    "matches": [{
      "figure_id": "fig_01_01",
      "figure_bbox": {"x1": 100, "y1": 200, "x2": 500, "y2": 600},
      "caption_id": "cap_01_01",
      "caption_bbox": {"x1": 100, "y1": 610, "x2": 500, "y2": 650},
      "confidence": 0.95,
      "reasoning": "C1 is directly below F1 and starts with 'Figure 1'"
    }],
    "unmatched_figures": [],
    "unmatched_captions": []
  }]
}
```

## 评估指标

评估器计算以下指标：

| 指标 | 说明 |
|------|------|
| Precision | 预测匹配中正确的比例 |
| Recall | 实际匹配中被找到的比例 |
| F1 Score | Precision 和 Recall 的调和平均 |
| True Positives | 正确的匹配 |
| False Positives | 错误的匹配（预测有匹配但实际没有） |
| False Negatives | 遗漏的匹配（实际有匹配但预测没有） |

## 实验结果

### 测试文档：YOLOv11.pdf

使用 Qwen2.5-VL-72B 模型进行标注，结果如下：

| 页面 | 元素 | VLM 标注 | CaptionMatcher |
|------|------|----------|----------------|
| 2 | Table 1 | ✅ 匹配 | ❌ 未匹配 |
| 3 | Figure 1 | ✅ 匹配 | ✅ 匹配 |
| 6 | Table 2 | ✅ 匹配 | ❌ 未匹配 |
| 7 | Figure 2 | ✅ 匹配 | ✅ 匹配 |

评估结果：
- **Precision**: 100%（预测的都是对的）
- **Recall**: 25%（只找到 4 个匹配中的 1 个）
- **F1 Score**: 40%

### 发现的问题

1. **CaptionMatcher 算法局限**：
   - 只搜索图表**下方**的图注
   - 学术论文中表格标题通常在表格**上方**
   - 导致 Table 匹配率为 0%

2. **LLM JSON 输出问题**：
   - VLM 返回的 JSON 可能有尾随逗号
   - 已在解析器中添加 `_fix_json_trailing_commas()` 方法修复

## 成本控制

| 方案 | 成本 | 适用场景 |
|------|------|----------|
| Ollama (llava) | 免费 | 大批量标注、本地测试 |
| OpenAI (gpt-4o) | ~$0.01/页 | 高精度标注、复核 |
| Anthropic (claude) | ~$0.01/页 | 高精度标注、复核 |

推荐：先用 Ollama 标注，抽样用 GPT-4o/Claude 复核验证质量。

## 后续改进建议

1. **改进 CaptionMatcher**：支持同时搜索图表上方和下方的图注
2. **改进评估器**：使用 bbox 重叠匹配而不是 ID 字符串匹配
3. **添加人工复核界面**：对低置信度匹配进行人工确认
