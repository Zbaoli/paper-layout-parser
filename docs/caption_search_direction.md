# CaptionMatcher 改进：支持表格标题在上方查找

## 问题背景

在学术论文中，表格标题通常位于表格**上方**，而图片标题通常位于图片**下方**。原有的 `CaptionMatcher` 只在元素下方查找 caption，导致表格标题无法正确匹配。

## 解决方案

添加搜索方向配置，支持 Figure 默认向下找，Table 默认向上找。

## 代码改动

### 1. `src/figure_table_extractor.py`

#### 新增 `SearchDirection` 枚举类

```python
class SearchDirection(Enum):
    BELOW = "below"   # Caption 在下方（Figure 默认）
    ABOVE = "above"   # Caption 在上方（Table 默认）
    BOTH = "both"     # 双向查找
```

#### 修改 `CaptionMatcher`

- `__init__`: 添加 `figure_search_direction` 和 `table_search_direction` 参数
- `_get_vertical_distance`: 返回 `(distance, is_valid_direction)` 元组，支持方向判断
- `_is_valid_match`: 使用方向感知的距离计算
- `match_items_to_captions`: 添加 `item_type` 参数，根据类型选择搜索方向

#### 修改 `FigureTableExtractor`

- `__init__`: 添加方向参数并传递给 `CaptionMatcher`
- 调用 `match_items_to_captions` 时传递 `item_type`

### 2. `config/config.yaml`

新增配置项：

```yaml
extraction:
  caption_search:
    figure_direction: "below"  # Figure 默认向下找
    table_direction: "above"   # Table 默认向上找
```

### 3. `main.py`

- 导入 `SearchDirection` 枚举
- 读取 `caption_search` 配置
- 传递方向参数给 `FigureTableExtractor`

## 核心算法

```python
def _get_vertical_distance(self, item_bbox, caption_bbox, search_direction):
    dist_below = caption_bbox["y1"] - item_bbox["y2"]  # caption 在下方
    dist_above = item_bbox["y1"] - caption_bbox["y2"]  # caption 在上方

    if search_direction == SearchDirection.BELOW:
        return abs(dist_below), dist_below >= 0
    elif search_direction == SearchDirection.ABOVE:
        return abs(dist_above), dist_above >= 0
    else:  # BOTH
        if dist_below >= 0:
            return dist_below, True
        elif dist_above >= 0:
            return dist_above, True
        return 0.0, True  # 重叠
```

## 向后兼容性

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `figure_search_direction` | `BELOW` | 保持原有行为 |
| `table_search_direction` | `ABOVE` | 修复表格标题匹配 |
| `item_type` | `"figure"` | 向后兼容 |

## 测试结果

使用 DocLayNet.pdf 进行测试：

| 类型 | 检测数量 | 匹配到标题 | 匹配率 |
|------|----------|------------|--------|
| Figure | 6 | 6 | 100% |
| Table | 5 | 3 | 60% |

**关键验证**：
- 3 个匹配成功的表格标题全部在表格**上方**（距离 56-61px）
- 6 个匹配成功的图片标题全部在图片**下方**
- 2 个未匹配的表格是因为模型未检测到对应的 Table-Caption

## 使用方法

```bash
# 运行检测和提取
uv run python main.py --single-pdf data/papers/test.pdf --extract

# 自定义配置（修改 config/config.yaml）
extraction:
  caption_search:
    figure_direction: "both"   # 双向查找
    table_direction: "above"   # 只在上方查找
```

## 配置选项

| 选项 | 说明 |
|------|------|
| `below` | 只在元素下方查找 caption |
| `above` | 只在元素上方查找 caption |
| `both` | 双向查找，优先匹配更近的 caption |
