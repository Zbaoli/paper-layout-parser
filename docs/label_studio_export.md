# Label Studio 导出工具

## 概述

`scripts/export_to_label_studio.py` 将 VLM 标注结果导出为 Label Studio 格式，支持人工校正图表-标注匹配关系。

## 输出文件

导出目录（默认 `label_studio_export/`）包含：

```
label_studio_export/
├── tasks.json              # Label Studio 任务文件
├── labeling_config.xml     # 标注界面配置
└── images/                 # 图片文件（仅 local 存储模式）
    └── {pdf_name}/
        └── page_0001.png
```

## 存储后端

支持两种存储模式：

### Local 存储（默认）

图片复制到本地 `images/` 目录，通过 Label Studio 本地文件服务访问。

```bash
uv run python scripts/export_to_label_studio.py \
    --input data/output \
    --output label_studio_export
```

`tasks.json` 中的图片路径格式：
```
/data/local-files/?d=images/{pdf_name}/page_0001.png
```

启动 Label Studio 时需配置本地文件服务：
```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/path/to/label_studio_export
label-studio start
```

### MinIO 存储

图片上传到 MinIO，适用于云端部署或多人协作。

```bash
uv run python scripts/export_to_label_studio.py \
    --input data/output \
    --output label_studio_export \
    --storage minio \
    --bucket label-studio
```

`tasks.json` 中的图片路径格式：
```
http://localhost:9000/label-studio/{pdf_name}/page_0001.png
```

MinIO 配置通过环境变量设置（见 `.env.example`）：
```
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
```

## 常见问题

### Q: tasks.json 中图片路径是 local 格式，但我想用 MinIO

确保运行时指定了 `--storage minio` 参数：

```bash
# 错误：未指定存储后端，默认使用 local
uv run python scripts/export_to_label_studio.py --input data/output

# 正确：显式指定 minio 存储
uv run python scripts/export_to_label_studio.py --input data/output --storage minio
```

运行时终端会显示当前使用的存储后端：
```
Storage backend: minio
MinIO bucket: label-studio
```
