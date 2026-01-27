# 增强图表提取系统设计文档

## 1. 概述

### 1.1 项目目标
将现有 YOLO PDF 布局检测项目扩展为完整的文档处理系统：
- 提取的图片和表格存入 MinIO 对象存储
- 图注、表注和 AI 生成的描述存入 Qdrant 向量数据库
- 支持语义搜索图表内容

### 1.2 技术选型
| 组件 | 选择 | 说明 |
|------|------|------|
| 对象存储 | MinIO | 本地 Docker 部署，S3 兼容 |
| 向量数据库 | Qdrant | 高性能，API 友好 |
| AI 描述生成 | OpenAI Compatible API | Silicon Flow 或自部署模型 |
| Embedding | BGE 系列 | 中文支持好 |

---

## 2. 系统架构

### 2.1 整体流程
```
┌─────────────────────────────────────────────────────────────────┐
│                        现有流程                                  │
│  PDF → PDFConverter → LayoutDetector → ResultProcessor          │
│                              ↓                                   │
│                     FigureTableExtractor                        │
│                     (提取图片/表格/图注)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        新增流程                                  │
│                     EnhancedExtractor                           │
│         ┌───────────────┼───────────────┐                       │
│         ↓               ↓               ↓                       │
│    MinIOStorage   DescriptionGen   QdrantStore                  │
│    (图片存储)     (AI描述生成)     (向量索引)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 新增模块结构
```
src/
├── storage/                    # 对象存储模块
│   ├── __init__.py
│   ├── base.py                 # 抽象存储接口
│   └── minio_storage.py        # MinIO 实现
│
├── description/                # AI 描述生成模块
│   ├── __init__.py
│   ├── base.py                 # 抽象生成器接口
│   └── openai_generator.py     # OpenAI Compatible API 实现
│
├── vectordb/                   # 向量数据库模块
│   ├── __init__.py
│   ├── embeddings.py           # Embedding 生成器
│   └── qdrant_store.py         # Qdrant 实现
│
└── pipeline/                   # 集成管道
    ├── __init__.py
    └── enhanced_extractor.py   # 整合所有功能的主类
```

---

## 3. 详细设计

### 3.1 存储模块 (src/storage/)

#### 3.1.1 抽象接口 (base.py)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class StorageResult:
    success: bool
    object_key: str              # e.g., "figures/doc1/fig_01_01.png"
    url: Optional[str] = None    # 访问 URL
    etag: Optional[str] = None   # 用于去重
    error: Optional[str] = None

class BaseStorage(ABC):
    @abstractmethod
    def upload_file(self, file_path: Path, object_key: str,
                    content_type: Optional[str] = None,
                    metadata: Optional[dict] = None) -> StorageResult:
        """上传文件"""
        pass

    @abstractmethod
    def exists(self, object_key: str) -> bool:
        """检查对象是否存在"""
        pass

    @abstractmethod
    def get_url(self, object_key: str, expires: int = 3600) -> str:
        """获取访问 URL"""
        pass

    @abstractmethod
    def delete(self, object_key: str) -> bool:
        """删除对象"""
        pass
```

#### 3.1.2 MinIO 实现 (minio_storage.py)
```python
from minio import Minio
from minio.error import S3Error
import hashlib
from pathlib import Path
from typing import Optional
import mimetypes
from datetime import timedelta

from .base import BaseStorage, StorageResult

class MinIOStorage(BaseStorage):
    def __init__(
        self,
        endpoint: str,           # "localhost:9000"
        access_key: str,         # "minioadmin"
        secret_key: str,         # "minioadmin"
        bucket_name: str = "document-extractions",
        secure: bool = False,
        auto_create_bucket: bool = True,
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket_name = bucket_name

        if auto_create_bucket:
            self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    def upload_file(self, file_path: Path, object_key: str,
                    content_type: Optional[str] = None,
                    metadata: Optional[dict] = None) -> StorageResult:
        try:
            if content_type is None:
                content_type, _ = mimetypes.guess_type(str(file_path))
                content_type = content_type or "application/octet-stream"

            result = self.client.fput_object(
                self.bucket_name,
                object_key,
                str(file_path),
                content_type=content_type,
                metadata=metadata,
            )

            return StorageResult(
                success=True,
                object_key=object_key,
                etag=result.etag,
                url=self.get_url(object_key),
            )
        except S3Error as e:
            return StorageResult(success=False, object_key=object_key, error=str(e))

    def exists(self, object_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket_name, object_key)
            return True
        except S3Error:
            return False

    def get_url(self, object_key: str, expires: int = 3600) -> str:
        return self.client.presigned_get_object(
            self.bucket_name,
            object_key,
            expires=timedelta(seconds=expires),
        )

    def delete(self, object_key: str) -> bool:
        try:
            self.client.remove_object(self.bucket_name, object_key)
            return True
        except S3Error:
            return False
```

### 3.2 描述生成模块 (src/description/)

#### 3.2.1 抽象接口 (base.py)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

@dataclass
class DescriptionResult:
    success: bool
    description: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    error: Optional[str] = None

class BaseDescriptionGenerator(ABC):
    @abstractmethod
    def generate_description(
        self,
        image_path: Path,
        item_type: str,              # "figure" or "table"
        caption: Optional[str] = None,
    ) -> DescriptionResult:
        """为图片生成 AI 描述"""
        pass
```

#### 3.2.2 OpenAI Compatible 实现 (openai_generator.py)
```python
import base64
from pathlib import Path
from typing import Optional
import httpx

from .base import BaseDescriptionGenerator, DescriptionResult

class OpenAICompatibleGenerator(BaseDescriptionGenerator):
    PROMPTS = {
        "figure": """分析这张来自科技/学术文档的图片。
请提供详细描述，包括：
1. 图片类型（图表、示意图、照片等）
2. 主要内容和元素
3. 关键观察或趋势
4. 任何重要特征或标注

图注（如有）: {caption}

请用2-4句话提供简洁但全面的描述。""",

        "table": """分析这张来自科技/学术文档的表格。
请提供详细描述，包括：
1. 表格呈现的数据内容
2. 结构（列、行、表头）
3. 关键发现或模式
4. 这个表格在文档中的作用

表注（如有）: {caption}

请用2-4句话提供简洁但全面的描述。""",
    }

    def __init__(
        self,
        api_base: str,               # "https://api.siliconflow.cn/v1"
        api_key: str,
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_tokens: int = 500,
        temperature: float = 0.3,
        timeout: float = 60.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = httpx.Client(timeout=timeout)

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(suffix, "image/png")

    def generate_description(
        self,
        image_path: Path,
        item_type: str,
        caption: Optional[str] = None,
    ) -> DescriptionResult:
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return DescriptionResult(success=False, error=f"Image not found: {image_path}")

            prompt = self.PROMPTS.get(item_type, self.PROMPTS["figure"])
            prompt = prompt.format(caption=caption or "无")

            image_data = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }]

            response = self.client.post(
                f"{self.api_base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages,
                      "max_tokens": self.max_tokens, "temperature": self.temperature},
            )
            response.raise_for_status()

            result = response.json()
            return DescriptionResult(
                success=True,
                description=result["choices"][0]["message"]["content"],
                model_used=self.model,
                tokens_used=result.get("usage", {}).get("total_tokens"),
            )
        except Exception as e:
            return DescriptionResult(success=False, error=str(e))
```

### 3.3 向量数据库模块 (src/vectordb/)

#### 3.3.1 Embedding 生成器 (embeddings.py)
```python
from abc import ABC, abstractmethod
from typing import List, Optional
import httpx

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAICompatibleEmbedding(BaseEmbedding):
    def __init__(
        self,
        api_base: str,               # "https://api.siliconflow.cn/v1"
        api_key: str,
        model: str = "BAAI/bge-large-zh-v1.5",
        timeout: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=timeout)
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.post(
            f"{self.api_base}/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        result = response.json()
        return [item["embedding"] for item in result["data"]]
```

#### 3.3.2 Qdrant 存储 (qdrant_store.py)
```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from .embeddings import BaseEmbedding

@dataclass
class VectorDocument:
    """存储到向量数据库的文档"""
    id: str
    pdf_name: str
    item_id: str                     # e.g., "fig_01_01"
    item_type: str                   # "figure" or "table"
    page_number: int
    caption: Optional[str]           # 原始图注
    ai_description: Optional[str]    # AI 生成的描述
    minio_url: Optional[str]
    minio_object_key: str
    bbox: Dict[str, float]
    created_at: str

    def to_text(self) -> str:
        """组合文本用于 embedding"""
        parts = []
        if self.caption:
            parts.append(f"图注: {self.caption}")
        if self.ai_description:
            parts.append(f"描述: {self.ai_description}")
        return " ".join(parts) if parts else f"{self.item_type} 来自第 {self.page_number} 页"


class QdrantVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "document_extractions",
        embedding: BaseEmbedding = None,
        api_key: Optional[str] = None,
    ):
        self.client = QdrantClient(host=host, port=port, api_key=api_key)
        self.collection_name = collection_name
        self.embedding = embedding
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding.dimension, distance=Distance.COSINE),
            )
            # 创建索引以加速过滤
            for field in ["pdf_name", "item_type", "item_id"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword",
                )

    def _generate_id(self, doc: VectorDocument) -> str:
        content = f"{doc.pdf_name}:{doc.item_id}:{doc.page_number}"
        return hashlib.md5(content.encode()).hexdigest()

    def exists(self, pdf_name: str, item_id: str) -> bool:
        """检查文档是否已存在"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="pdf_name", match=models.MatchValue(value=pdf_name)),
                models.FieldCondition(key="item_id", match=models.MatchValue(value=item_id)),
            ]),
            limit=1,
        )
        return len(results[0]) > 0

    def upsert(self, doc: VectorDocument) -> str:
        """插入或更新文档"""
        text = doc.to_text()
        vector = self.embedding.embed(text)
        point_id = self._generate_id(doc)

        payload = {
            "pdf_name": doc.pdf_name,
            "item_id": doc.item_id,
            "item_type": doc.item_type,
            "page_number": doc.page_number,
            "caption": doc.caption,
            "ai_description": doc.ai_description,
            "minio_url": doc.minio_url,
            "minio_object_key": doc.minio_object_key,
            "bbox": doc.bbox,
            "created_at": doc.created_at,
            "text_content": text,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        return point_id

    def search(
        self,
        query: str,
        limit: int = 10,
        pdf_name: Optional[str] = None,
        item_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """语义搜索"""
        query_vector = self.embedding.embed(query)

        filter_conditions = []
        if pdf_name:
            filter_conditions.append(
                models.FieldCondition(key="pdf_name", match=models.MatchValue(value=pdf_name))
            )
        if item_type:
            filter_conditions.append(
                models.FieldCondition(key="item_type", match=models.MatchValue(value=item_type))
            )

        search_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
        )

        return [{"id": hit.id, "score": hit.score, **hit.payload} for hit in results]

    def delete_by_pdf(self, pdf_name: str):
        """删除某个 PDF 的所有数据"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="pdf_name", match=models.MatchValue(value=pdf_name))
                ])
            ),
        )
```

### 3.4 集成管道 (src/pipeline/enhanced_extractor.py)

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..figure_table_extractor import FigureTableExtractor, ExtractedItem
from ..storage.base import BaseStorage
from ..description.base import BaseDescriptionGenerator
from ..vectordb.qdrant_store import QdrantVectorStore, VectorDocument


@dataclass
class EnhancedExtractedItem(ExtractedItem):
    """扩展的提取结果"""
    minio_url: Optional[str] = None
    minio_object_key: Optional[str] = None
    ai_description: Optional[str] = None
    vector_id: Optional[str] = None


@dataclass
class EnhancedExtractionResult:
    """完整的增强提取结果"""
    pdf_name: str
    total_pages: int
    figures: List[EnhancedExtractedItem] = field(default_factory=list)
    tables: List[EnhancedExtractedItem] = field(default_factory=list)
    processing_stats: Dict[str, Any] = field(default_factory=dict)


class EnhancedExtractor:
    """
    整合存储、描述生成和向量索引的增强提取器
    """

    def __init__(
        self,
        base_extractor: FigureTableExtractor,
        storage: Optional[BaseStorage] = None,
        description_generator: Optional[BaseDescriptionGenerator] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        skip_existing: bool = True,           # 增量处理
        generate_descriptions: bool = True,
    ):
        self.base_extractor = base_extractor
        self.storage = storage
        self.description_generator = description_generator
        self.vector_store = vector_store
        self.skip_existing = skip_existing
        self.generate_descriptions = generate_descriptions

    def _should_skip_item(self, pdf_name: str, item_id: str) -> bool:
        """检查是否应该跳过（已处理过）"""
        if not self.skip_existing:
            return False

        if self.storage:
            object_key = f"{pdf_name}/{item_id}.png"
            if self.storage.exists(object_key):
                return True

        if self.vector_store:
            if self.vector_store.exists(pdf_name, item_id):
                return True

        return False

    def process(
        self,
        pdf_path: str,
        detection_result: Dict[str, Any],
        model_type: str = "doclayout",
    ) -> EnhancedExtractionResult:
        """
        完整处理流程:
        1. 基础提取（图片/表格）
        2. 上传到 MinIO
        3. 生成 AI 描述
        4. 索引到 Qdrant
        """
        # 1. 基础提取
        base_result = self.base_extractor.extract_from_detection_results(
            pdf_path=pdf_path,
            detection_result=detection_result,
            model_type=model_type,
        )

        pdf_name = base_result.pdf_name.replace(".pdf", "")
        extraction_dir = self.base_extractor.output_dir / pdf_name

        stats = {
            "total_figures": len(base_result.figures),
            "total_tables": len(base_result.tables),
            "uploaded_to_minio": 0,
            "descriptions_generated": 0,
            "indexed_to_qdrant": 0,
            "skipped_existing": 0,
        }

        enhanced_figures = []
        enhanced_tables = []

        # 处理图片
        for item in base_result.figures:
            enhanced = self._process_item(item, pdf_name, extraction_dir, stats)
            enhanced_figures.append(enhanced)

        # 处理表格
        for item in base_result.tables:
            enhanced = self._process_item(item, pdf_name, extraction_dir, stats)
            enhanced_tables.append(enhanced)

        return EnhancedExtractionResult(
            pdf_name=base_result.pdf_name,
            total_pages=base_result.total_pages,
            figures=enhanced_figures,
            tables=enhanced_tables,
            processing_stats=stats,
        )

    def _process_item(
        self,
        item: ExtractedItem,
        pdf_name: str,
        extraction_dir: Path,
        stats: dict,
    ) -> EnhancedExtractedItem:
        """处理单个提取项"""

        # 检查是否跳过
        if self._should_skip_item(pdf_name, item.item_id):
            stats["skipped_existing"] += 1
            return EnhancedExtractedItem(
                item_type=item.item_type,
                item_id=item.item_id,
                page_number=item.page_number,
                item_bbox=item.item_bbox,
                caption_text=item.caption_text,
                caption_bbox=item.caption_bbox,
                image_path=item.image_path,
            )

        local_path = extraction_dir / item.image_path

        # 上传到 MinIO
        minio_url, minio_key = None, None
        if self.storage:
            object_key = f"{pdf_name}/{item.item_type}s/{item.item_id}.png"
            result = self.storage.upload_file(
                file_path=local_path,
                object_key=object_key,
                content_type="image/png",
                metadata={"pdf_name": pdf_name, "item_id": item.item_id},
            )
            if result.success:
                minio_url, minio_key = result.url, object_key
                stats["uploaded_to_minio"] += 1

        # 生成 AI 描述
        ai_description = None
        if self.description_generator and self.generate_descriptions:
            result = self.description_generator.generate_description(
                image_path=local_path,
                item_type=item.item_type,
                caption=item.caption_text,
            )
            if result.success:
                ai_description = result.description
                stats["descriptions_generated"] += 1

        # 创建增强项
        enhanced = EnhancedExtractedItem(
            item_type=item.item_type,
            item_id=item.item_id,
            page_number=item.page_number,
            item_bbox=item.item_bbox,
            caption_text=item.caption_text,
            caption_bbox=item.caption_bbox,
            image_path=item.image_path,
            minio_url=minio_url,
            minio_object_key=minio_key,
            ai_description=ai_description,
        )

        # 索引到 Qdrant
        if self.vector_store:
            doc = VectorDocument(
                id="",
                pdf_name=pdf_name,
                item_id=item.item_id,
                item_type=item.item_type,
                page_number=item.page_number,
                caption=item.caption_text,
                ai_description=ai_description,
                minio_url=minio_url,
                minio_object_key=minio_key or "",
                bbox=item.item_bbox,
                created_at=datetime.now().isoformat(),
            )
            vector_id = self.vector_store.upsert(doc)
            enhanced.vector_id = vector_id
            stats["indexed_to_qdrant"] += 1

        return enhanced
```

---

## 4. 配置文件

### 4.1 config/config.yaml 新增内容

```yaml
# ============================================
# 新增配置：增强提取功能
# ============================================

# MinIO 对象存储配置
minio:
  enabled: true
  endpoint: "localhost:9000"          # MinIO 服务地址
  access_key: "minioadmin"            # 访问密钥（生产环境使用环境变量）
  secret_key: "minioadmin"            # 秘密密钥（生产环境使用环境变量）
  bucket_name: "document-extractions" # 存储桶名称
  secure: false                       # 是否使用 HTTPS
  auto_create_bucket: true            # 自动创建存储桶

# Qdrant 向量数据库配置
qdrant:
  enabled: true
  host: "localhost"
  port: 6333
  grpc_port: 6334                     # 可选：gRPC 端口
  collection_name: "document_extractions"
  api_key: null                       # 可选：Qdrant Cloud 密钥

# AI 描述生成配置
description:
  enabled: true
  provider: "openai_compatible"
  api_base: "https://api.siliconflow.cn/v1"  # Silicon Flow 或其他兼容 API
  api_key: "${SILICONFLOW_API_KEY}"          # 使用环境变量
  model: "Qwen/Qwen2-VL-7B-Instruct"         # Vision 模型
  max_tokens: 500
  temperature: 0.3
  timeout: 60.0

# Embedding 配置
embedding:
  provider: "openai_compatible"
  api_base: "https://api.siliconflow.cn/v1"
  api_key: "${SILICONFLOW_API_KEY}"
  model: "BAAI/bge-large-zh-v1.5"
  timeout: 30.0

# 增强提取设置
enhanced_extraction:
  enabled: true                       # 是否启用增强管道
  skip_existing: true                 # 跳过已处理的项目
  generate_descriptions: true         # 生成 AI 描述
  batch_size: 10                      # API 批处理大小
  delete_local_after_upload: false    # 上传后删除本地文件
```

### 4.2 环境变量 (.env)

```bash
# Silicon Flow API
SILICONFLOW_API_KEY=your_api_key_here

# 或自部署 OpenAI Compatible API
# OPENAI_API_BASE=http://localhost:8000/v1
# OPENAI_API_KEY=your_key
```

---

## 5. CLI 扩展

### 5.1 main.py 新增参数

```python
# 新增参数组
enhanced_group = parser.add_argument_group("Enhanced Extraction")

enhanced_group.add_argument(
    "--enhanced-extract",
    action="store_true",
    help="启用增强提取（MinIO + AI描述 + Qdrant）",
)
enhanced_group.add_argument(
    "--no-minio",
    action="store_true",
    help="禁用 MinIO 上传",
)
enhanced_group.add_argument(
    "--no-description",
    action="store_true",
    help="禁用 AI 描述生成",
)
enhanced_group.add_argument(
    "--no-qdrant",
    action="store_true",
    help="禁用 Qdrant 索引",
)
enhanced_group.add_argument(
    "--force-reprocess",
    action="store_true",
    help="强制重新处理所有项目",
)
enhanced_group.add_argument(
    "--description-api-base",
    type=str,
    help="覆盖描述 API 地址",
)
enhanced_group.add_argument(
    "--description-model",
    type=str,
    help="覆盖描述生成模型",
)
```

### 5.2 使用示例

```bash
# 完整增强提取
uv run python main.py --enhanced-extract

# 禁用 AI 描述（更快）
uv run python main.py --enhanced-extract --no-description

# 强制重新处理
uv run python main.py --enhanced-extract --force-reprocess

# 使用自定义模型
uv run python main.py --enhanced-extract --description-model "gpt-4-vision-preview"

# 单个 PDF
uv run python main.py --single-pdf data/papers/test.pdf --enhanced-extract

# 只上传到 MinIO，不索引
uv run python main.py --enhanced-extract --no-qdrant --no-description
```

---

## 6. 数据 Schema

### 6.1 MinIO 存储结构

```
bucket: document-extractions/
├── paper1/
│   ├── figures/
│   │   ├── fig_01_01.png
│   │   ├── fig_01_02.png
│   │   └── fig_02_01.png
│   └── tables/
│       ├── table_01_01.png
│       └── table_02_01.png
├── paper2/
│   ├── figures/
│   └── tables/
└── ...
```

### 6.2 Qdrant Payload Schema

```json
{
  "pdf_name": "research_paper",
  "item_id": "fig_01_02",
  "item_type": "figure",
  "page_number": 1,
  "caption": "Figure 1: 系统架构概览",
  "ai_description": "这是一张系统架构图，展示了微服务设计的三个主要组件：API 网关、服务网格和数据库集群。图中使用方框和箭头说明了组件之间的数据流向。",
  "minio_url": "http://localhost:9000/document-extractions/research_paper/figures/fig_01_02.png",
  "minio_object_key": "research_paper/figures/fig_01_02.png",
  "bbox": {"x1": 100.5, "y1": 200.0, "x2": 500.5, "y2": 600.0},
  "created_at": "2026-01-27T10:30:00Z",
  "text_content": "图注: Figure 1: 系统架构概览 描述: 这是一张系统架构图..."
}
```

---

## 7. 依赖

### 7.1 pyproject.toml 新增

```toml
[project]
dependencies = [
    # ... 现有依赖 ...
    "minio>=7.2.0",           # MinIO SDK
    "qdrant-client>=1.7.0",   # Qdrant SDK
    "httpx>=0.26.0",          # HTTP 客户端（用于 API 调用）
]
```

### 7.2 安装命令

```bash
uv add minio qdrant-client httpx
```

---

## 8. Docker 开发环境

### 8.1 docker-compose.yml

```yaml
version: '3.8'

services:
  minio:
    image: minio/minio:latest
    container_name: yolo-minio
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  qdrant:
    image: qdrant/qdrant:latest
    container_name: yolo-qdrant
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

volumes:
  minio_data:
  qdrant_data:
```

### 8.2 启动服务

```bash
# 启动服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 8.3 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| MinIO API | http://localhost:9000 | S3 兼容 API |
| MinIO Console | http://localhost:9001 | Web 管理界面 |
| Qdrant REST | http://localhost:6333 | REST API |
| Qdrant Dashboard | http://localhost:6333/dashboard | Web 界面 |

---

## 9. 实现步骤清单

### Phase 1: 基础设施
- [ ] 创建 `src/storage/__init__.py`
- [ ] 创建 `src/storage/base.py`
- [ ] 创建 `src/storage/minio_storage.py`
- [ ] 创建 `src/vectordb/__init__.py`
- [ ] 创建 `src/vectordb/embeddings.py`
- [ ] 创建 `src/vectordb/qdrant_store.py`
- [ ] 更新 `pyproject.toml` 添加依赖

### Phase 2: 描述生成
- [ ] 创建 `src/description/__init__.py`
- [ ] 创建 `src/description/base.py`
- [ ] 创建 `src/description/openai_generator.py`

### Phase 3: 集成管道
- [ ] 创建 `src/pipeline/__init__.py`
- [ ] 创建 `src/pipeline/enhanced_extractor.py`

### Phase 4: 配置与 CLI
- [ ] 更新 `config/config.yaml`
- [ ] 更新 `main.py` 添加新参数
- [ ] 更新 `src/__init__.py` 导出新模块

### Phase 5: 测试与部署
- [ ] 创建 `docker-compose.yml`
- [ ] 创建 `.env.example`
- [ ] 测试完整流程
- [ ] 更新 `CLAUDE.md`

---

## 10. 验证测试

### 10.1 启动环境
```bash
docker-compose up -d
```

### 10.2 运行提取
```bash
uv run python main.py --single-pdf data/papers/test.pdf --enhanced-extract
```

### 10.3 验证 MinIO
1. 访问 http://localhost:9001
2. 登录 (minioadmin/minioadmin)
3. 检查 bucket 中的图片

### 10.4 验证 Qdrant
1. 访问 http://localhost:6333/dashboard
2. 检查 collection 数据

### 10.5 测试搜索
```python
from src.vectordb import QdrantVectorStore, OpenAICompatibleEmbedding

embedding = OpenAICompatibleEmbedding(
    api_base="https://api.siliconflow.cn/v1",
    api_key="your_key",
    model="BAAI/bge-large-zh-v1.5",
)

store = QdrantVectorStore(
    host="localhost",
    port=6333,
    collection_name="document_extractions",
    embedding=embedding,
)

results = store.search("系统架构图", limit=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['item_id']}: {r['caption']}")
```

---

## 附录 A: 常见问题

### Q1: MinIO 连接失败
确保 Docker 容器正在运行:
```bash
docker-compose ps
docker-compose logs minio
```

### Q2: Qdrant 创建 collection 失败
检查 embedding 维度是否正确，BGE-large 的维度是 1024。

### Q3: AI 描述生成超时
增加 timeout 配置或使用更快的模型。

### Q4: 如何切换到其他向量数据库？
实现 `BaseVectorStore` 接口并在配置中切换。
