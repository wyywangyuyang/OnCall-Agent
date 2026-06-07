"""
向量索引服务模块
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.config import config
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """索引结果类"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        """增加成功计数"""
        self.success_count += 1

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """添加失败文件"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """获取耗时（毫秒）"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(self):
        """初始化向量索引服务"""
        self.upload_path = "./uploads"
        logger.info("向量索引服务初始化完成")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """
        索引指定目录下的所有文件

        Args:
            directory_path: 目录路径（可选，默认使用配置的上传目录）

        Returns:
            IndexingResult: 索引结果
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()  # 将相对路径转换为绝对路径

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 获取所有支持的文件（从处理器注册表中动态获取）
            supported_exts = document_splitter_service.supported_extensions
            files: list[Path] = []
            for ext in supported_exts:
                # 去掉点号，如 ".md" → "md"
                pattern = f"*.{ext.lstrip('.')}"
                files.extend(dir_path.glob(pattern))

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历并索引每个文件
            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str):
        """
        索引单个文件（自动根据文件类型选择处理器）

        支持的文件类型: .txt, .md, .markdown, .pdf, .docx, .doc

        Args:
            file_path: 文件路径

        Raises:
            ValueError: 文件不存在或文件类型不支持
            RuntimeError: 索引失败时抛出
        """
        path = Path(file_path).resolve()   # 将相对路径转换为绝对路径

        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")

        try:
            # 1. 检查文件类型是否支持
            ext = path.suffix.lower()
            supported = document_splitter_service.supported_extensions
            if ext not in supported:
                raise ValueError(
                    f"不支持的文件类型: {ext}，支持的类型: {', '.join(supported)}"
                )

            # 2. 删除该文件的旧数据（如果已存在）
            normalize_path = path.as_posix()
            vector_store_manager.delete_by_source(normalize_path)
            logger.info(f"删除旧数据（如果存在）: {normalize_path}")

            # 3. 直接传文件路径给分片服务（分片服务内部会根据扩展名自动选择处理器并读取文件）
            documents = document_splitter_service.split_documents(normalize_path)
            logger.info(f"文档分割完成: {file_path} → {len(documents)} 个分片")

            # 4. 添加文档到向量存储
            if documents:
                vector_store_manager.add_documents(documents)

                # 5. 同步写入 BM25 FTS5 索引（如果启用）
                if config.bm25_enabled:
                    self._sync_bm25_add(documents)

                logger.info(f"文件索引完成: {file_path}, 共 {len(documents)} 个分片")
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e

    def _sync_bm25_add(self, documents) -> None:
        """
        将文档分片同步写入 BM25 FTS5 索引

        从 doc.metadata["chunk_id"] 读取 ID（由 vector_store_manager.add_documents 写入），
        保证与 Milvus 主键一致。

        Args:
            documents: 已完成 Milvus 写入的文档列表（metadata 中已含 chunk_id）
        """
        try:
            from app.services.bm25_index_service import bm25_index_service

            chunk_ids = [doc.metadata["chunk_id"] for doc in documents]
            bm25_index_service.add_chunks(documents, chunk_ids)
        except Exception as e:
            logger.warning(f"BM25 索引同步写入失败（不影响主流程）: {e}")


# 全局单例
vector_index_service = VectorIndexService()
