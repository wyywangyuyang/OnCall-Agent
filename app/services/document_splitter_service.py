"""
文档分割服务模块 - 基于 LangChain 的智能文档分割

采用「处理器注册表」模式：
- 每种文件类型对应一个独立的处理器（Processor）
- 根据文件扩展名自动路由到对应的处理器
- 新增文件类型只需在注册表中添加一行即可

支持的文件类型及分片策略：
┌──────────┬──────────────────────────────────────────────────────┐
│ 文件类型  │ 分片策略                                               │
├──────────┼──────────────────────────────────────────────────────┤
│ Markdown │ ① 按 H1/H2 标题分割                                   │
│ (.md)    │ ② RecursiveCharacterTextSplitter 二次分割              │
│          │ ③ 合并 < 300 字符的小片段                              │
├──────────┼──────────────────────────────────────────────────────┤
│ 纯文本    │ ① RecursiveCharacterTextSplitter 直接分割              │
│ (.txt)   │                                                      │
├──────────┼──────────────────────────────────────────────────────┤
│ PDF      │ ① PyPDFLoader 逐页提取文本                             │
│ (.pdf)   │ ② RecursiveCharacterTextSplitter 二次分割              │
├──────────┼──────────────────────────────────────────────────────┤
│ Word     │ ① Docx2txtLoader 提取文本                              │
│ (.docx)  │ ② RecursiveCharacterTextSplitter 二次分割              │
└──────────┴──────────────────────────────────────────────────────┘
"""
from pathlib import Path
from typing import Callable, Dict, List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config


# ============================================================================
# 处理器注册表（Processor Registry）
# ============================================================================
# 这是一个字典，key 是文件扩展名（小写，带点），value 是对应的处理器函数。
# 每种处理器函数签名：processor(file_path: Path) -> list[Document]
# 新增文件类型时，只需在这里添加一行即可。
# ============================================================================

# 类型别名：处理器函数
ProcessorFunc = Callable[[Path], List[Document]]

# 全局处理器注册表（在 DocumentSplitterService.__init__ 中填充）
_PROCESSOR_REGISTRY: Dict[str, ProcessorFunc] = {}


def register_processor(extensions: List[str]):
    """
    装饰器：将处理器函数注册到注册表中

    用法：
        @register_processor([".md", ".markdown"])
        def _process_markdown(file_path: Path) -> list[Document]:
            ...

    Args:
        extensions: 该处理器支持的文件扩展名列表
    """
    def decorator(func: ProcessorFunc) -> ProcessorFunc:
        for ext in extensions:
            _PROCESSOR_REGISTRY[ext.lower()] = func
            logger.debug(f"注册处理器: {ext} -> {func.__name__}")
        return func
    return decorator


def get_processor(file_path: str | Path) -> ProcessorFunc:
    """
    根据文件扩展名查找对应的处理器

    Args:
        file_path: 文件路径

    Returns:
        对应的处理器函数

    Raises:
        ValueError: 不支持的文件类型
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in _PROCESSOR_REGISTRY:
        supported = ", ".join(sorted(_PROCESSOR_REGISTRY.keys()))
        raise ValueError(
            f"不支持的文件类型: {ext}\n"
            f"当前支持的类型: {supported}"
        )

    return _PROCESSOR_REGISTRY[ext]


# ============================================================================
# 通用工具函数
# ============================================================================

def _create_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """创建递归字符分割器（统一配置）"""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_max_size * 2,  # 1600 字符/块
        chunk_overlap=config.chunk_overlap,    # 100 字符重叠
        length_function=len,
        is_separator_regex=True,
    )


def _add_file_metadata(documents: List[Document], file_path: str, extension: str) -> List[Document]:
    """
    为文档分片添加文件元数据

    Args:
        documents: 文档分片列表
        file_path: 文件路径
        extension: 文件扩展名

    Returns:
        添加了元数据的文档分片列表
    """
    path = Path(file_path)
    for doc in documents:
        doc.metadata["source"] = file_path
        doc.metadata["extension"] = extension
        doc.metadata["file_name"] = path.name
    return documents


def _merge_small_chunks(
    documents: List[Document],
    min_size: int = 300,
    max_size: int | None = None,
) -> List[Document]:
    """
    合并太小的分片到前一个分片中

    为什么要合并？
    - 按标题分割 Markdown 时，某些小节可能只有一两句话（如 "## 简介\n这是简介。"）
    - 这些小片段单独存储会导致检索结果碎片化
    - 合并到前一个片段中，能保证每个分片的信息量充足

    Args:
        documents: 文档分片列表
        min_size: 最小分片大小（字符数），小于此值的分片会被合并。默认 300
        max_size: 合并后的最大分片大小（字符数），超过则不合并。默认使用 chunk_max_size * 2

    Returns:
        合并后的文档分片列表
    """
    if max_size is None:
        max_size = config.chunk_max_size * 2

    if not documents:
        return []

    merged: List[Document] = []
    current = None

    for doc in documents:
        doc_size = len(doc.page_content)

        if current is None:
            current = doc
        elif doc_size < min_size and len(current.page_content) < max_size:
            # 当前分片太小，且合并后不会超过最大限制 → 合并
            current.page_content += "\n\n" + doc.page_content
        else:
            # 保存当前分片，开始新的
            merged.append(current)
            current = doc

    if current:
        merged.append(current)

    return merged


# ============================================================================
# 各文件类型处理器（Processor）
# ============================================================================

@register_processor([".md", ".markdown"])
def _process_markdown(file_path: Path) -> List[Document]:
    """
    Markdown 文件处理器 —— 三段式分片

    分片流程：
    ┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │ 原始 Markdown │ →  │ 第一阶段：按标题分割 │ →  │ 第二阶段：按大小分割 │
    │   文档        │    │ H1(一级标题)       │    │ 最大 1600 字符/块  │
    │              │    │ H2(二级标题)       │    │ 重叠 100 字符      │
    └─────────────┘     └──────────────────┘     └──────────────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ 第三阶段：合并     │
                                              │ < 300 字符的小块   │
                                              │ 合并到前一个块     │
                                              └──────────────────┘

    为什么只用 H1 和 H2？（不用 H3 及以下）
    - H3 及更小的标题通常对应很小的内容块（如几个要点）
    - 按这些小标题分割会导致分片过于碎片化
    - 只用 H1/H2 保证每个分片都有足够的信息量

    Args:
        file_path: Markdown 文件路径

    Returns:
        文档分片列表
    """
    content = file_path.read_text(encoding="utf-8")

    if not content or not content.strip():
        logger.warning(f"Markdown 文档内容为空: {file_path}")
        return []

    try:
        normalized_path = file_path.as_posix()

        # 第一阶段：按标题分割
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不使用 ###，避免过度碎片化
            ],
            strip_headers=False,  # 标题保留在分片内容中
        )
        header_docs = markdown_splitter.split_text(content)

        # 第二阶段：按大小二次分割
        recursive_splitter = _create_recursive_splitter()
        split_docs = recursive_splitter.split_documents(header_docs)

        # 第三阶段：合并小片段
        final_docs = _merge_small_chunks(split_docs, min_size=300)

        # 添加元数据
        final_docs = _add_file_metadata(final_docs, normalized_path, ".md")

        logger.info(
            f"[Markdown] 分割完成: {file_path.name} → "
            f"标题分 {len(header_docs)} 块 → "
            f"大小分 {len(split_docs)} 块 → "
            f"合并后 {len(final_docs)} 块"
        )
        return final_docs

    except Exception as e:
        logger.error(f"[Markdown] 分割失败: {file_path}, 错误: {e}")
        raise


@register_processor([".txt"])
def _process_text(file_path: Path) -> List[Document]:
    """
    纯文本文件处理器 —— 单阶段分片

    分片流程：
    ┌─────────────┐     ┌──────────────────┐
    │ 原始纯文本    │ →  │ RecursiveCharacterTextSplitter │
    │   文件       │    │ 最大 1600 字符/块               │
    │              │    │ 重叠 100 字符                   │
    └─────────────┘     └──────────────────┘

    为什么纯文本不用标题分割？
    - 纯文本没有 Markdown 的 #/## 标题结构
    - 直接用字符数分割最简单、最可靠
    - RecursiveCharacterTextSplitter 会优先在段落/句子边界分割，保证可读性

    Args:
        file_path: 文本文件路径

    Returns:
        文档分片列表
    """
    content = file_path.read_text(encoding="utf-8")

    if not content or not content.strip():
        logger.warning(f"文本文档内容为空: {file_path}")
        return []

    try:
        normalized_path = file_path.as_posix()
        recursive_splitter = _create_recursive_splitter()

        docs = recursive_splitter.create_documents(
            texts=[content],
            metadatas=[{
                "source": normalized_path,
                "extension": ".txt",
                "file_name": file_path.name,
            }],
        )

        logger.info(f"[Text] 分割完成: {file_path.name} → {len(docs)} 块")
        return docs

    except Exception as e:
        logger.error(f"[Text] 分割失败: {file_path}, 错误: {e}")
        raise


@register_processor([".pdf"])
def _process_pdf(file_path: Path) -> List[Document]:
    """
    PDF 文件处理器 —— 两阶段分片

    分片流程：
    ┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
    │  PDF 文件    │ →  │ PyPDFLoader    │ →  │ RecursiveCharacter │
    │  (二进制)    │    │ 逐页提取文本    │    │ TextSplitter      │
    │              │    │ 每页→一个Document│   │ 最大 1600 字符/块  │
    └─────────────┘     └────────────────┘     └──────────────────┘

    为什么逐页提取后再分片？
    - PyPDFLoader 每页生成一个 Document（保持页面边界信息）
    - 但有些页面可能很长（如全页文字），需要进一步按大小分片
    - RecursiveCharacterTextSplitter 确保每块不超过 1600 字符

    关于 pypdf：
    - pypdf 是纯 Python 的 PDF 库，MIT 许可证，无需系统依赖
    - 它是 LangChain 官方推荐的 PDF 加载后端
    - 能提取大部分 PDF 中的文本（扫描版 PDF 除外——扫描版需要 OCR）

    Args:
        file_path: PDF 文件路径

    Returns:
        文档分片列表
    """
    try:
        normalized_path = file_path.as_posix()

        # 第一阶段：用 PyPDFLoader 逐页提取文本
        loader = PyPDFLoader(str(file_path))
        page_docs = loader.load()

        if not page_docs:
            logger.warning(f"[PDF] 文件内容为空或无法提取文本: {file_path}")
            return []

        # 从每页的 metadata 中删除 source（因为我们要用自定义的文件路径）
        # PyPDFLoader 默认把 source 设为 PDF 文件路径，我们需要用 normalize_path
        for doc in page_docs:
            doc.metadata["source"] = normalized_path

        logger.debug(f"[PDF] PyPDFLoader 提取: {file_path.name} → {len(page_docs)} 页")

        # 第二阶段：用 RecursiveCharacterTextSplitter 按大小分片
        recursive_splitter = _create_recursive_splitter()
        split_docs = recursive_splitter.split_documents(page_docs)

        # 添加元数据
        split_docs = _add_file_metadata(split_docs, normalized_path, ".pdf")

        logger.info(f"[PDF] 分割完成: {file_path.name} → {len(page_docs)} 页 → {len(split_docs)} 块")
        return split_docs

    except Exception as e:
        logger.error(f"[PDF] 分割失败: {file_path}, 错误: {e}")
        raise


@register_processor([".docx", ".doc"])
def _process_docx(file_path: Path) -> List[Document]:
    """
    Word 文件处理器 —— 两阶段分片

    分片流程：
    ┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
    │  Word 文件   │ →  │ Docx2txtLoader │ →  │ RecursiveCharacter │
    │  (.docx)    │    │ 提取全文文本    │    │ TextSplitter      │
    │  (二进制)    │    │ → 一个 Document │   │ 最大 1600 字符/块  │
    └─────────────┘     └────────────────┘     └──────────────────┘

    关于 docx2txt：
    - docx2txt 是纯 Python 的 .docx 解析库，MIT 许可证
    - 它会提取文档中的文本内容（包括段落、表格中的文字）
    - 不保留格式信息（加粗、颜色等），只提取纯文本
    - 不支持旧版 .doc 格式（二进制 Word 97-2003），仅支持 .docx（Office 2007+）

    局限说明：
    - .doc 文件（Word 97-2003）：当前处理器无法提取。如果上传 .doc 文件，会尝试
      用 Docx2txtLoader 加载，通常会报错。建议用户在 Word 中将 .doc 另存为 .docx。
    - 图片、嵌入对象中的文字：无法提取
    - 复杂表格：提取为纯文本，可能丢失表格结构

    Args:
        file_path: Word 文件路径

    Returns:
        文档分片列表
    """
    try:
        normalized_path = file_path.as_posix()

        # 第一阶段：用 Docx2txtLoader 提取全文文本
        loader = Docx2txtLoader(str(file_path))
        full_docs = loader.load()

        if not full_docs:
            logger.warning(f"[Word] 文件内容为空或无法提取文本: {file_path}")
            return []

        logger.debug(f"[Word] Docx2txtLoader 提取: {file_path.name} → {len(full_docs)} 个文档")

        # 第二阶段：用 RecursiveCharacterTextSplitter 按大小分片
        recursive_splitter = _create_recursive_splitter()
        split_docs = recursive_splitter.split_documents(full_docs)

        # 添加元数据
        split_docs = _add_file_metadata(split_docs, normalized_path, ".docx")

        logger.info(f"[Word] 分割完成: {file_path.name} → {len(split_docs)} 块")
        return split_docs

    except Exception as e:
        logger.error(f"[Word] 分割失败: {file_path}, 错误: {e}")
        raise


# ============================================================================
# DocumentSplitterService 统一入口
# ============================================================================

class DocumentSplitterService:
    """
    文档分割服务 —— 统一入口

    通过处理器注册表自动匹配文件类型，调用对应的处理器。

    使用方式：
        service = DocumentSplitterService()
        documents = service.split_documents("/path/to/file.pdf")
        # documents 是 List[Document]，可直接存入向量库
    """

    def __init__(self):
        """初始化文档分割服务"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # 收集所有已注册的扩展名
        supported_types = sorted(_PROCESSOR_REGISTRY.keys())

        logger.info(
            f"文档分割服务初始化完成 | "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap} | "
            f"支持 {len(supported_types)} 种文件类型: {', '.join(supported_types)}"
        )

    def split_documents(self, file_path: str | Path) -> List[Document]:
        """
        智能分割文档 —— 根据文件类型自动选择处理器

        这是统一入口，所有文件类型都通过这个方法处理。
        方法会根据文件扩展名自动查找对应的处理器，并调用它。

        Args:
            file_path: 文件路径（字符串或 Path 对象）

        Returns:
            List[Document]: 文档分片列表，每个 Document 包含：
                - page_content: 分片文本内容
                - metadata: {"source", "extension", "file_name"}

        Raises:
            ValueError: 不支持的文件类型
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 查找处理器
        processor = get_processor(path)
        ext = path.suffix.lower()
        logger.info(f"文件类型: {ext} → 处理器: {processor.__name__}")

        # 调用处理器
        documents = processor(path)

        if not documents:
            logger.warning(f"分割结果为空: {file_path}")

        return documents

    @property
    def supported_extensions(self) -> List[str]:
        """获取所有支持的文件扩展名"""
        return sorted(_PROCESSOR_REGISTRY.keys())

    @property
    def supported_types_description(self) -> Dict[str, str]:
        """获取支持的文件类型及其描述"""
        descriptions = {}
        for ext, processor in _PROCESSOR_REGISTRY.items():
            doc = processor.__doc__ or ""
            # 提取第一行作为描述
            first_line = doc.strip().split("\n")[0] if doc.strip() else "无描述"
            descriptions[ext] = first_line
        return descriptions


# 全局单例
document_splitter_service = DocumentSplitterService()
