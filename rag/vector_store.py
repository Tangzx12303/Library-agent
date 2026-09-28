import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    # ------------------------------------------------------------------
    # MD5 台账：记录已入库文件的指纹，用于增量加载时去重
    # ------------------------------------------------------------------
    @property
    def _md5_store_path(self) -> str:
        return get_abs_path(chroma_conf["md5_hex_store"])

    def _load_md5_hex_set(self) -> set[str]:
        """读取台账并返回 MD5 集合，查重由 O(n) 线性扫描降为 O(1)。"""
        if not os.path.exists(self._md5_store_path):
            # 创建空台账文件
            open(self._md5_store_path, "w", encoding="utf-8").close()
            return set()

        with open(self._md5_store_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def _save_md5_hex(self, md5_hex: str) -> None:
        with open(self._md5_store_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")

    def reset(self) -> None:
        """清空向量库与 MD5 台账，用于重建知识库。

        load_document() 是「只增不减」的增量加载：台账记录过的文件会被永久跳过，
        因此源文件被删除或改动后无法自动感知。需要完全重建时先调用本方法。
        """
        self.vector_store.reset_collection()
        open(self._md5_store_path, "w", encoding="utf-8").close()
        logger.info("[重置知识库]向量库与MD5台账已清空")

    # ------------------------------------------------------------------
    # 文档加载
    # ------------------------------------------------------------------
    @staticmethod
    def _get_file_documents(read_path: str) -> list[Document]:
        if read_path.endswith("txt"):
            return txt_loader(read_path)

        if read_path.endswith("pdf"):
            return pdf_loader(read_path)

        return []

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 一次性读入台账，避免每个文件都重新扫描整个文件
        processed_md5_hex: set[str] = self._load_md5_hex_set()

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if md5_hex is None:
                logger.warning(f"[加载知识库]{path}的MD5计算失败，跳过")
                continue

            if md5_hex in processed_md5_hex:
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = self._get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                self._save_md5_hex(md5_hex)
                processed_md5_hex.add(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    # 命令行入口：将 data/ 目录下的知识库文档增量加载进向量库
    # 用法：python -m rag.vector_store
    VectorStoreService().load_document()
