import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from aurora.utils.groq_embeddings import GroqEmbeddings


class RAGPipeline:
    def __init__(self, persist_directory: str = "chroma_store"):
        self.embed_model = GroqEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.persist_directory = persist_directory

        # vectorstore will be built lazily
        self.vectorstore = None

    def index_documents(self, docs: List[Document], persist: bool = True):
        # split docs
        splitted = []
        for d in docs:
            chunks = self.text_splitter.split_text(d.page_content)
            for i, chunk in enumerate(chunks):
                meta = dict(d.metadata) if d.metadata else {}
                meta["chunk"] = i
                splitted.append(Document(page_content=chunk, metadata=meta))

        # prepare texts & metadata
        texts = [d.page_content for d in splitted]
        metadatas = [d.metadata for d in splitted]

        # build Chroma index
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embed_model,
            metadatas=metadatas,
            persist_directory=self.persist_directory,
        )

        if persist:
            self.vectorstore.persist()

    def load_index(self):
        if not self.persist_directory:
            raise ValueError("persist_directory is None")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embed_model,
        )

    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[Document]]:
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized - index or load first")

        docs = self.vectorstore.similarity_search(query_text, k=k)

        # join context for LLM prompt usage
        context = "\n\n---\n\n".join(
            [f"Source: {d.metadata.get('source', 'unknown')}\n\n{d.page_content}" for d in docs]
        )
        return context, docs
