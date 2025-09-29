import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from .utils.groq_embeddings import GroqEmbeddings
class RAGPipeline:
    def __init__(self, persist_directory: str = None):
        self.embed_model = GroqEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.persist_directory = persist_directory

        # vectorstore will be built lazily
        self.vectorstore = None

    def index_documents(self, docs: List[Document], persist: bool = False):
        # split docs
        splitted = []
        for d in docs:
            chunks = self.text_splitter.split_text(d.page_content)
            for i, chunk in enumerate(chunks):
                meta = dict(d.metadata) if d.metadata else {}
                meta["chunk"] = i
                splitted.append(Document(page_content=chunk, metadata=meta))

        # create FAISS index
        texts = [d.page_content for d in splitted]
        metadatas = [d.metadata for d in splitted]

        if self.persist_directory and persist:
            self.vectorstore = FAISS.from_texts(texts, self.embed_model, metadatas=metadatas)
            self.vectorstore.save_local(self.persist_directory)
        else:
            self.vectorstore = FAISS.from_texts(texts, self.embed_model, metadatas=metadatas)

    def load_index(self):
        if self.persist_directory is None:
            raise ValueError("persist_directory is None")
        self.vectorstore = FAISS.load_local(self.persist_directory, self.embed_model)

    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[Document]]:
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized - index documents first")
        docs = self.vectorstore.similarity_search(query_text, k=k)
        # join context for LLM prompt usage
        context = "\n\n---\n\n".join([f"Source: {d.metadata.get('source', 'unknown')}\n\n{d.page_content}" for d in docs])
        return context, docs
