from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from aurora.utils.groq_embeddings import GroqEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import os

class RAGPipeline:
    def __init__(self, persist_directory: str = None):
        self.embed_model = GroqEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150
        )
        self.persist_directory = persist_directory
        self.vectorstore = None

    def index_documents(self, docs: List[Document], persist: bool = False):
        splitted = []
        for d in docs:
            chunks = self.text_splitter.split_text(d.page_content)
            for i, chunk in enumerate(chunks):
                meta = dict(d.metadata) if d.metadata else {}
                meta["chunk"] = i
                splitted.append(Document(page_content=chunk, metadata=meta))

        texts = [d.page_content for d in splitted]
        metadatas = [d.metadata for d in splitted]

        if self.persist_directory and persist:
            # Persist to disk
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embed_model,
                metadatas=metadatas,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
        else:
            # In-memory only
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embed_model,
                metadatas=metadatas
            )

    def load_index(self):
        if self.persist_directory is None:
            raise ValueError("persist_directory is None")
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"No index found at {self.persist_directory}")

        # Reload Chroma
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embed_model
        )

    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[Document]]:
        # Auto-load if vectorstore missing
        if self.vectorstore is None:
            if self.persist_directory and os.path.exists(self.persist_directory):
                self.load_index()
            else:
                raise ValueError("Vectorstore not initialized and no persisted index found")

        docs = self.vectorstore.similarity_search(query_text, k=k)
        context = "\n\n---\n\n".join(
            [f"Source: {d.metadata.get('source', 'unknown')}\n\n{d.page_content}" for d in docs]
        )
        return context, docs
