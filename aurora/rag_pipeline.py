from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from aurora.utils.groq_embeddings import GroqEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple

class RAGPipeline:
    def __init__(self, persist_directory: str = None):
        self.embed_model = GroqEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
            # ✅ Use persist_directory directly
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embed_model,
                metadatas=metadatas,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()  # safe now
        else:
            self.vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=self.embed_model,
                metadatas=metadatas
            )

    def load_index(self):
    if self.persist_directory is None:
        raise ValueError("persist_directory is None")
    self.vectorstore = FAISS.load_local(
        self.persist_directory,
        self.embed_model,
        allow_dangerous_deserialization=True  # 👈 required
    )


    def query(self, query_text: str, k: int = 4) -> Tuple[str, List[Document]]:
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized - index documents first")
        docs = self.vectorstore.similarity_search(query_text, k=k)
        context = "\n\n---\n\n".join(
            [f"Source: {d.metadata.get('source', 'unknown')}\n\n{d.page_content}" for d in docs]
        )
        return context, docs
