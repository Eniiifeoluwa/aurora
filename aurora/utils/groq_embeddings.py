# aurora/utils/mistral_embeddings.py
import os
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_mistralai import MistralAIEmbeddings


class GroqEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper for MistralAI embeddings API.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set")

        self.model = model or os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")


        self.client = MistralAIEmbeddings(
            model=self.model,
            api_key=self.api_key
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (list of strings).
        """
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        """
        return self.client.embed_query(text)
