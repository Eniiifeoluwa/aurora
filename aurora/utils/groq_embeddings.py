# aurora/utils/groq_embeddings.py
import os
from typing import List, Sequence
from langchain.embeddings.base import Embeddings
from groq import Groq
import numpy as np

class GroqEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper for Groq embeddings API.
    Uses groq.Groq client under the hood.
    """

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        self.model = model or os.environ.get("GROQ_EMBED_MODEL", "nomic-embed-text-v1.5")
        self.client = Groq(api_key=self.api_key)

    def _call_groq(self, texts: Sequence[str]) -> List[Sequence[float]]:
        """
        Call Groq embeddings endpoint.
        Response objects differ across versions; this handles the common patterns.
        """
        resp = self.client.embeddings.create(input=list(texts), model=self.model)
        # try several ways to extract embeddings:
        # 1) resp.data -> list of objects each with 'embedding'
        # 2) resp['data']
        # 3) resp.embeddings or resp.embedding
        embeddings = []
        # Favor typed pydantic models (common in groq client)
        try:
            # if resp has attribute data
            data = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None
            if data:
                for item in data:
                    # item might be pydantic object or dict
                    emb = getattr(item, "embedding", None) or item.get("embedding")
                    embeddings.append(list(emb))
                return embeddings
        except Exception:
            pass

        # fallback: if resp itself is a dict with 'embedding' or 'embeddings'
        try:
            if isinstance(resp, dict) and "embedding" in resp:
                return [resp["embedding"]]
        except Exception:
            pass

        # last-ditch: try to read resp.embedding or resp.embeddings
        try:
            emb_attr = getattr(resp, "embedding", None) or getattr(resp, "embeddings", None)
            if emb_attr:
                if isinstance(emb_attr[0], Sequence):
                    return [list(e) for e in emb_attr]
                return [list(emb_attr)]
        except Exception:
            pass

        # If none of the above worked, raise so the user can inspect raw response
        raise RuntimeError(f"Unable to parse embeddings response from Groq: {resp}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call_groq(texts)

    def embed_query(self, text: str) -> List[float]:
        return list(self._call_groq([text])[0])
