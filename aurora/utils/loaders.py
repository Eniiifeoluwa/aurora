from langchain_community.document_loaders import TextLoader, PyPDFLoader
from typing import List
from langchain.schema import Document
import io
from aurora.utils.ocr import image_to_text
def load_text_file(file_bytes: bytes, filename: str) -> List[Document]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": filename})]

def load_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    tmp_path = f"/tmp/{filename}"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = filename
    return docs

def load_image(file_bytes: bytes, filename: str) -> List[Document]:
    text = image_to_text(file_bytes)
    return [Document(page_content=text, metadata={"source": filename})]
