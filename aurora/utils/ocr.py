from PIL import Image
import pytesseract
from io import BytesIO

def image_to_text(file_bytes: bytes) -> str:
    """
    Converts image bytes to text using Tesseract.
    """
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img)
    return text
