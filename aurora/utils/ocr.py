from PIL import Image
import pytesseract
from io import BytesIO

# Fallback
import easyocr

# Initialize EasyOCR reader once
easyocr_reader = easyocr.Reader(["en"], gpu=False)

def image_to_text(file_bytes: bytes) -> str:
    """
    Converts image bytes to text.
    - Uses pytesseract if available (local).
    - Falls back to EasyOCR if Tesseract binary is missing (e.g., Streamlit Cloud).
    """
    img = Image.open(BytesIO(file_bytes)).convert("RGB")

    try:
        # Try pytesseract first
        return pytesseract.image_to_string(img).strip()
    except pytesseract.TesseractNotFoundError:
        # Fallback to EasyOCR
        results = easyocr_reader.readtext(img)
        return " ".join([text for _, text, _ in results]).strip()
