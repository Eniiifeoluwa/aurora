from PIL import Image
from io import BytesIO

# Try Tesseract first
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # adjust if needed

    def image_to_text(file_bytes: bytes) -> str:
        """
        Converts image bytes to text using Tesseract if available.
        """
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)

except (ImportError, pytesseract.TesseractNotFoundError):
    # Fall back to EasyOCR if Tesseract is missing
    import easyocr
    import numpy as np
    reader = easyocr.Reader(["en"], gpu=False)

    def image_to_text(file_bytes: bytes) -> str:
        """
        Converts image bytes to text using EasyOCR as fallback.
        """
        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        results = reader.readtext(np.array(img), detail=0)
        return " ".join(results)
