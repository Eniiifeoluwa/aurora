# Aurora 🐦

![Aurora Banner](assets/aurora.png)
**Aurora** is an  AI assistant that lets you **chat with your PDFs, images, or text.**.

Requirements:
- Python 3.8+
- Tesseract OCR installed (for image OCR)
- Groq API key

Install:
pip install -r requirements.txt
cp .env.example .env
# set GROQ_API_KEY and optional GROQ_EMBED_MODEL in .env

Run:
streamlit run aurora/app.py

Notes:
- This app uses Groq for embeddings (via the groq python client).
- To use Groq for LLM/chat completions, either call Groq's chat API directly or configure OpenAI-compatible base URL.
