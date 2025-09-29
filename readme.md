# Aurora 🐦

![Aurora Banner](assets/aurora.png)
**Aurora** is an AI assistant that lets you **chat with your PDFs, images, or text.**.

Requirements:
- Python 3.8+
- Tesseract OCR installed (for image OCR)
- Groq API key

Install:
pip install -r requirements.txt


Run:
streamlit run aurora/app.py

Notes:
- This app uses MistralAI for embeddings (via the Langchain python client).
- To use Groq for LLM/chat completions, either call Groq's chat API directly or configure OpenAI-compatible base URL.
