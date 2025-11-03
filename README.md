# News Research Tool 📈

**News Research Tool** is an interactive Streamlit application that extracts content from news articles, creates embeddings for the text, and answers questions based on the collected information. The app uses **Google Gemini (via LangChain)** as the LLM and **HuggingFace** for text embeddings.

---
<img width="1919" height="926" alt="image" src="https://github.com/user-attachments/assets/78ea312a-f665-4a45-88a9-1c781ed1dd43" />

---

## Features

- Input up to 3 news article URLs via the sidebar.
- Load article content dynamically using Selenium.
- Split text into chunks with overlap to preserve context.
- Generate text embeddings with HuggingFace models.
- Store embeddings locally with FAISS for fast retrieval.
- Ask questions using Google Gemini LLM.
- Display answers along with sources.
