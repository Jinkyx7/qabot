# RAG PDF QA Bot

This project serves a simple retrieval-augmented generation (RAG) chatbot that answers questions about an uploaded PDF using IBM watsonx.ai models, LangChain, and a Gradio UI.

## Prerequisites
- Python 3.10+
- IBM watsonx.ai credentials with access to the configured model and project.
- Optional: Hugging Face token if required by downstream models (`huggingface-cli login`).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Locally
```bash
python qabot.py
```
- The Gradio interface starts on `http://0.0.0.0:7860/`.
- Upload a single PDF and ask a question to trigger retrieval and generation.

## Configuration
- Edit model IDs, project ID, and generation parameters in `qabot.py` (e.g., `model_id`, `project_id`, chunk sizes).
- Provide credentials via environment variables or your IBM Cloud config; avoid hard-coding secrets.

## Project Structure
- `qabot.py`: Gradio app wiring PDF loading, text splitting, Watsonx embeddings/LLM, Chroma vector store, and a RetrievalQA chain.
- `AGENTS.md`: Repository guidelines for contributors.

## Development Notes
- Functions are structured to keep the pipeline modular (loader → splitter → embedder → vector store → retriever → QA chain).
- Add tests with `pytest` as logic grows (e.g., chunking behavior, retriever wiring). Run with `pytest` after installing dev deps.
- See `AGENTS.md` for coding style, commit practices, and security tips.
