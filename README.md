# RAG PDF QA Bot

This project serves a simple retrieval-augmented generation (RAG) chatbot that answers questions about an uploaded PDF using open-source Hugging Face models, LangChain, and a Gradio UI.

## Prerequisites

- Python 3.10+
- Hugging Face token with access to the configured models (`huggingface-cli login` or set `HF_TOKEN`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Dependencies are pinned (e.g., `langchain==0.3.0`, `langchain-huggingface==0.1.0`, `gradio==3.50.2`) to match the current code paths and avoid recent breaking API changes.

## Running Locally

```bash
python qabot.py
```

- The Gradio interface starts on `http://0.0.0.0:7860/`.
- Upload a single PDF and ask a question to trigger retrieval and generation.
- Logs emit to stdout; check the console for loader or QA chain errors.

## Tests

- Install dev deps (`pytest`, `fpdf2`) via `pip install -r requirements.txt`.
- Run smoke tests for the loader/splitter pipeline:

```bash
pytest
```

## Configuration

- Edit model IDs and generation parameters in `qabot.py` (e.g., `model_id` for the LLM, `chunk_size`/`chunk_overlap` for the splitter).
- Provide credentials via environment variables; avoid hard-coding secrets. The app auto-loads `.env` (see below) so `HF_TOKEN` can be stored there.
- To adjust chunking, tweak `chunk_size`/`chunk_overlap`; larger chunks improve context but increase token usage.

### Using a .env file
- Create `.env` (already gitignored) with your token: `HF_TOKEN=your_token_here` (or `HUGGINGFACEHUB_API_TOKEN=...`).
- The app runs `python-dotenv` on startup, so no manual `export` is needed.

## Project Structure

- `qabot.py`: Gradio app wiring PDF loading, text splitting, open-source embeddings/LLM (Hugging Face Hub), Chroma vector store, and a RetrievalQA chain.
- `AGENTS.md`: Repository guidelines for contributors.

## Development Notes

- Functions are structured to keep the pipeline modular (loader → splitter → embedder → vector store → retriever → QA chain).
- Add tests with `pytest` as logic grows (e.g., chunking behavior, retriever wiring). Run with `pytest` after installing dev deps.
- See `AGENTS.md` for coding style, commit practices, and security tips.
