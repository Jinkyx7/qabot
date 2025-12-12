# RAG PDF QA Bot

This project serves a simple retrieval-augmented generation (RAG) chatbot that answers questions about an uploaded PDF using open-source Hugging Face models, LangChain, and a Gradio UI. The LLM now runs locally via `transformers` with a default Qwen 7B Instruct model (configurable), and embeddings use `sentence-transformers/all-mpnet-base-v2`.

## Demo

Demo video:

[![Demo – RAG PDF QA Bot](https://img.youtube.com/vi/UlKbUZWjbB4/0.jpg)](https://youtu.be/UlKbUZWjbB4)

## Prerequisites

- Python 3.10+
- Hugging Face token (set `HF_TOKEN` or login) if the configured model requires gated access; models are downloaded locally.
- Sufficient local resources for the chosen `HF_MODEL_ID` (defaults to `Qwen/Qwen2.5-7B-Instruct`; adjust if hardware-constrained).

## Pipeline Flow

```
PDF upload
   ↓
Gradio UI (file + query input)
   ↓
LangChain PyPDFLoader (document_loader)
   ↓
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
   ↓
Embeddings: sentence-transformers/all-mpnet-base-v2 (HuggingFaceEmbeddings)
   ↓
Vector store: Chroma (from_documents → as_retriever)
   ↓
LLM: local transformers pipeline (HF_MODEL_ID, default Qwen/Qwen2.5-1.5B-Instruct)
   ↓
RetrievalQA chain (stuff) returns answer
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Dependencies are pinned (e.g., `langchain==0.3.0`, `langchain-huggingface==0.1.0`, `gradio==3.50.2`) to match the current code paths and avoid recent breaking API changes.
- Local LLM dependencies: `transformers`, `accelerate` (installed via `requirements.txt`).
- Using a `.env` file (recommended):
  - Create `.env` (already gitignored) with your token and model settings. Example:
    ```
    HF_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
    HF_MAX_NEW_TOKENS=256
    HF_TEMPERATURE=0.2
    HF_DEVICE_MAP=auto
    HF_TORCH_DTYPE=auto
    HF_TOKEN=your_token_here  # optional if model is ungated
    ```
  - The app runs `python-dotenv` on startup, so no manual `export` is needed.

## Running Locally

```bash
cp .env.example .env  # create and fill in env vars, then:
python qabot.py
```

- The app binds to `127.0.0.1` and uses `GRADIO_SERVER_PORT` if set; otherwise it prefers `7860` and falls back to a free port if occupied. The port is logged on startup.
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
- Configure the local LLM via environment variables:
  - `HF_MODEL_ID` (default `Qwen/Qwen2.5-7B-Instruct`)
  - `HF_MAX_NEW_TOKENS` (default `256`)
  - `HF_TEMPERATURE` (default `0.2`)
  - `HF_DEVICE_MAP` (default `auto`)
  - `HF_TORCH_DTYPE` (default `auto`)
  - For smaller/faster downloads, set `HF_MODEL_ID` to a lighter instruct model (e.g., `Qwen/Qwen2.5-1.5B-Instruct` or `microsoft/Phi-3-mini-4k-instruct`).

## Project Structure

- `qabot.py`: Gradio app wiring PDF loading, text splitting (chunk size 1000 / overlap 200), open-source embeddings/LLM (local transformers), Chroma vector store, and a RetrievalQA chain. All runtime artifacts stay in memory and are not persisted.

## Development Notes

- Functions are structured to keep the pipeline modular (loader → splitter → embedder → vector store → retriever → QA chain).
- Add tests with `pytest` as logic grows (e.g., chunking behavior, retriever wiring). Run with `pytest` after installing dev deps.
