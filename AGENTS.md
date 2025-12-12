# Repository Guidelines

## Project Structure & Module Organization
- `qabot.py`: Single entrypoint running a Gradio UI that wires together document loading (PDF), chunking, Hugging Face Hub LLM, sentence-transformer embeddings, Chroma vector store, and a RetrievalQA chain. No other source or asset directories are present; uploaded PDFs are handled in-memory at runtime.
- Runtime artifacts (Chroma index) are created in-process and not persisted; no cache or storage directories are expected in the repo.

## Setup, Build, and Local Run
- Use Python 3.10+ and a virtual environment: `python -m venv .venv && source .venv/bin/activate`.
- Install runtime deps: `pip install -r requirements.txt` (LangChain, Gradio, Chroma, Hugging Face Hub, sentence-transformers, dotenv).
- Launch the app locally with `python qabot.py`; the interface serves at `http://0.0.0.0:7860/`.
- Provide a Hugging Face token via `.env` (auto-loaded) or `huggingface-cli login` for hub-hosted LLMs and embeddings.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and snake_case for functions/variables; keep functions small and composable (loader, splitter, embedder, retriever).
- Keep configuration constants near the top (model IDs, chunk sizes, generation params) and pass dependencies through functions rather than using globals.
- Add docstrings for public-facing functions and short inline comments only where behavior is non-obvious (e.g., why a specific model or parameter is chosen).

## Testing Guidelines
- Smoke tests live in `tests/` and can be run with `pytest`; they validate PDF loading and text splitting using generated sample PDFs.
- For UI changes, run `python qabot.py` and manually verify upload, query, and response paths; test with small and large PDFs to validate chunking/latency.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (e.g., `Add retriever timeouts`, `Document Watsonx config`). There is no prior history, so establish consistency now.
- In pull requests, include: a short summary, the primary behavior change, validation steps (commands run), and any credential assumptions. Attach screenshots/GIFs for UI-impacting changes.
- Keep diffs minimal and localized; avoid reformatting unrelated sections unless applying a consistent formatter across the file.

## Security & Configuration Tips
- Do not commit credentials or tokens. Prefer environment variables or local config files ignored by git.
- If changing model IDs, document required access/quotas. Avoid logging PDF contents or user queries in production deployments.
