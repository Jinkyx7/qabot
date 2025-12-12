"""
Simple RAG chatbot over PDFs using open-source embeddings and a Hugging Face Hub LLM.
"""

from pathlib import Path
import logging
import os
import json
import socket
from typing import Iterable, List

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dotenv import load_dotenv
import gradio as gr

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()  # auto-load .env for HF_TOKEN


def _patch_hf_inference_client_post():
    """Add a .post method to InferenceClient classes if missing (hf-hub >=0.36)."""
    from huggingface_hub import AsyncInferenceClient, InferenceClient  # type: ignore

    if not hasattr(InferenceClient, "post"):
        def _post(self, *, json=None, stream=False, task=None):
            inputs = (json or {}).get("inputs")
            params = (json or {}).get("parameters") or {}
            if stream:
                # LangChain's HF endpoint wrapper doesn't stream via .post
                raise NotImplementedError("Streaming via .post is not supported.")
            output = self.text_generation(inputs, **params)
            return json.dumps([{"generated_text": output}]).encode()

        InferenceClient.post = _post  # type: ignore[attr-defined]

    if not hasattr(AsyncInferenceClient, "post"):
        async def _apost(self, *, json=None, stream=False, task=None):
            inputs = (json or {}).get("inputs")
            params = (json or {}).get("parameters") or {}
            if stream:
                raise NotImplementedError("Streaming via .post is not supported.")
            output = await self.text_generation(inputs, **params)
            return json.dumps([{"generated_text": output}]).encode()

        AsyncInferenceClient.post = _apost  # type: ignore[attr-defined]


def get_llm():
    """Create a local transformers LLM pipeline (no remote inference)."""
    model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "256"))
    temperature = float(os.getenv("HF_TEMPERATURE", "0.2"))
    device_map = os.getenv("HF_DEVICE_MAP", "auto")
    torch_dtype = os.getenv("HF_TORCH_DTYPE", "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return HuggingFacePipeline(
        pipeline=text_gen,
        model_kwargs={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    )


def document_loader(file) -> List:
    """Load PDF pages into LangChain Documents."""
    # Preserve full paths for pathlib.Path inputs; fall back to file-like .name
    if isinstance(file, (str, Path)):
        path = Path(file)
    else:
        path = Path(getattr(file, "name"))
    try:
        loader = PyPDFLoader(str(path))
        loaded_document = loader.load()
        return loaded_document
    except Exception as exc:  # pragma: no cover - logged and re-raised
        logger.exception("Failed to load PDF: %s", path)
        raise RuntimeError(f"Unable to load PDF: {path}") from exc


def text_splitter(data: Iterable) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks


def embedding_model():
    """Use a compact sentence-transformer for embeddings."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


def vector_database(chunks):
    embedder = embedding_model()
    try:
        vectordb = Chroma.from_documents(chunks, embedder)
        return vectordb
    except Exception as exc:  # pragma: no cover - logged and re-raised
        logger.exception("Failed to build vector store")
        raise RuntimeError("Unable to build vector store") from exc


def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever


def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True,
    )
    try:
        response = qa.invoke({"query": query})
        full_answer = response["result"]
        # Trim common prompt preamble if the model echoes it back.
        if "Helpful Answer:" in full_answer:
            trimmed = full_answer.split("Helpful Answer:", 1)[1].strip()
        else:
            trimmed = full_answer.strip()
        logger.info("----- Model output start -----")
        logger.info("Full LLM response: %s", full_answer)
        logger.info("----- Model output end -----")
        return trimmed
    except Exception as exc:  # pragma: no cover - logged and handled
        logger.exception("QA chain failed")
        return "An error occurred while generating the answer. Please retry."


def build_interface():
    """Construct the Gradio interface."""
    return gr.Interface(
        fn=retriever_qa,
        inputs=[
            gr.File(
                label="Upload PDF File",
                file_count="single",
                file_types=[".pdf"],
                type="file",
            ),
            gr.Textbox(
                label="Input Query", lines=2, placeholder="Type your question here..."
            ),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="RAG Chatbot (Open Source Models)",
        description="Upload a PDF and ask questions. The chatbot answers using retrieved chunks from the document.",
        analytics_enabled=False,
    )


if __name__ == "__main__":
    rag_application = build_interface()
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        port = int(env_port)
    else:
        # Prefer 7860; if unavailable, grab an ephemeral free port.
        preferred_port = 7860
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", preferred_port))
                port = preferred_port
        except OSError:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]

    rag_application.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
    )
