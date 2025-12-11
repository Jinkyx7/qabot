from pathlib import Path
import logging

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

## LLM
def get_llm():
    model_id = 'ibm/granite-3-2-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # Specify the max tokens you want to generate
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

## Document loader
def document_loader(file):
    """Load PDF pages into LangChain Documents."""
    path = Path(file.name if hasattr(file, "name") else file)
    try:
        loader = PyPDFLoader(str(path))
        loaded_document = loader.load()
        return loaded_document
    except Exception as exc:  # pragma: no cover - logged and re-raised
        logger.exception("Failed to load PDF: %s", path)
        raise RuntimeError(f"Unable to load PDF: {path}") from exc

## Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks

## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,  # retain context for retrieval
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    try:
        vectordb = Chroma.from_documents(chunks, embedding_model)
        return vectordb
    except Exception as exc:  # pragma: no cover - logged and re-raised
        logger.exception("Failed to build vector store")
        raise RuntimeError("Unable to build vector store") from exc

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type="stuff", 
                                    retriever=retriever_obj, 
                                    return_source_documents=True)
    try:
        response = qa.invoke({"query": query})
        return response['result']
    except Exception as exc:  # pragma: no cover - logged and handled
        logger.exception("QA chain failed")
        return "An error occurred while generating the answer. Please retry."

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="file"),  # Accept file object to expose .name
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Chatbot with IBM watsonx.ai",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7860)
