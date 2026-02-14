import os
import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


st.set_page_config(page_title="Catalog PDF AI", page_icon="🛒", layout="wide")
st.title("🛒 Catalog PDF AI")
st.caption("Sube un catálogo en PDF y pregunta por características, diferencias y comparativas.")

with st.sidebar:
    st.header("Configuración")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Obtenla en console.groq.com"
    )
    hf_api_key = st.text_input(
        "Hugging Face API Key",
        type="password",
        help="Necesaria para descargar/usar modelos de Hugging Face en algunos entornos."
    )

    st.divider()
    st.header("Catálogo")
    uploaded_pdf = st.file_uploader("Sube el catálogo (PDF)", type="pdf")

    st.divider()
    st.header("Modelo Groq (LLM)")
    id_model = st.selectbox(
        "Modelo Groq",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"],
        index=0
    )
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.header("Embeddings (Hugging Face)")
    emb_model = st.selectbox(
        "Modelo embeddings",
        ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        index=0
    )

    st.divider()
    st.header("RAG")
    k = st.slider("k (chunks recuperados)", 2, 12, 6)
    chunk_size = st.slider("chunk_size", 300, 1400, 700, 50)
    chunk_overlap = st.slider("chunk_overlap", 0, 300, 80, 10)

# Validación de keys
if not groq_api_key or not hf_api_key:
    st.warning("Introduce ambas API Keys (Groq y Hugging Face) en la barra lateral para comenzar.")
    st.stop()

@st.cache_resource
def load_base_models(groq_key: str, hf_key: str, model_name: str, temp: float, embedding_name: str):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
    os.environ["HF_TOKEN"] = hf_key

    llm = ChatGroq(api_key=groq_key, model=model_name, temperature=temp)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_name,
        model_kwargs={"device": "cpu"}
    )
    return llm, embeddings

llm, embeddings = load_base_models(groq_api_key, hf_api_key, id_model, temperature, emb_model)

def build_retriever_from_pdf(file, chunk_size: int, chunk_overlap: int, k: int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file.getbuffer())
        pdf_path = tf.name

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load() 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " • ", " - ", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": k})


if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_fingerprint" not in st.session_state:
    st.session_state.pdf_fingerprint = None

def fingerprint_file(file) -> str:
    return f"{file.name}:{file.size}"

# Si cambia el PDF, reinicia el retriever
if uploaded_pdf is not None:
    fp = fingerprint_file(uploaded_pdf)
    if st.session_state.pdf_fingerprint != fp:
        st.session_state.pdf_fingerprint = fp
        st.session_state.retriever = None  # fuerza reindexado


if uploaded_pdf and st.session_state.retriever is None:
    with st.spinner("Indexando catálogo PDF..."):
        st.session_state.retriever = build_retriever_from_pdf(
            uploaded_pdf,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k
        )
    st.success("✅ Catálogo indexado. Ya puedes preguntar.")

if st.session_state.retriever is None:
    st.info("Sube un catálogo PDF para activar el chat.")
    st.stop()


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

system_prompt = (
    "You are a product-catalog assistant.\n"
    "You MUST answer using ONLY the retrieved context.\n"
    "If the context does not contain the needed info, say: 'No lo sé con la información del catálogo.'\n"
    "Create a table with the product information comparing products in the context.\n"
    "If you cite information, mention the page number when available.\n"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human",
     "User question: {input}\n\n"
     "Retrieved context:\n{context}\n\n"
     "Answer in Spanish.")
])

def format_docs(docs):
    out = []
    for d in docs:
        page = d.metadata.get("page", None)
        if page is not None:
            out.append(f"[page {page}]\n{d.page_content}")
        else:
            out.append(d.page_content)
    return "\n\n---\n\n".join(out)

chain = (
    {
        "context": st.session_state.retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

if prompt := st.chat_input("Pregunta por productos, características o 'compara A vs B'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = chain.invoke(prompt)
            clean_response = response.split("</think>")[-1].strip() if "</think>" in response else response
            st.write(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})
