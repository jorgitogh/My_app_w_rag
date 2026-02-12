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

# ---------------- UI ----------------
st.set_page_config(page_title="Catalog PDF AI", page_icon="🛒", layout="wide")
st.title("🛒 Catalog PDF AI")
st.caption("Sube un catálogo en PDF y pregunta por características, diferencias y comparativas.")

with st.sidebar:
    st.header("Configuración")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Obtenla en console.groq.com")

    st.divider()
    st.header("Catálogo")
    uploaded_pdf = st.file_uploader("Sube el catálogo (PDF)", type="pdf")

    st.divider()
    st.header("Modelo")
    id_model = st.selectbox(
        "Modelo Groq",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"],
        index=0
    )
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.header("RAG")
    k = st.slider("k (chunks recuperados)", 2, 12, 6)
    chunk_size = st.slider("chunk_size", 300, 1400, 700, 50)
    chunk_overlap = st.slider("chunk_overlap", 0, 300, 80, 10)

if not groq_api_key:
    st.warning("Introduce la Groq API Key para comenzar.")
    st.stop()

@st.cache_resource
def load_base_models(groq_key: str, model_name: str, temp: float):
    llm = ChatGroq(api_key=groq_key, model=model_name, temperature=temp)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    return llm, embeddings

llm, embeddings = load_base_models(groq_api_key, id_model, temperature)

def build_retriever_from_pdf(file, chunk_size: int, chunk_overlap: int, k: int):
    # guarda el PDF temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(file.getbuffer())
        pdf_path = tf.name

    # carga texto del PDF
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()  # lista de Document, con metadata (página, etc.)

    # splitter sobre documentos (mejor que juntar todo, mantiene metadata por página)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " • ", " - ", ". ", " "]
    )
    chunks = splitter.split_documents(docs)

    # índice vectorial
    vs = FAISS.from_documents(chunks, embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": k})

# estado
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# procesa PDF
if uploaded_pdf and st.session_state.retriever is None:
    with st.spinner("Indexando catálogo PDF..."):
        st.session_state.retriever = build_retriever_from_pdf(
            uploaded_pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
    st.success("✅ Catálogo indexado. Ya puedes preguntar.")

# si aún no hay retriever
if st.session_state.retriever is None:
    st.info("Sube un catálogo PDF para activar el chat.")
    st.stop()

# pinta historial
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# prompt del sistema: forzar comparativas y trazabilidad
system_prompt = (
    "You are a product-catalog assistant.\n"
    "You MUST answer using ONLY the retrieved context.\n"
    "If the context does not contain the needed info, say: 'No lo sé con la información del catálogo.'\n"
    "When comparing products, create a compact comparison (bullet list or small table) with the attributes found.\n"
    "Always include citations to the pages/chunks using the metadata if available (e.g., page number).\n"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human",
     "User question: {input}\n\n"
     "Retrieved context:\n{context}\n\n"
     "Answer in Spanish.")
])

def format_docs(docs):
    # añade página si viene en metadata
    out = []
    for d in docs:
        page = d.metadata.get("page", None)
        if page is not None:
            out.append(f"[page {page}]\n{d.page_content}")
        else:
            out.append(d.page_content)
    return "\n\n---\n\n".join(out)

# chain: recupera docs -> formatea contexto -> llm
chain = (
    {
        "context": st.session_state.retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# input chat
if prompt := st.chat_input("Pregunta por productos, características o 'compara A vs B'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = chain.invoke(prompt)

            # por si el modelo devuelve tags raros de pensamiento
            clean_response = response.split("</think>")[-1].strip() if "</think>" in response else response

            st.write(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})
