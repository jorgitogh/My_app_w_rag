# Catalog PDF AI (RAG with Streamlit)

Aplicacion web en Streamlit para hacer preguntas sobre catalogos PDF usando un flujo RAG (Retrieval-Augmented Generation).

Permite:
- Subir un catalogo en PDF.
- Indexar el contenido en FAISS con embeddings de Hugging Face.
- Consultar el catalogo en lenguaje natural.
- Pedir comparativas de productos en tabla.
- Responder en espanol con referencia de pagina cuando exista en el contexto.

## Tech stack
- `Python`
- `Streamlit`
- `LangChain`
- `Groq` (LLM)
- `Hugging Face` (embeddings)
- `FAISS` (vector store)
- `PyMuPDF` (lectura de PDF)

## Requisitos
- Python 3.10+ recomendado.
- API key de Groq.
- API key de Hugging Face.

Dependencias en `requirements.txt`:
- `streamlit`
- `python-dotenv`
- `pandas`
- `openpyxl`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-groq`
- `langchain-huggingface`
- `faiss-cpu`
- `pymupdf`
- `sentence-transformers`

## Instalacion
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion
Desde la raiz del proyecto:

```bash
streamlit run app.py
```

Se abrira la UI de Streamlit en el navegador.

## Uso
1. En la barra lateral, ingresa:
- `Groq API Key`
- `Hugging Face API Key`

2. Sube un archivo PDF de catalogo.

3. Ajusta parametros opcionales:
- Modelo Groq (`llama-3.3-70b-versatile` o `llama-3.1-70b-versatile`)
- Temperatura
- Modelo de embeddings
- Parametros RAG (`k`, `chunk_size`, `chunk_overlap`)

4. Espera a que termine el indexado.

5. Haz preguntas como:
- "Que caracteristicas tiene el modelo X?"
- "Compara X vs Y"
- "Que diferencias hay entre los productos de la linea A?"

## Como funciona (resumen)
1. El PDF se carga y se divide en chunks con `RecursiveCharacterTextSplitter`.
2. Cada chunk se vectoriza con `HuggingFaceEmbeddings`.
3. Los vectores se guardan en `FAISS`.
4. Para cada pregunta, se recuperan los chunks mas relevantes (`k`).
5. El LLM de Groq genera la respuesta usando solo el contexto recuperado.

## Notas
- Si no hay contexto suficiente, el asistente esta instruido para responder: `No lo se con la informacion del catalogo.`
- El retriever se reconstruye cuando cambias de PDF.
- En algunos entornos, el token de Hugging Face es necesario para descargar modelos.

## Archivo principal
- `app.py`: interfaz Streamlit, configuracion del pipeline RAG, indexado y chat.

## Futuras mejoras sugeridas
- Persistencia de indices FAISS en disco.
- Soporte para multiples PDFs.
- Historial exportable de conversaciones.
- Filtros por paginas o secciones del catalogo.
