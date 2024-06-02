import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configuración de Streamlit
st.set_page_config(page_title="LSDatasheet by Alfred - Pau", page_icon=":robot:", layout="wide")
st.title("LSDatasheet by Alfred - Pau")

# Aplicar estilo CSS para el fondo y la separación
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;
    }
    .separator {
        border-top: 2px solid #bbb;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .user-msg {
        background-color: #d4f4dd;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
    }
    .bot-msg {
        background-color: #e1e1e1;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
        white-space: pre-wrap;  /* Mantiene los saltos de línea y los espacios */
    }
    .bot-ref {
        background-color: #e1e1e1;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inicializar el historial de conversación
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'summary' not in st.session_state:
    st.session_state['summary'] = None

def split_text_into_chunks(text, chunk_size=512):
    """Divide el texto en fragmentos más pequeños"""
    text_chunks = []
    while len(text) > chunk_size:
        idx = text[:chunk_size].rfind('.')
        if idx == -1:
            idx = chunk_size
        text_chunks.append(text[:idx])
        text = text[idx:]
    if text:
        text_chunks.append(text)
    return text_chunks

def is_response_informative(response, threshold=5):
    """Comprueba si la respuesta tiene una cantidad mínima de palabras"""
    words = response.split()
    return len(words) > threshold

def are_query_words_in_pdf(query, pdf_text):
    """Comprueba si las palabras de la consulta están en el texto del PDF"""
    query_words = query.lower().split()
    pdf_words = pdf_text.lower().split()
    return all(word in pdf_words for word in query_words)

def process_pdf(uploaded_file):
    """Procesa el PDF y genera los embeddings y la base de datos vectorial"""
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Dividir el texto en trozos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    
    # Convertir textos en documentos
    documents = [Document(page_content=chunk) for chunk in texts]
    
    # Generar embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Crear base de datos vectorial
    vector_db = FAISS.from_documents(documents, embeddings)
    
    # Generar resumen del PDF
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text_chunks = split_text_into_chunks(text, chunk_size=1024)
    summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in text_chunks]
    summary = " ".join(summaries)
    
    return text, vector_db, summary

def get_response_and_references(user_query, text, vector_db):
    """Genera la respuesta y las referencias basadas en la consulta del usuario"""
    if not are_query_words_in_pdf(user_query, text):
        return "No trobo cap referència rellevant en el pdf proporcionat. Si necessites alguna cosa més, fes-m'ho saber.", []
    
    try:
        docs = vector_db.similarity_search(query=user_query, k=3)
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        context = "\n".join([doc.page_content for doc in docs])
        context_chunks = split_text_into_chunks(context, chunk_size=512)

        # Generar respuesta basada en los fragmentos relevantes
        answers = []
        for chunk in context_chunks:
            prompt = f"Context: {chunk}\nQuestion: {user_query}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
            outputs = model.generate(inputs.input_ids, max_new_tokens=50, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answers.append(response)

        final_answer = " ".join(answers).split('Answer:')[-1].strip()

        if not is_response_informative(final_answer):
            final_answer = "Disculpa, no tinc la informació necessària per donar una resposta."

        references = [doc.page_content for doc in docs]
        return final_answer, references

    except Exception as e:
        return f"Error: {str(e)}", []

def display_conversation_history():
    """Muestra el historial de conversación en estilo tipo WhatsApp"""
    st.subheader("Historial de conversació")
    for chat in st.session_state['history']:
        st.markdown(f"<div class='user-msg'><strong>Pregunta:</strong> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-msg'><strong>Resposta:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
        if chat['references']:
            st.markdown(f"<div class='bot-ref'><strong>Referències:</strong> {'; '.join(chat['references'])}</div>", unsafe_allow_html=True)
    
    if st.button("Neteja l'historial"):
        st.session_state['history'].clear()
        st.experimental_rerun()

# Definir columnas antes de usarlas
col1, col2 = st.columns([1, 2])

with col1:
    # Cargar y procesar el PDF
    uploaded_file = st.file_uploader("Afegeix el teu PDF aquí", type=["pdf"])
    if uploaded_file:
        text, vector_db, summary = process_pdf(uploaded_file)
        st.session_state['summary'] = summary
        st.success("Processat correcte.")

        # Mostrar contenido del PDF
        st.subheader("Contingut del PDF")
        st.text_area("Text del PDF", value=text, height=400)

        # Mostrar resumen inicial como bienvenida
        st.subheader("Benvingut a LSDatasheet!")
        st.write(f"El document tracta sobre: {summary}")
        st.write("Estaré encantat d’ajudar-te en qualsevol cosa que necessitis.")

with col2:
    if uploaded_file:
        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)  # Línea separadora
        # Realizar una consulta
        st.subheader("XatBot")
        user_query = st.text_input("En què et puc ajudar?")
        if user_query:
            final_answer, references = get_response_and_references(user_query, text, vector_db)
            st.write("Resposta:", final_answer)

            # Mostrar referencias utilizadas
            if references:
                st.subheader("Referències utilitzades")
                for ref in references:
                    st.write(ref)

            # Guardar historial de conversación
            st.session_state['history'].append({"user": user_query, "bot": final_answer, "references": references})

        display_conversation_history()
