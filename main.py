import os
import uvicorn
import sqlite3
import logging
import threading
import json

from fastapi import FastAPI, Request
from dotenv import load_dotenv

# --- LIBRERÍAS DE OPENAI ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para detección de idiomas
DetectorFactory.seed = 0
load_dotenv()

# Configuración de Logging profesional
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Forex AI Agent - OpenAI Powered")

# --- CONFIGURACIÓN DE RUTAS ---
DOCUMENTS_DIR = "documents"
DATABASE = "forex_memory.db" # Nombre específico para evitar conflictos

# --- ESTRATEGIA DE ACENTOS (Persuasión de Ventas) ---
COUNTRY_ADAPTATION = {
    "ar": {"acento": "argentino", "jerga": "Usa 'plata', 'che', 'vos'. Sé directo y seguro."},
    "mx": {"acento": "mexicano", "jerga": "Usa 'lana', 'platicar', 'ahorita'. Sé muy cordial."},
    "es": {"acento": "español de España", "jerga": "Usa 'vale', 'venga', 'invertir'. Sé rápido y profesional."},
    "uy": {"acento": "uruguayo", "jerga": "Usa 'bo', 'ta', 'dinero'. Sé cercano y confiable."},
    "default": {"acento": "neutro", "jerga": "Usa un español profesional estándar."}
}

vector_db = None
index_lock = threading.Lock()

# ---------------- BASE DE DATOS (Memoria Local) ----------------
def init_db():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            user_id TEXT, role TEXT, message TEXT, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(user_id, role, message):
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    conn.commit()
    conn.close()

def get_history(user_id, limit=4):
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [HumanMessage(content=m) if r == "user" else AIMessage(content=m) for r, m in rows]

# ---------------- MOTOR OPENAI ----------------
def process_message(user_id, message, country_code="default"):
    try:
        # Detección de tono
        lang = country_code if country_code != "default" else "es"
        try: 
            if country_code == "default": lang = detect(message)
        except: lang = "es"

        style = COUNTRY_ADAPTATION.get(lang, COUNTRY_ADAPTATION["default"])

        # RAG: Búsqueda Semántica
        contexto = ""
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=2)
                contexto = "\n".join([d.page_content for d in docs])

        # LLM (Busca automáticamente OPENAI_API_KEY)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            max_tokens=85 # Un poco más de margen para fluidez
        )

        system_prompt = f"""
        PERSONALIDAD: Eres Elena, experta en Forex e inversiones de alto nivel. 
        TONO: Tienes acento {style['acento']}. {style['jerga']}
        
        TAREA: Convence al cliente de la oportunidad de inversión de forma profesional y magnética.
        REGLA DE ORO: Máximo 20 palabras. Responde en el idioma que te hablen.

        CONTEXTO ESTRATÉGICO:
        {contexto if contexto else "Usa persuasión general sobre la volatilidad del mercado y rentabilidad."}
        """

        history = get_history(user_id)
        messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=message)]
        
        response = llm.invoke(messages)
        
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", response.content)
        
        return response.content

    except Exception as e:
        logging.error(f"Error OpenAI Forex: {e}")
        return "El mercado está movido hoy. ¿Qué me decías sobre tu capital de inversión?"

# ---------------- INDEXACIÓN ----------------
def build_index():
    global vector_db
    try:
        # OpenAIEmbeddings detecta automáticamente la llave
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                all_docs.extend(loader.load())
        
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                logging.info("--- DOCUMENTOS DE FOREX INDEXADOS ---")
    except Exception as e:
        logging.error(f"Error indexando PDFs: {e}")

# ---------------- WEBHOOK VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    try:
        data = await request.json()
        message_data = data.get("message", {})

        if "toolCalls" in message_data:
            tc = message_data["toolCalls"][0]
            func = tc.get("function", {})
            args = func.get("arguments", {})
            
            if isinstance(args, str):
                try: args = json.loads(args)
                except: args = {"query": args}
            
            query = args.get("query", "")
            if not query: query = "Hola" # Fallback para evitar errores
            
            customer_info = message_data.get("customer", {})
            country = customer_info.get("country", "default").lower()

            respuesta = process_message("vapi_forex_user", query, country_code=country)
            
            return {
                "results": [
                    {
                        "toolCallId": tc.get("id"),
                        "result": respuesta 
                    }
                ]
            }
    except Exception as e:
        logging.error(f"Error Webhook: {e}")
        return {"error": str(e)}
    
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def health_check():
    return {"status": "Elena Forex Online", "engine": "OpenAI Standard Key"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
