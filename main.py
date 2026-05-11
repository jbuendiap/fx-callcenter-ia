import os
import uvicorn
import sqlite3
import logging
import threading
import json

from fastapi import FastAPI, Request
from dotenv import load_dotenv

# Librerías de Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langdetect import detect, DetectorFactory

# Estabilidad para detección de idiomas
DetectorFactory.seed = 0
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Forex AI Agent - Gemini Powered")

# --- CONFIGURACIÓN DE RUTAS ---
DOCUMENTS_DIR = "documents"
DATABASE = "memory.db"

# --- ESTRATEGIA DE ACENTOS POR PAÍS ---
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
    conn = sqlite3.connect(DATABASE)
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
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    conn.commit()
    conn.close()

def get_history(user_id, limit=4):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [HumanMessage(content=m) if r == "user" else AIMessage(content=m) for r, m in rows]

# ---------------- MOTOR GEMINI 1.5 FLASH ----------------
def process_message(user_id, message, country_code="default"):
    try:
        # 1. Tono y acento
        lang = country_code if country_code != "default" else "es"
        try: 
            if country_code == "default": lang = detect(message)
        except: lang = "es"

        style = COUNTRY_ADAPTATION.get(lang, COUNTRY_ADAPTATION["default"])

        # 2. RAG: Buscar info en PDFs
        contexto = ""
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=2)
                contexto = "\n".join([d.page_content for d in docs])

        # 3. Configuración Gemini
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.6,
            max_output_tokens=100
        )

        system_prompt = f"""
        PERSONALIDAD: Eres experto en Forex he inversiones. 
        TONO: Tienes acento {style['acento']}. {style['jerga']}
        
        TAREA: Convence al cliente de la oportunidad de inversión sin sonar desesperado.
        REGLA DE ORO: Máximo 20 palabras por respuesta. Usa el CONTEXTO para dar datos reales.

        CONTEXTO DE DOCUMENTOS:
        {contexto if contexto else "Usa persuasión general en Forex basada en rentabilidad segura."}
        """

        history = get_history(user_id)
        messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=message)]
        
        response = llm.invoke(messages)
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", response.content)
        
        return response.content

    except Exception as e:
        logging.error(f"Error en Gemini: {e}")
        return "Disculpe, la señal está fallando un poco. ¿Qué me decía sobre la inversión?"

# ---------------- INDEXACIÓN DE DOCUMENTOS ----------------
def build_index():
    global vector_db
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(DOCUMENTS_DIR, file))
                all_docs.extend(loader.load())
        
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                logging.info("--- DOCUMENTOS INDEXADOS ---")
    except Exception as e:
        logging.error(f"Error indexando: {e}")

# ---------------- WEBHOOK PARA VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    try:
        data = await request.json()
        message_data = data.get("message", {})

        if "toolCalls" in message_data:
            tc = message_data["toolCalls"][0]
            args = tc.get("function", {}).get("arguments", {})
            
            # Manejo de argumentos si vienen como string
            if isinstance(args, str):
                args = json.loads(args)
            
            query = args.get("query", "")
            
            # Obtener país del lead
            customer_info = message_data.get("customer", {})
            country = customer_info.get("country", "default").lower()

            respuesta = process_message("vapi_user", query, country_code=country)
            
            # Formato exacto para que Vapi llene el campo 'result'
            return {
                "results": [
                    {
                        "toolCallId": tc.get("id"),
                        "result": respuesta 
                    }
                ]
            }
    except Exception as e:
        logging.error(f"Error en Webhook: {e}")
        return {"error": str(e)}
    
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def health_check():
    return {"status": "Vendedor Forex Activo", "project": "FX-CallCenter"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
