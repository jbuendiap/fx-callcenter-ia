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

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI(title="Forex AI Agent - OpenAI Powered")

# --- CONFIGURACIÓN DE RUTAS ---
DOCUMENTS_DIR = "documents"
DATABASE = "memory.db"

# --- ESTRATEGIA DE ACENTOS ---
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
    # check_same_thread=False es vital para FastAPI
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
        # 1. Determinación de tono
        lang = country_code if country_code != "default" else "es"
        try: 
            if country_code == "default": lang = detect(message)
        except: lang = "es"

        style = COUNTRY_ADAPTATION.get(lang, COUNTRY_ADAPTATION["default"])

        # 2. RAG: Búsqueda Semántica
        contexto = ""
        if vector_db is not None:
            with index_lock:
                docs = vector_db.similarity_search(message, k=2)
                contexto = "\n".join([d.page_content for d in docs])

        # 3. LLM (Usa automáticamente la variable OPENAI_API_KEY)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            max_tokens=80
        )

        system_prompt = f"""
        PERSONALIDAD: Eres Elena, experta en Forex e inversiones. 
        TONO: Tienes acento {style['acento']}. {style['jerga']}
        
        TAREA: Convence al cliente de la oportunidad de inversión de forma profesional.
        REGLA DE ORO: Máximo 20 palabras por respuesta. Responde siempre en el idioma que te hablen.

        CONTEXTO ESTRATÉGICO:
        {contexto if contexto else "Usa persuasión general en Forex basada en rentabilidad segura."}
        """

        history = get_history(user_id)
        messages = [SystemMessage(content=system_prompt), *history, HumanMessage(content=message)]
        
        response = llm.invoke(messages)
        
        # Persistencia
        save_message(user_id, "user", message)
        save_message(user_id, "assistant", response.content)
        
        return response.content

    except Exception as e:
        logging.error(f"Error en proceso OpenAI: {e}")
        return "Disculpe, la señal está fallando un poco. ¿Qué me decía sobre la inversión?"

# ---------------- INDEXACIÓN DE DOCUMENTOS ----------------
def build_index():
    global vector_db
    try:
        if not os.getenv("OPENAI_API_KEY"):
            logging.error("CRÍTICO: No se encontró OPENAI_API_KEY en las variables de entorno.")
            return

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if not os.path.exists(DOCUMENTS_DIR): os.makedirs(DOCUMENTS_DIR)
        
        all_docs = []
        for file in os.listdir(DOCUMENTS_DIR):
            if file.endswith(".pdf"):
                path = os.path.join(DOCUMENTS_DIR, file)
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
        
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            chunks = splitter.split_documents(all_docs)
            with index_lock:
                vector_db = FAISS.from_documents(chunks, embeddings)
                logging.info("--- DOCUMENTOS INDEXADOS EXITOSAMENTE ---")
        else:
            logging.info("No se encontraron documentos PDF para indexar.")

    except Exception as e:
        logging.error(f"Error indexando documentos: {e}")

# ---------------- WEBHOOK PARA VAPI ----------------
@app.post("/vapi-webhook")
async def vapi_webhook(request: Request):
    try:
        data = await request.json()
        message_data = data.get("message", {})

        # Manejo de Tool Calls de Vapi
        if "toolCalls" in message_data:
            tc = message_data["toolCalls"][0]
            func = tc.get("function", {})
            args = func.get("arguments", {})
            
            # Asegurar que args sea un diccionario
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {"query": args}
            
            query = args.get("query", "")
            customer_info = message_data.get("customer", {})
            country = customer_info.get("country", "default").lower()

            respuesta = process_message("vapi_user", query, country_code=country)
            
            return {
                "results": [
                    {
                        "toolCallId": tc.get("id"),
                        "result": respuesta 
                    }
                ]
            }
    except Exception as e:
        logging.error(f"Error en Webhook Vapi: {e}")
        return {"error": str(e)}
    
    return {"ok": True}

@app.on_event("startup")
async def startup():
    init_db()
    # Ejecutar indexación en hilo separado para no bloquear el inicio
    threading.Thread(target=build_index, daemon=True).start()

@app.get("/")
def health_check():
    return {
        "status": "Vendedor Forex Activo", 
        "engine": "OpenAI GPT-4o-Mini",
        "rag_ready": vector_db is not None
    }

if __name__ == "__main__":
    # Railway asigna el puerto dinámicamente
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
