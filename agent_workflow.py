import os
import ssl
import asyncio
import chromadb
import json
import httpx
from dotenv import load_dotenv

# --- 1. פתרון SSL גלובלי ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.base.llms.types import ChatMessage

# --- 2. הגדרת אירועים ---
class ExtractionEvent(Event):
    query_str: str

class RetrievalEvent(Event):
    nodes: list
    query_str: str

class ValidationEvent(Event):
    response: str

# --- 3. בניית ה-Workflow ---
class SmartAgentWorkflow(Workflow):
    
    def _get_llm(self):
        """הגדרה מעודכנת ללא שגיאת http_client"""
        api_key = os.getenv("COHERE_API_KEY")
        # אנחנו מסתמכים על הגדרות ה-SSL הגלובליות ששמת בתחילת הקוד
        return Cohere(
            api_key=api_key, 
            model="command-r-plus-08-2024"
        )

    def _get_embed_model(self):
        """פונקציית עזר להגדרת Embedding - כולל עקיפת נטפרי ו-SSL"""
        api_key = os.getenv("COHERE_API_KEY")
        unsafe_client = httpx.Client(verify=False, trust_env=False)
        return CohereEmbedding(
            api_key=api_key, 
            model_name="embed-multilingual-v3.0",
            http_client=unsafe_client
        )

    @step
    async def route_query(self, ev: StartEvent) -> ExtractionEvent | StartEvent:
        """שלב הנתב (Router) - מחליט לאן ללכת"""
        Settings.llm = self._get_llm()
        
        print(f"🚦 הנתב מנתח: {ev.query}")
        
        prompt = (
            f"עליך להחליט לאן לנתב: '{ev.query}'.\n"
            f"אם השאלה כוללת את המילה 'JSON', 'החלטות', או 'רשימה' - ענה אך ורק DATA.\n"
            f"אחרת - ענה SEARCH.\n"
            f"תשובה במילה אחת בלבד: DATA"
        )
        
        try:
            response = await Settings.llm.acomplete(prompt)
            choice = response.text.strip().upper()
            
            if "DATA" in choice:
                print(f"🔀 הנתב בחר: Structured Data (JSON)")
                return ExtractionEvent(query_str=ev.query)
            else:
                print(f"🔀 הנתב בחר: Semantic Search (ChromaDB)")
                return ev 
        except Exception as e:
            print(f"⚠️ שגיאה בנתב: {e}. ממשיך לחיפוש רגיל.")
            return ev

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievalEvent:
        """שלב השליפה מה-ChromaDB"""
        Settings.embed_model = self._get_embed_model()
        
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("my_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        print(f"🔎 מבצע חיפוש עבור: {ev.query}")
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(ev.query)
        return RetrievalEvent(nodes=nodes, query_str=ev.query)

    @step
    async def extract_structured_data(self, ev: ExtractionEvent) -> ValidationEvent:
        """שלב ג': שליפת נתונים מובנים מ-JSON"""
        print(f"📊 ניגש לקובץ הנתונים המובנה...")
        try:
            if not os.path.exists("structured_data.json"):
                return ValidationEvent(response="קובץ הנתונים structured_data.json לא נמצא.")
                
            with open("structured_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            context = json.dumps(data, indent=2, ensure_ascii=False)
            prompt = f"מבוסס על ה-JSON הבא, ענה על השאלה בצורה תמציתית:\nנתונים:\n{context}\n\nשאלה: {ev.query_str}"
            
            response = await Settings.llm.acomplete(prompt)
            return ValidationEvent(response=response.text)
        except Exception as e:
            return ValidationEvent(response=f"שגיאה בקריאת הנתונים: {e}")

    @step
    async def validate_and_generate(self, ev: RetrievalEvent) -> ValidationEvent:
        """שלב הניסוח (RAG)"""
        if not ev.nodes:
            return ValidationEvent(response="לא נמצא מידע רלוונטי.")

        context = "\n".join([n.text for n in ev.nodes])
        content = f"ענה אך ורק לפי ההקשר הבא:\n{context}\n\nשאלה: {ev.query_str}"
        
        messages = [ChatMessage(role="user", content=content)]
        
        # שימוש בלקוח הלא-מאובטח גם כאן
        custom_llm = self._get_llm()
        chat_response = await custom_llm.achat(messages)
        return ValidationEvent(response=chat_response.message.content)

    @step
    async def final_step(self, ev: ValidationEvent) -> StopEvent:
        print("🎯 התהליך הסתיים.")
        return StopEvent(result=ev.response)

# --- הרצה לבדיקה ---
if __name__ == "__main__":
    async def run_test():
        agent = SmartAgentWorkflow(timeout=60)
        result = await agent.run(query="הצג רשימת החלטות מה-JSON")
        print(f"\n🤖 תוצאה סופית: {result}")
    
    asyncio.run(run_test())