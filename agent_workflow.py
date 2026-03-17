import os
import ssl
import asyncio
import json
import httpx
import urllib3
from dotenv import load_dotenv

# ייבוא של Pinecone ו-LlamaIndex
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.base.llms.types import ChatMessage

# --- 1. פתרון SSL גלובלי עבור נטפרי ---
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# --- 2. הגדרת אירועים ---
class ExtractionEvent(Event):
    query_str: str

class SemanticSearchEvent(Event): # אירוע חדש כדי לנקות את התרשים
    query_str: str

class RetrievalEvent(Event):
    nodes: list
    query_str: str

class ValidationEvent(Event):
    response: str

# --- 3. בניית ה-Workflow ---
class SmartAgentWorkflow(Workflow):
    
    def _get_llm(self):
        api_key = os.getenv("COHERE_API_KEY")
        return Cohere(api_key=api_key, model="command-r-plus-08-2024")

    def _get_embed_model(self):
        api_key = os.getenv("COHERE_API_KEY")
        unsafe_client = httpx.Client(verify=False, trust_env=False)
        return CohereEmbedding(
            api_key=api_key, 
            model_name="embed-multilingual-v3.0",
            http_client=unsafe_client
        )

    @step
    async def route_query(self, ev: StartEvent) -> ExtractionEvent | SemanticSearchEvent:
        """שלב הנתב - מחליט אם ללכת ל-JSON או לחיפוש סמנטי"""
        Settings.llm = self._get_llm()
        print(f"🚦 הנתב מנתח: {ev.query}")
        
        prompt = (
            f"עליך להחליט לאן לנתב את השאלה: '{ev.query}'.\n"
            f"אם השאלה דורשת רשימה מפורטת, החלטות טכניות, או נתונים מובנים מקובץ ה-JSON - ענה DATA.\n"
            f"אם זו שאלת ידע כללית על הפרויקט - ענה SEARCH.\n"
            f"תשובה במילה אחת בלבד: DATA או SEARCH"
        )
        
        try:
            response = await Settings.llm.acomplete(prompt)
            choice = response.text.strip().upper()
            if "DATA" in choice:
                print("🔀 ניתוב ל-Structured Data (JSON)")
                return ExtractionEvent(query_str=ev.query)
            else:
                print("🔀 ניתוב ל-Semantic Search (Pinecone)")
                # במקום להחזיר StartEvent, מחזירים אירוע סמנטי ייעודי
                return SemanticSearchEvent(query_str=ev.query) 
        except Exception as e:
            print(f"⚠️ שגיאה בניתוב: {e}")
            return SemanticSearchEvent(query_str=ev.query)

    @step
    async def retrieve(self, ev: SemanticSearchEvent) -> RetrievalEvent:
        """שלב השליפה מ-Pinecone - מופעל רק אחרי הנתב"""
        Settings.embed_model = self._get_embed_model()
        
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            ssl_verify=False 
        )
        
        pinecone_index = pc.Index(host="https://agentic-docs-iaygeyt.svc.aped-4627-b74a.pinecone.io")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        print(f"🔎 מבצע חיפוש ב-Pinecone עבור: {ev.query_str}")
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(ev.query_str)
        return RetrievalEvent(nodes=nodes, query_str=ev.query_str)

    @step
    async def extract_structured_data(self, ev: ExtractionEvent) -> ValidationEvent:
        """שליפת נתונים מ-JSON מקומי"""
        print(f"📊 ניגש לקובץ הנתונים המובנה...")
        if not os.path.exists("structured_data.json"):
            return ValidationEvent(response="שגיאה: קובץ הנתונים structured_data.json לא נמצא.")
            
        with open("structured_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        context = json.dumps(data, indent=2, ensure_ascii=False)
        prompt = f"מבוסס על ה-JSON הבא, ענה על השאלה:\n{context}\n\nשאלה: {ev.query_str}"
        response = await Settings.llm.acomplete(prompt)
        return ValidationEvent(response=response.text)

    @step
    async def validate_and_generate(self, ev: RetrievalEvent) -> ValidationEvent:
        """יצירת תשובה סופית על בסיס המידע שנשלף מה-Vector DB"""
        if not ev.nodes:
            return ValidationEvent(response="לא נמצא מידע רלוונטי בתיעוד הפרויקט.")

        context = "\n".join([n.text for n in ev.nodes])
        content = f"ענה בצורה מקצועית על בסיס התיעוד הבא בלבד:\n{context}\n\nשאלה: {ev.query_str}"
        
        messages = [ChatMessage(role="user", content=content)]
        chat_response = await self._get_llm().achat(messages)
        return ValidationEvent(response=chat_response.message.content)

    @step
    async def final_step(self, ev: ValidationEvent) -> StopEvent:
        """סיום ה-Workflow והחזרת התוצאה"""
        print("🎯 התהליך הסתיים בהצלחה.")
        return StopEvent(result=ev.response)

# --- פונקציית ייצור התרשים ---
async def generate_workflow_diagram():
    from llama_index.utils.workflow import draw_all_possible_flows
    
    # מייצרים מופע כדי למנוע שגיאות self
    w = SmartAgentWorkflow()
    
    # ציור הזרימה
    draw_all_possible_flows(SmartAgentWorkflow, filename="workflow_diagram.html")
    print("✅ התרשים המעודכן נוצר בהצלחה! פתחי את workflow_diagram.html")

if __name__ == "__main__":
    asyncio.run(generate_workflow_diagram())