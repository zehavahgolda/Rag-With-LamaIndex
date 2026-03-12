import os
import asyncio
import chromadb
from dotenv import load_dotenv

# טעינת המשתנים מקובץ ה-.env
load_dotenv()

from llama_index.core.workflow import (
    Event, 
    StartEvent, 
    StopEvent, 
    Workflow, 
    step
)
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.base.llms.types import ChatMessage

# --- 1. הגדרת אירועים ---
class RetrievalEvent(Event):
    """אירוע שקורה אחרי שליפת המידע"""
    nodes: list
    query_str: str

class ValidationEvent(Event):
    """אירוע שקורה אחרי שהמידע עבר אישור או נכשל"""
    response: str
    nodes: list

# --- 2. בניית ה-Workflow ---
class SmartAgentWorkflow(Workflow):
    
    @step
    async def retrieve(self, ev: StartEvent) -> RetrievalEvent:
        """שלב השליפה מה-ChromaDB"""
        api_key = os.getenv("COHERE_API_KEY")
        
        # הגדרת מודלים
        Settings.embed_model = CohereEmbedding(api_key=api_key, model_name="embed-multilingual-v3.0")
        Settings.llm = Cohere(api_key=api_key, model="command-r-plus-08-2024")
        
        # חיבור למסד הנתונים
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("my_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        print(f"🔎 מחפש מידע עבור: {ev.query}")
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(ev.query)
        
        return RetrievalEvent(nodes=nodes, query_str=ev.query)

    @step
    async def validate_and_generate(self, ev: RetrievalEvent) -> ValidationEvent:
        """שלב הוולידציה והניסוח עם הגבלה למסמכים בלבד"""
        
        # אם לא נמצאו מסמכים רלוונטיים
        if not ev.nodes:
            print(f"⚠️ וולידציה נכשלה: לא נמצא מידע במסמכים עבור: {ev.query_str}")
            return ValidationEvent(
                response="אני מצטער, אבל אין לי מידע על כך במסמכי המערכת. אני מתוכנת לענות רק על שאלות הקשורות למסמכים שסופקו לי.",
                nodes=[]
            )

        print("✅ נמצא מידע רלוונטי, מנסח תשובה קפדנית...")
        context = "\n".join([n.text for n in ev.nodes])
        
        # הנחיה קשיחה שמכריחה את המודל להשתמש רק ב-Context
        content = (
            f"אתה סוכן AI מקצועי. תפקידך לענות על השאלה אך ורק על בסיס המידע המסופק להלן. "
            f"אם התשובה אינה מופיעה במידע המסופק, אמור בפירוש שאינך יודע.\n\n"
            f"המידע מהמסמכים:\n{context}\n\n"
            f"שאלה: {ev.query_str}\n"
            f"תשובה:"
        )
        
        messages = [ChatMessage(role="user", content=content)]
        
        # הרצה של ה-Chat API
        chat_response = await Settings.llm.achat(messages)
        final_text = chat_response.message.content
        
        return ValidationEvent(response=final_text, nodes=ev.nodes)

    @step
    async def final_step(self, ev: ValidationEvent) -> StopEvent:
        """שלב סיום והחזרת התוצאה"""
        print("🎯 ה-Workflow הסתיים בהצלחה.")
        return StopEvent(result=ev.response)

# --- 3. פונקציית הרצה לבדיקה ---
async def main():
    agent = SmartAgentWorkflow(timeout=60)
    query = "מה הצבע העיקרי של המערכת?"
    result = await agent.run(query=query)
    print(f"\n🤖 תשובה: {result}")

if __name__ == "__main__":
    asyncio.run(main())