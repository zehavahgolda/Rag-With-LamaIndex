import os
import ssl
import urllib3
# 1. החלפת הייבוא של Chroma ב-Pinecone
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.cohere import CohereEmbedding
from dotenv import load_dotenv

# --- הגנות SSL עבור נטפרי ---
os.environ['CURL_CA_BUNDLE'] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

def run_ingestion():
    print("🚀 Starting Pinecone Ingestion...")
    
    # הגדרת נתיב המקור
    data_path = r"C:\Users\PC\Desktop\לימודים\יד\מחצית ב\מלכה ברוק\Rag With LamaIndex\data_source"
    
    # טעינת המסמכים
    reader = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    documents = reader.load_data()
    
    if not documents:
        print("❌ No documents found!")
        return

    print(f"✅ Loaded {len(documents)} documents.")

    # 2. התחברות ל-Pinecone באמצעות ה-Host שנטפרי אישרו
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    
    # השתמשי בקישור שנטפרי פתחו לך כאן
    pinecone_index = pc.Index(host="https://agentic-docs-iaygeyt.svc.aped-4627-b74a.pinecone.io")
    
    # 3. הגדרת ה-Vector Store עבור Pinecone
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. הגדרת מודל ה-Embedding
    embed_model = CohereEmbedding(
        cohere_api_key=os.environ["COHERE_API_KEY"],
        model_name="embed-multilingual-v3.0"
    )
    
    # 5. יצירת האינדקס והעלאה לענן (Pinecone)
    print("📤 Uploading and Indexing documents to Pinecone cloud...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    print("✅ SUCCESS! Data is now hosted on Pinecone.")

if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        print(f"❌ Error occurred: {e}")