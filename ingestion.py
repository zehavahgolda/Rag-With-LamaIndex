import os
import ssl
import urllib3
import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from dotenv import load_dotenv

# --- הגנות SSL עבור נטפרי (תקשורת עם Cohere) ---
os.environ['CURL_CA_BUNDLE'] = ""
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

def run_ingestion():
    print("🚀 Starting Local Ingestion (No Pinecone needed)...")
    
    # 1. הגדרת הנתיב המלא לתיקיית המקור
    # ה-r לפני המרכאות פותר את בעיית ה-Unicode ב-Windows
    data_path = r"C:\Users\PC\Desktop\לימודים\יד\מחצית ב\מלכה ברוק\Rag With LamaIndex\data_source"
    
    # 2. טעינת המסמכים (הוספנו recursive=True ליתר ביטחון)
    print(f"Reading documents from: {data_path}")
    reader = SimpleDirectoryReader(input_dir=data_path, recursive=True)
    documents = reader.load_data()
    
    if not documents:
        print("❌ No documents found! Please check if there are files in the data_source folder.")
        return

    print(f"✅ Loaded {len(documents)} documents.")

    # 3. יצירת מסד נתונים מקומי (ChromaDB)
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("my_docs")
    
    # 4. הגדרת ה-Vector Store וה-Storage Context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 5. הגדרת מודל ה-Embedding של Cohere
    embed_model = CohereEmbedding(
        cohere_api_key=os.environ["COHERE_API_KEY"],
        model_name="embed-multilingual-v3.0"
    )
    
    # 6. יצירת האינדקס ושמירה מקומית על הדיסק
    print("📤 Indexing documents locally...")
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    print("✅ SUCCESS! Data saved locally in /chroma_db folder.")

if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        print(f"❌ Error occurred: {e}")