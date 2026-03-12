import streamlit as st
import os
import ssl
import urllib3
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.postprocessor import SimilarityPostprocessor

# --- הגנות נטפרי ---
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['PYTHONHTTPSVERIFY'] = "0"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

st.set_page_config(page_title="הסוכן החכם שלי", page_icon="🤖")
st.title("🤖 צ'אט סוכן - שלב א' MVP")

@st.cache_resource
def load_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("my_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    embed_model = CohereEmbedding(cohere_api_key=os.environ["COHERE_API_KEY"], model_name="embed-multilingual-v3.0")
    llm = Cohere(api_key=os.environ["COHERE_API_KEY"], model="command-r-plus-08-2024") 
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # הורדנו את הרף ל-0.4 כדי שיהיה מאוד סלחני בשלב הזה
    processor = SimilarityPostprocessor(similarity_cutoff=0.4)
    
    # שימוש ב-Query Engine במקום Chat Engine לצורך יציבות המקורות בשלב א'
    return index.as_query_engine(
        similarity_top_k=3, 
        node_postprocessors=[processor]
    )

if "query_engine" not in st.session_state:
    st.session_state.query_engine = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת היסטוריה
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("שאלו אותי משהו..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("מחפש תשובה ומקורות..."):
            # שימוש ב-query במקום chat כדי לוודא שליפת מקורות נקייה
            response = st.session_state.query_engine.query(prompt)
            
            final_text = str(response)
            if not final_text.strip() or final_text == "None":
                final_text = "לא מצאתי מידע רלוונטי מספיק במסמכים."
            
            st.markdown(final_text)
            
            # הצגת המקורות - עכשיו זה תמיד יבדוק אם יש Metadata
            if hasattr(response, 'source_nodes') and response.source_nodes:
                with st.expander("🔍 מקורות וציוני רלוונטיות (Metadata)"):
                    for node in response.source_nodes:
                        # שליפת שם הקובץ מה-Metadata
                        fname = node.metadata.get('file_name', 'קובץ ללא שם')
                        st.write(f"- **קובץ:** `{fname}` | **ציון דמיון:** {node.score:.4f}")
            else:
                st.info("לא נמצאו מקורות רלוונטיים מעל רף הדיון.")

            st.session_state.messages.append({"role": "assistant", "content": final_text})