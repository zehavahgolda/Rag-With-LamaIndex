import os
import ssl
import streamlit as st
import asyncio
from dotenv import load_dotenv
from agent_workflow import SmartAgentWorkflow

# טעינת משתני סביבה (חובה עבור API Keys)
load_dotenv()

# הגדרות רשת ו-SSL עבור נטפרי
os.environ['HTTP_PROXY'] = ""
os.environ['HTTPS_PROXY'] = ""
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# עיצוב דף
st.set_page_config(page_title="Agentic RAG", page_icon="🤖")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; }
    [data-testid="stChatMessage"] { background-color: #161b22 !important; color: white; }
    h1, h2, h3, p, span { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 הסוכן החכם שלי")
st.subheader("מערכת RAG מבוססת Pinecone & LlamaIndex")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("מה תרצו לדעת על הפרויקט?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 סורק מסמכים ב-Pinecone...")
        
        try:
            async def run_agent():
                # יצירת מופע של ה-Workflow המעודכן
                workflow = SmartAgentWorkflow(timeout=60)
                return await workflow.run(query=query)

            # הרצה של ה-Async בתוך Streamlit
            result = asyncio.run(run_agent())
            
            placeholder.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as e:
            st.error(f"שגיאת תקשורת: {e}")