import os
import ssl
import streamlit as st
import asyncio
from agent_workflow import SmartAgentWorkflow

# ניסיון אחרון לעקוף הגדרות רשת כפויות
os.environ['HTTP_PROXY'] = ""
os.environ['HTTPS_PROXY'] = ""
os.environ['NO_PROXY'] = "cohere.com,api.cohere.com"

# נטרול SSL עבור נטפרי
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# 1. הגדרת עיצוב כהה (Dark Mode)
st.set_page_config(page_title="Agentic RAG", page_icon="🤖")

# הזרקת CSS שתהפוך את הכל לשחור עמוק (Darker than Dark)
st.markdown("""
    <style>
    /* הופך את כל רקע האפליקציה לשחור */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
    }
    /* הופך את הבועות של הצ'אט לכהות מאוד */
    [data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d;
        color: white;
    }
    /* צבע לבן לכל הטקסט */
    h1, h2, h3, p, span, li {
        color: white !important;
    }
    /* שורת קלט שחורה */
    .stChatInputContainer {
        background-color: #0e1117 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 הסוכן החכם שלי (Agentic RAG)")
st.subheader("Workflow מערכת שאלות ותשובות מבוססת")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("מה תרצו לדעת?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔍 הסוכן חושב...")
        
        try:
            # 2. הדרך היחידה להריץ ב-Streamlit בלי שגיאות Loop
            async def run_agent():
                workflow = SmartAgentWorkflow(timeout=60)
                return await workflow.run(query=query)

            # הרצה שמחכה לתוצאה (Await) בצורה נכונה
            result = asyncio.run(run_agent())
            
            placeholder.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as e:
            # אם מופיעה שגיאת SSL (הפס האדום), זה בגלל נטפרי
            st.error(f"אירעה שגיאה: {e}")