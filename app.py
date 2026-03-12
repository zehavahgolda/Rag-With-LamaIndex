import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from agent_workflow import SmartAgentWorkflow

# טעינת הגדרות
load_dotenv()

st.set_page_config(page_title="AI Agent System", page_icon="🤖", layout="centered")

# עיצוב
st.markdown("""
    <style>
    .stApp { background-color: #1e1e2f; color: white; }
    .stButton>button { background-color: #6d28d9; color: white; border-radius: 10px; width: 100%; }
    .stTextInput>div>div>input { background-color: #2d2d44; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 הסוכן החכם שלי (Agentic RAG)")
st.subheader("מערכת שאלות ותשובות מבוססת Workflow")

query = st.text_input("מה תרצו לדעת?", placeholder="למשל: מה הצבע העיקרי של המערכת?")

if st.button("שאל את הסוכן"):
    if query:
        with st.spinner("🕵️ הסוכן חוקר את המידע ומגבש תשובה..."):
            try:
                # התיקון לשגיאת ה-Event Loop:
                async def run_agent():
                    agent = SmartAgentWorkflow(timeout=60)
                    return await agent.run(query=query)

                # הרצה בטוחה של הקוד האסינכרוני
                try:
                    # מנסים לקבל loop קיים
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # אם אין loop, יוצרים חדש
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(run_agent())
                
                st.success("✅ תשובת הסוכן:")
                st.write(result)
                
                with st.expander("ראה לוגים של ה-Workflow"):
                    st.info("הסוכן ביצע שליפה (Retrieve) ולאחר מכן וולידציה (Validation) מבוססת תוכן.")
            
            except Exception as e:
                st.error(f"אירעה שגיאה: {e}")
    else:
        st.warning("אנא הזינו שאלה קודם.")

st.markdown("---")
st.caption("פיתוח שלב ב' - Agentic RAG Workflow | 2026")