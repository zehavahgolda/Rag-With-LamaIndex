import asyncio
from llama_index.utils.workflow import draw_all_possible_flows
from agent_workflow import SmartAgentWorkflow

async def main():
    print("🎨 מייצר את תרשים ה-Workflow...")
    try:
        # יצירת מופע (Instance) של ה-Workflow
        workflow_instance = SmartAgentWorkflow()
        
        # העברת המופע לפונקציית הציור
        draw_all_possible_flows(workflow_instance, filename="workflow_diagram.html")
        
        print("✅ בוצע בהצלחה!")
        print("📍 חפשי את הקובץ 'workflow_diagram.html' בתיקיית rag_app")
        
    except Exception as e:
        print(f"❌ שגיאה: {e}")
        print("\nנסיון שני - העברת המחלקה ישירות:")
        try:
            draw_all_possible_flows(SmartAgentWorkflow, filename="workflow_diagram.html")
            print("✅ בוצע בנסיון השני!")
        except Exception as e2:
            print(f"❌ גם הנסיון השני נכשל: {e2}")

if __name__ == "__main__":
    asyncio.run(main())