🤖 Agentic RAG System - Project Decision Assistant
An advanced Agentic RAG (Retrieval-Augmented Generation) system designed to query documents and structured data (JSON) using an intelligent workflow. Built specifically for managing project decisions with a sleek Custom Dark Mode interface and support for restricted network environments (e.g., Netfree).

🚀 Project Evolution
Phase A: Basic RAG MVP
Established the core infrastructure:

Document Processing: Ingested raw files and converted them into vector embeddings using Cohere.

Vector Storage: Implemented ChromaDB for efficient semantic data storage.

Retrieval: Created a simple semantic search engine to fetch relevant context based on user queries.

Phase B: Migration to Agentic Workflow
Upgraded the system from a simple search to a sophisticated workflow-based agent:

Intelligent Routing: Added a Router step where the agent analyzes the query to decide between Semantic Search (ChromaDB) or Direct Data Extraction (JSON).

Structured Data Extraction: Enabled high-precision reading of structured_data.json for factual project-specific answers.

Streamlit Integration: Overcame asyncio loop challenges to run complex workflows within a web interface.

🛠 Tech Stack
LlamaIndex: Framework for managing the Agentic Workflow and steps.

Cohere (Command R+): High-performance LLM optimized for agentic tasks.

Streamlit: Interactive UI with a custom-engineered Dark Theme.

ChromaDB: Vector database for context-aware retrieval.

🔒 Technical Challenges & Solutions
SSL Bypass: Implemented global SSL overrides to ensure connectivity in environments with custom security certificates.

Python 3.11 Compatibility: Optimized environment setup to ensure stable asyncio performance.

Network Connectivity: Configured custom httpx clients and environment variables to bypass proxy-related issues (Netfree/Corporate firewalls).

🎨 UI/UX Features
Total Dark Mode: Custom CSS for a deep black background (#0e1117).

Chat Interface: Clean, distinguishable chat bubbles with optimized typography.

RTL Support: Full support for Hebrew text and Right-to-Left alignment.

How to add this to your project:
Create/Open the README.md file in your root folder.

Paste this English version inside.

Commit message: docs: translate README to English and update technical architecture
⚙️ Installation & Setup
1. Prerequisites
Python 3.11 (Recommended for stability with asyncio and SSL bypass).

A Cohere API Key.
2. Clone and Install Dependencies
# Clone the repository
git clone <your-repository-url>
cd rag_app

# Install required packages
COHERE_API_KEY=your_api_key_here
4. Running the Application
python -m streamlit run app.py
📂 Project Structure
app.py: The Streamlit frontend with custom Dark Mode CSS.

agent_workflow.py: The core logic, including the Router, Retrieval, and Extraction steps.

structured_data.json: The source for factual project decisions.

chroma_db/: Vector storage for semantic search.
python -m pip install streamlit llama-index-core llama-index-llms-cohere llama-index-embeddings-cohere nest_asyncio python-dotenv chromadb httpx
3. Environment Variables
Create a .env file in the root directory and add your API key:
