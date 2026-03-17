🤖 Agentic RAG: Project Decision Assistant
Bridging the gap between messy documentation and structured insights.
This project implements an advanced Event-Driven RAG (Retrieval-Augmented Generation) system. It doesn't just search; it thinks. Using an intelligent Agentic Workflow, the system routes queries between semantic vector search and structured data extraction to provide precise answers about project decisions, rules, and logs.

📊 System Architecture (Workflow)
The system is built on a non-linear, event-driven architecture using LlamaIndex Workflows.

The Router: Analyzes the user's intent.

Branch A (Semantic Search): Queries a high-performance Pinecone Vector DB for general context and "vibe" questions.

Branch B (Structured Extraction): Directly queries structured_data.json for hard facts (rules, dates, tech decisions).

Synthesizer: Merges the gathered context and formulates a human-like response using Cohere's Command R+.

🚀 Key Features
Intelligent Routing: Automatically chooses the best data source for the question.

Netfree-Ready: Built-in SSL and Proxy bypass mechanisms for restricted network environments.

Hybrid Knowledge: Combines unstructured Markdown files (from Cursor/Claude) with structured JSON data.

Sleek UI: A custom Streamlit "Deep Dark" interface optimized for developers.

🛠 Tech Stack
Framework: LlamaIndex (Workflows & Agentic orchestration).

LLM: Cohere Command R+ (Optimized for RAG and tool use).

Vector Database: Pinecone (Cloud-native vector storage).

Embeddings: Cohere multilingual-v3.0.

UI: Streamlit with custom CSS.

📂 Project Structure
Plaintext
├── rag_app/
│   ├── agent_workflow.py     # The "Brain" - Event-Driven logic
│   ├── app.py               # Streamlit UI & Async runner
│   ├── ingestion.py         # Data processing & Pinecone upload
│   ├── generate_viz.py      # Generates the Workflow HTML diagram
│   ├── structured_data.json  # Structured facts and rules
│   └── .env                 # API Keys and configuration
└── data_source/             # Raw Markdown files from Coding Agents
⚙️ Setup & Installation
1. Environment Setup
Create a .env file in the rag_app folder:

קטע קוד
COHERE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
2. Install Dependencies
Bash
pip install llama-index-core llama-index-llms-cohere llama-index-embeddings-cohere llama-index-vector-stores-pinecone pinecone-client streamlit python-dotenv httpx pyvis
3. Ingest Data
Bash
python ingestion.py
4. Run the Agent
Bash
streamlit run app.py
🔍 Example Queries
Semantic: "How does the system handle user authentication?"

Structured: "List all technical decisions made in the last month."

Hybrid: "Are there any specific CSS rules for the Dark Mode?"

🎨 Visualizing the Workflow
To regenerate the interactive workflow diagram, run:
python generate_viz.py
