# KCG Ssles Agent
A streaming AI assistant that analyzes retail sales data using a local LLM (Mistral). The application uses a Python Flask backend with LangChain agents and an Angular frontend.

# Project Stucture
```
SALES-AGENT/
├── backend/
│   └── app/
│       ├── databases/
│       │   ├── database.py             # Script to initialize/convert CSV to SQLite
│       │   ├── rawdata.py              # Utility functions for raw data
│       │   ├── retail_database.db      # The actual SQLite database used by the Agent
│       │   └── retail-dataset-new.csv  # Raw source data (input file)
│       │
│       ├── agent.py                    # "Brain" of the AI; connects LangChain to Ollama
│       ├── server.py                   # Flask Backend; handles API requests & streaming
│       └── test.py                     # Terminal-based script for quick testing without UI
│
└── frontend/
    └── agent-interface/
        ├── package.json                # Frontend dependencies list (Angular, RxJS, etc.)
        └── src/
            ├── main.ts                 # Entry point for the Angular application
            ├── styles.css              # Global styles for the entire website
            └── app/
                ├── api.service.ts      # Connects to Backend; handles SSE (Streaming)
                ├── language.service.ts # Utility for text formatting/language switching
                ├── app.routes.ts       # Defines URL navigation logic
                ├── app.config.ts       # Main application configuration
                │
                ├── app.component.ts    # Main "Shell" logic of the website
                ├── app.component.html  # Main "Shell" UI structure
                ├── app.component.css   # Main "Shell" styling
                │
                ├── chat.component.ts   # Logic for the chat window & user input
                ├── chat.component.html # Visual template for chat bubbles
                └── chat.component.css  # Styling specific to the chat interface
```
# Prerequisites
 - Python 3.10+ installed.
 - Ollama installed and running.
 - Node.js & Angular CLI

# Setup & Installation
1. Backend (Python)
    It is recommended to use a virtual environment.

    ```bash
        cd SALES-AGENT/backend/app
    
        # Create virtual environment
        python -m venv venv

        # Activate it (Windows)
        venv\Scripts\activate

        # Activate it (Mac/Linux)
        source venv/bin/activate

        # Install dependencies
        pip install -r requirements.txt

        # Initialize Database
        python databases/database.py

        # Run Server
        python server.py
    ```
2. Frontend Setup
    ```bash
       cd SALES-AGENT/frontend/agent-interface
       npm install
       ng serve
      ```

# How is it work
1. User Input: The user asks a question via the Angular chat interface.
2. Greeting Check: If the input is a greeting (e.g., "Hi"), the server responds instantly without     invoking the heavy AI model.
3. Agent Reasoning: For data questions, the Flask server passes the query to the LangChain agent in agent.py.
4. Code Execution: The agent generates Python code to query the SQLite database (e.g., df.groupby('date').sum()).
5. Streaming: The "Thought" process is hidden, and the final answer is streamed token-by-token to the frontend using Server-Sent Events (SSE).

# Example Questions
###  Impact of Marketing & Events (The "Why")
"How much higher is the average revenue on Payday compared to normal days?"

"What is the total revenue generated during Holiday periods versus non-holidays?"

"Do transactions with a Promo active have a higher average unit price?"

"Which product category sells best during holidays?"

###  Channels & Locations (The "Where")
"Which Sales Channel (Tokopedia, Lazada, Website) brings in the highest total revenue?"

"Compare the total units sold in Jakarta (Grand Indonesia) versus Surabaya (Galaxy Mall)."

"What is the most popular product sold on Tokopedia?"

"Which store location has the highest average transaction value?"

###  Product Performance (The "What")
"What is the top-selling Product Name by total revenue?"

"Which Product Category has the highest number of units sold?"

"List the top 3 items sold in the 'Activewear' category."

###  Trends & Patterns (The "When")
"What is the total revenue for January 1, 2024?"

"Show the daily sales trend for 'Mens Clothing'."

"Which date had the highest total revenue?"

###  Advanced / Compound Questions
"Does the Payday Effect have a bigger impact on 'Activewear' or 'Footwear'?"

"What is the total revenue from Offline stores vs Online channels?" (If you treat specific locations as offline and websites as online).

"Find the day with the lowest sales that was NOT a holiday."

# Troubleshooting
- "Agent not available": Ensure retail_database.db exists. Run python database.py again.
- "Connection refused": Make sure Ollama is running in the background (ollama serve).
- Streaming is slow: Streaming speed depends on your local hardware (GPU/CPU) running the Mistral model.
