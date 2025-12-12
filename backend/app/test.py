import sqlite3
import pandas as pd
import os
import logging
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- CONFIGURATION ---
MODEL_NAME = "mistral"
# FIX: Use absolute path relative to this script file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Points to: backend/app/databases/retail_database.db
DB_PATH = os.path.join(current_dir, "databases", "retail_database.db")

# Global cache to prevent reloading the agent on every request
_cached_agent = None

def specific_error_handler(error: Exception) -> str:
    """
    This function catches 'OutputParserException'.
    If the AI gives a good answer but forgets to say 'Final Answer:',
    LangChain thinks it's an error. We catch that here and return the text anyway.
    """
    response = str(error)
    # Check if the error contains the output we want
    if "Could not parse LLM output:" in response:
        # Extract the text after the error message and clean up quotes/ticks
        try:
            clean_response = response.split("Could not parse LLM output:")[1].strip(" `")
            return clean_response
        except IndexError:
            pass
    
    # If it's a real code error, return the error message
    return f"I encountered an error processing that request: {str(error)}"

def get_or_create_agent():
    """
    Singleton pattern: Loads the DB and creates the Agent only once.
    Returns the cached agent.
    """
    global _cached_agent
    
    if _cached_agent is not None:
        return _cached_agent

    print("🔄 Loading Database and initializing Agent...")
    print(f"   Looking for DB at: {DB_PATH}")
    
    # 1. Check/Create Database Path
    if not os.path.exists(DB_PATH):
        # Fallback: create a dummy DB if the file is missing so the app doesn't crash
        print(f"❌ Error: DB file not found at {DB_PATH}")
        return None

    # 2. Load Data into Pandas
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        df = pd.read_sql_query("SELECT * FROM sales", conn)
        conn.close()
            
        logging.info(f"Data loaded. Rows: {len(df)}")

        # 3. Setup LLM
        llm = OllamaLLM(model=MODEL_NAME)

        # 4. Create Agent
        # allow_dangerous_code=True is required because the agent writes Python to analyze data
        _cached_agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=specific_error_handler
        )
        
        return _cached_agent

    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return None

def run_agent_logic(user_input, conversation_history=[]):
    """
    The main entry point called by Flask (app/main.py).
    """
    try:
        agent = get_or_create_agent()
        
        if agent is None:
            return (f"Error: Could not load the database at {DB_PATH}. "
                    "Please ensure 'retail_database.db' exists and has a 'sales' table.")

        # Optional: You can prepend a system instruction to the user input
        # But for DataFrame agents, it's often better to just pass the query directly.
        # FIX: Corrected string concatenation syntax below
        contextualized_input = (
            "You are a data analysis assistant. "
            "Use the DataFrame `df` to answer: " + user_input + " "
            "Answer concisely. "
            "Please start your final response with 'Final Answer:'."
        )

        # Invoke the agent
        response = agent.invoke(contextualized_input)
        
        # Return the 'output' string from LangChain
        return response['output']

    except Exception as e:
        print(f"Agent Execution Error: {e}")
        return f"I ran into an error analyzing the data: {str(e)}"

# --- TERMINAL TESTING BLOCK ---
if __name__ == "__main__":
    # This block allows you to chat interactively in the terminal
    print("="*60)
    print("🛒 RETAIL AGENT CLI MODE")
    print("   Type 'q' or 'quit' to exit.")
    print("="*60)

    # Check if DB exists before asking
    if not os.path.exists(DB_PATH):
        print(f"⚠️  Warning: Database file not found at {DB_PATH}")
        print("   The agent might fail, but we will try initializing now.")

    # Pre-load the agent so we don't wait on the first question
    get_or_create_agent()

    while True:
        try:
            # This 'input' function waits for you to type in the terminal
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ["q", "quit", "exit", "bye"]:
                print("👋 Exiting...")
                break
                
            if user_input:
                print("🤔 Analyzing...")
                result = run_agent_logic(user_input)
                print(f"\n🤖 Agent: {result}")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")