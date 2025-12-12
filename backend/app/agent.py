import sqlite3
import pandas as pd
import os
import logging
import re
from langchain_ollama import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- CONFIGURATION ---
MODEL_NAME = "mistral"
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "databases", "retail_database.db")


_cached_agent = None

def specific_error_handler(error: Exception) -> str:
    """
    If the LLM gives the answer but fails the strict format check,
    we catch the error, extract the answer, and force it as the Final Answer.
    """
    response = str(error)
    
    # Check if the error is just a parsing issue
    if "Could not parse LLM output:" in response:
        try:
            # Regex to pull out the text that the LLM actually said
            # It usually looks like: Could not parse LLM output: `The answer is X`
            match = re.search(r"Could not parse LLM output:\s*`?([^`]+)`?", response)
            if match:
                clean_response = match.group(1).strip()
                # TRICK: We prepend "Final Answer:" so the Agent knows it's done.
                return f"Final Answer: {clean_response}"
        except Exception:
            pass
            
    return f"Final Answer: I encountered an error processing that request: {str(error)}"

def get_or_create_agent():
    global _cached_agent
    
    if _cached_agent is not None:
        return _cached_agent

    print("🔄 Loading Database and initializing Agent...")
    print(f"   Looking for DB at: {DB_PATH}")
    
    # 1. Check/Create Database Path
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: DB file not found at {DB_PATH}")
        return None

    # 2. Load Data into Pandas
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        df = pd.read_sql_query("SELECT * FROM sales", conn)
        conn.close()
            
        logging.info(f"Data loaded. Rows: {len(df)}")

        
        llm = OllamaLLM(
            model=MODEL_NAME,
            temperature=0.1,
            callbacks=[StreamingStdOutCallbackHandler()] 
        )
        # 4. Create Agent
        _cached_agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=False, 
            allow_dangerous_code=True,
            handle_parsing_errors=specific_error_handler
        )
        
        return _cached_agent

    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return None

def run_agent_logic(user_input):
    try:
        agent = get_or_create_agent()
        
        if agent is None:
            return (f"Error: Could not load the database at {DB_PATH}. "
                    "Please ensure 'retail_database.db' exists and has a 'sales' table.")

        contextualized_input = (
            "Answer this concisely using the DataFrame 'df': " + user_input + 
            " Start with 'Final Answer:' immediately."
        )

        response = agent.invoke(contextualized_input)
        
        return response['output']

    except Exception as e:
        print(f"Agent Execution Error: {e}")
        return f"I ran into an error analyzing the data: {str(e)}"

# --- TERMINAL TESTING BLOCK ---
if __name__ == "__main__":
 
    print("="*60)
    print("🛒 RETAIL AGENT CLI MODE")
    print("   Type 'q' or 'quit' to exit.")
    print("="*60)

    # Check if DB exists before asking
    if not os.path.exists(DB_PATH):
        print(f"⚠️  Warning: Database file not found at {DB_PATH}")
        print("   The agent might fail, but we will try initializing now.")

    get_or_create_agent()

    while True:
        try:
          
            user_input = input("\n💬 You: ").strip()

            if user_input in ["hello", "Hi", "Hallo", "Hai"]:
                print("Hello, I'm your assistant to help you know little bit more about your sales. Ask me anything related your sales.👋 ")
                continue
            
            if user_input.lower() in ["q", "quit", "exit", "bye"]:
                print("👋 Exiting...")
                break
                
            if user_input:
                print("🤔 Analyzing...")
                print("\n🤖 Agent: ", end="", flush=True) 
                result = run_agent_logic(user_input)
                print("\n")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")