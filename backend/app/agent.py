import sqlite3
import pandas as pd
import os
import logging
import re
from langchain_ollama import OllamaLLM
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.tools import Tool
from prophet import Prophet

# --- CONFIGURATION ---
MODEL_NAME = "mistral"
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "databases", "retail_database.db")
_cached_agent = None

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- ERROR HANDLER ---
def specific_error_handler(error: Exception) -> str:
    """
    If the AI gives a good answer but forgets to say 'Final Answer:',
    LangChain thinks it's an error. We catch that here and return the text anyway.
    """
    response = str(error)
    # Check if the error contains the output we want
    if "Could not parse LLM output:" in response:
        try:
            # Extract the text after the error message and clean up quotes/ticks
            clean_response = response.split("Could not parse LLM output:")[1].strip(" `")
            return clean_response
        except:
            pass
            
    return f"I found an answer but couldn't format it perfectly. Here is what I found: {str(error)}"

# --- CUSTOM TOOL: FORECASTING (Adapted from server.py) ---
def predict_sales_tool(periods_str: str) -> str:
    """
    Forecasts future sales revenue using Prophet.
    Input: Number of months to predict (as a string, e.g., "6").
    Output: A string table of forecasted dates and values.
    """
    try:
        # 1. Load Data Fresh from DB (To ensure we have all dates)
        if not os.path.exists(DB_PATH):
            return "Error: Database not found."
            
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM sales", conn)
        conn.close()

        # 2. Preprocess (Match server.py logic but use correct column names)
        # Note: Your DB uses 'date' (lowercase) and 'revenue'. 
        # The server example used 'Date' and 'Order_Demand', so I adapted it here.
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['revenue', 'date'], inplace=True)

        # 3. Parse Periods
        try:
            periods = int(str(periods_str).strip())
        except:
            periods = 6 # Default to 6 months

        # 4. Prepare Data for Prophet (Server.py Logic)
        # Group data by month and sum the revenue
        df_monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum().to_timestamp()
        sales = df_monthly.reset_index()
        
        # Prophet requires columns to be named 'ds' (datestamp) and 'y' (value)
        sales.columns = ['ds', 'y']

        # 5. Fit Prophet
        m = Prophet()
        m.fit(sales)

        # 6. Predict
        future = m.make_future_dataframe(periods=periods, freq='M')
        forecast = m.predict(future)

        # 7. Format Result
        # Get just the future rows
        future_forecast = forecast.tail(periods)[['ds', 'yhat']]
        
        # Format nice string table
        result_str = "\nForecasted Revenue:\n"
        result_str += f"{'Date':<15} | {'Predicted Revenue':<20}\n"
        result_str += "-"*40 + "\n"
        
        for _, row in future_forecast.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            val_str = f"{row['yhat']:,.2f}"
            result_str += f"{date_str:<15} | {val_str:<20}\n"
            
        return result_str

    except Exception as e:
        return f"Error generating forecast: {str(e)}"

# --- AGENT INITIALIZATION ---
def get_or_create_agent():
    global _cached_agent
    if _cached_agent is not None:
        return _cached_agent

    print("Loading Database and initializing Agent...")
    if not os.path.exists(DB_PATH):
        print(f"Error: DB file not found at {DB_PATH}")
        return None

    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Load a random sample so df.head() isn't just one date
        df = pd.read_sql_query("SELECT * FROM sales ORDER BY RANDOM()", conn)
        conn.close()
        
        # Preprocessing (Just like in server.py)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        
        logging.info(f"Data loaded. Rows: {len(df)}")

        llm = OllamaLLM(
            model=MODEL_NAME,
            temperature=0.1, # Low temp for precision
            num_predict=500,
            callbacks=[StreamingStdOutCallbackHandler()] 
        )

        # Create the Tool Object
        forecast_tool = Tool(
            name="predict_sales",
            func=predict_sales_tool,
            description="Use this tool when asked to predict, forecast, or estimate future sales. Input should be the number of months to predict (e.g. '6')."
        )

        # Create Agent with the tool
        _cached_agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=False, 
            allow_dangerous_code=True,
            handle_parsing_errors=specific_error_handler,
            extra_tools=[forecast_tool]
        )

        return _cached_agent

    except Exception as e:
        print(f"Error initializing agent: {e}")
        return None

# --- MAIN LOGIC ---
def run_agent_logic(user_input):
    agent = get_or_create_agent()
    if agent is None:
        return "Database is not available."

    # Pre-calculate latest date for context
    try:
        conn = sqlite3.connect(DB_PATH)
        max_row = pd.read_sql_query("SELECT MAX(date) as last_date FROM sales", conn)
        real_latest_date = max_row['last_date'].iloc[0]
        conn.close()
    except:
        real_latest_date = "Unknown"

    prompt = f"""
    You are a Data Analyst assistant.
    
    CRITICAL CONTEXT:
    - The database has data up to **{real_latest_date}**.
    - Ignore dates in df.head().
    
    RULES:
    1. If asked to PREDICT or FORECAST, you MUST use the `predict_sales` tool. 
    2. If asked about current/past data, use the dataframe directly.
    3. Answer concisely.
    4. You MUST start your final response with "Final Answer:".

    Question: {user_input}
    """

    try:
        result = agent.invoke(prompt)
        
        if isinstance(result, dict) and "output" in result:
            return result["output"].strip()
        if isinstance(result, str):
            return result.strip()
        return "Sorry, I couldn't find the answer."

    except Exception as e:
        return specific_error_handler(e)

if __name__ == "__main__":
    print("="*60)
    print("RETAIL AGENT CLI MODE")
    print("   Type 'q' or 'quit' to exit.")
    print("="*60)
    
    # Ensure DB exists
    if not os.path.exists(DB_PATH):
        print("⚠️  Warning: DB not found.")
        
    get_or_create_agent() 

    while True:
        try:
            user_input = input("\n You: ").strip()
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Exiting...")
                break
            if user_input:
                print("Analyzing...")
                print("\n Agent: ", end="", flush=True) 
                # Note: The StreamingCallback will print the thinking process.
                # run_agent_logic returns the Final Answer string.
                final_answer = run_agent_logic(user_input)
                # If the streaming handler already printed the answer, we don't want to double print.
                # But since the handler prints to stdout, and the function returns a string,
                # we usually just let the function return handle the "Final Answer" display if needed.
                # Given specific_error_handler usage, we just print the result if it wasn't captured.
                if "Final Answer" not in final_answer: 
                     print(f"\n{final_answer}")
                print("\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")