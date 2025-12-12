import sqlite3
import pandas as pd
import os

# Step 0: Get the absolute path of the directory where this script is located
# This ensures the code works regardless of where you run the terminal command from
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for input (CSV) and output (DB)
csv_file_path = os.path.join(current_dir, "retail-dataset-new.csv")
db_file_path = os.path.join(current_dir, "retail_database.db")

print(f"Looking for CSV at: {csv_file_path}")

try:
    # Step 1: Check file existence
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found at {csv_file_path}. Please ensure the CSV is in the 'backend/app/databases' folder.")

    # Read the first line of raw text to see if there are hidden quotes
    with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_line = f.readline().strip()
        print(f"DEBUG - Raw first line of file: {raw_line}")

    # Load csv data. 
    # engine='python' is more robust for auto-detecting format.
    # quoting=3 (csv.QUOTE_NONE) forces pandas to ignore quotes and split strictly on ';'.
    df = pd.read_csv(csv_file_path, sep=";", engine='python', quoting=3)

    # Step 1.5: Clean up quotes from the data values
    # Since quoting=3 treats quotes as text, we need to manually remove them from string columns
    print("Cleaning quotes from data...")
    for col in df.select_dtypes(include=['object']).columns:
        # Remove " and ' characters, then trim whitespace
        df[col] = df[col].str.replace('"', '').str.replace("'", "").str.strip()

    # DEBUG: Print the first few rows to verify columns are split correctly
    print("--- PREVIEW OF DATA (Check if columns are split) ---")
    print(df.head())
    print("----------------------------------------------------")

    # Step 2: data clean up
    # Remove quotes from column names if they stuck around
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Step 3: connect to SQLite using the absolute path
    conn = sqlite3.connect(db_file_path)
    print(f"Database created/connected at: {db_file_path}")

    # Step 4: load data to SQLite
    df.to_sql("sales", conn, if_exists="replace", index=False)
    print("Data successfully loaded into the 'sales' table.")

    # Step 5: close the connection
    conn.close()

except Exception as e:
    print(f"An error occurred: {e}")