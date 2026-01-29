import sqlite3  # SQLite database connector
import pandas as pd  # Data manipulation and analysis library
import os  # Operating system utilities for file path handling

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the CSV file containing retail sales data
csv_file_path = os.path.join(current_dir, "retail-dataset-new.csv")
# Define the path where the SQLite database will be created/stored
db_file_path = os.path.join(current_dir, "retail_database.db")

# Print the CSV file path for debugging purposes
print(f"Looking for CSV at: {csv_file_path}")

try:
    # Validate that the CSV file exists before attempting to load it
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found at {csv_file_path}. Please ensure the CSV is in the 'backend/app/databases' folder.")

    # Open and read the first line of the CSV file for inspection
    with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_line = f.readline().strip()
        # Print raw content for debugging data format issues
        print(f"DEBUG - Raw first line of file: {raw_line}")

    # Load the CSV file into a pandas DataFrame
    # sep=";" specifies semicolon as the delimiter
    # quoting=3 disables quote parsing to handle quote characters as regular text
    df = pd.read_csv(csv_file_path, sep=";", engine='python', quoting=3)

    # Clean up quote characters from all text columns in the dataset
    print("Cleaning quotes from data...")
    for col in df.select_dtypes(include=['object']).columns:
        # Remove double and single quotes, then strip whitespace
        df[col] = df[col].str.replace('"', '').str.replace("'", "").str.strip()

    # Display a preview of the processed data to verify correct column parsing
    print("--- PREVIEW OF DATA (Check if columns are split) ---")
    print(df.head())
    print("----------------------------------------------------")

    # Clean up column names by removing quotes and extra whitespace
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_file_path)
    print(f"Database created/connected at: {db_file_path}")

    # Write the DataFrame to the 'sales' table in the database
    # if_exists="replace" overwrites the table if it already exists
    # index=False prevents writing the DataFrame index as a column
    df.to_sql("sales", conn, if_exists="replace", index=False)
    print("Data successfully loaded into the 'sales' table.")

    # Close the database connection to release resources
    conn.close()

# Handle any errors that occur during the process
except Exception as e:
    print(f"An error occurred: {e}")