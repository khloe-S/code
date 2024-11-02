import os
import pandas as pd

# Define the directory where your CSV files are located
directory = r'F:\\DESKTOP(SUB-FOLDER)\\python files\\Automation analysis of multiple indicators'

# List of specific files you're working with
file_list = ['C001_FakeHypotension.csv', 'C001_FakeSepsis.csv', 'ZZZ_Sepsis_Data_From_R.csv']

# List to store all the dataframes
all_data = []

# Loop through each specified file
for filename in file_list:
    file_path = os.path.join(directory, filename)
    
    # Read the CSV file
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {filename}")
        
        # Append to the list of dataframes (filename, dataframe)
        all_data.append((filename, df))
        
        # Display file structure information
        print(f"\n--- File: {filename} ---")
        print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
        print("\nData Types and Null Values:\n")
        print(df.info())  # Displays data types and non-null counts
        
        print("\nSample of the Data:\n")
        print(df.head())  # Shows the first 5 rows
        
        print("\nSummary Statistics (for numeric columns):\n")
        print(df.describe())  # Shows summary statistics for numeric columns
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")
