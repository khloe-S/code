import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ensure this section is included to load the data before preprocessing
directory = r'F:\\DESKTOP(SUB-FOLDER)\\python files\\Automation analysis of multiple indicators'
file_list = ['C001_FakeHypotension.csv', 'C001_FakeSepsis.csv', 'ZZZ_Sepsis_Data_From_R.csv']

# List to store all the dataframes
all_data = []

# Load the data from the CSV files into all_data
for filename in file_list:
    file_path = os.path.join(directory, filename)
    try:
        df = pd.read_csv(file_path)
        all_data.append((filename, df))
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Continue with preprocessing
processed_data = []

def preprocess_data(df):
    # Step 1: Remove unnecessary columns (like 'Unnamed: 0')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Step 2: Data cleaning (removing duplicates)
    df.drop_duplicates(inplace=True)

    # Step 3: Normalization
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    return df

# Loop through each dataframe for preprocessing
for filename, df in all_data:
    processed_df = preprocess_data(df)
    processed_data.append((filename, processed_df))

print("Preprocessing completed successfully.")
