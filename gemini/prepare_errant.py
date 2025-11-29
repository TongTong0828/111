import pandas as pd

print("--- Script: prepare_errant.py ---")

file_name = 'gemini_evaluation_results_full.csv'

COL_INDEX_SOURCE = 2
COL_INDEX_HYPOTHESIS = 6
COL_INDEX_REFERENCE = 3

COL_NAME_SOURCE = 'Ungrammatical Statement'
COL_NAME_HYPOTHESIS = 'Standard English'
COL_NAME_REFERENCE = 'api_correction'

try:
    df = pd.read_csv(
        file_name,
        header=None,
        usecols=[COL_INDEX_SOURCE, COL_INDEX_HYPOTHESIS, COL_INDEX_REFERENCE],
        encoding='utf-8'
    )
    
    df.columns = [COL_NAME_SOURCE, COL_NAME_HYPOTHESIS, COL_NAME_REFERENCE]
    
    print(f"Successfully loaded file: {file_name}")
except FileNotFoundError:
    print(f"Error: File not found '{file_name}'.")
    print("Please ensure 'gemini_evaluation_results_full.csv' is in this directory.")
    exit()

df_clean = df.dropna(subset=[COL_NAME_REFERENCE])
df_clean = df_clean[~df_clean[COL_NAME_REFERENCE].astype(str).str.contains("Error:", na=False)]

print(f"Data loaded and cleaned. Total {len(df_clean)} valid rows for ERRANT evaluation.")

out_source = 'source.txt'
out_hypothesis = 'hypothesis.txt'
out_reference = 'reference.txt'

print("Processing and saving .txt files...")

for col in [COL_NAME_SOURCE, COL_NAME_HYPOTHESIS, COL_NAME_REFERENCE]:
    df_clean[col] = df_clean[col].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')

df_clean[COL_NAME_SOURCE].to_csv(out_source, index=False, header=False)
df_clean[COL_NAME_REFERENCE].to_csv(out_hypothesis, index=False, header=False)
df_clean[COL_NAME_HYPOTHESIS].to_csv(out_reference, index=False, header=False)

print("\nSuccess! Three files required by ERRANT have been generated:")
print(f"1. {out_source} (Original ungrammatical sentences)")
print(f"2. {out_hypothesis} (API corrected sentences)")
print(f"3. {out_reference} (Ground truth references)")
print("\nNext step: Please run 'run_errant.sh' script.")