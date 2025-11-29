import google.generativeai as genai
import os
import time
import json
import re 
from tqdm import tqdm
import pandas as pd
import sys

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: Environment variable GOOGLE_API_KEY not found. Please check settings or hardcode it.")

if api_key:
    genai.configure(api_key=api_key)

INPUT_FILE = 'Grammar Correction.csv'
OUTPUT_FILE = 'gemini_evaluation_results_v2.csv'
COL_INCORRECT = 'Ungrammatical Statement'
COL_ERROR_TYPE = 'Error Type'
COL_CORRECT = 'Standard English'

RUN_MODE = 0

OFFICIAL_ERROR_TYPES = [
    "Capitalization Errors",
    "Preposition Usage",
    "Infinitive Errors",
    "Faulty Comparisons",
    "Mixed Conditionals",
    "Contractions Errors",
    "Incorrect Auxiliaries",
    "Clichés",
    "Ambiguity",
    "Relative Clause Errors",
    "Conjunction Misuse",
    "Verb Tense Errors",
    "Ellipsis Errors",
    "Quantifier Errors",
    "Spelling Mistakes",
    "Passive Voice Overuse",
    "Agreement in Comparative and Superlative Forms",
    "Lack of Parallelism in Lists or Series",
    "Abbreviation Errors",
    "Modifiers Misplacement",
    "Punctuation Errors",
    "Tautology",
    "Parallelism Errors",
    "Run-on Sentences",
    "Negation Errors",
    "Word Choice/Usage",
    "Pronoun Errors",
    "Inappropriate Register",
    "Sentence Fragments",
    "Article Usage",
    "Subject-Verb Agreement",
    "Mixed Metaphors/Idioms",
    "Redundancy/Repetition",
    "Sentence Structure Errors",
    "Slang, Jargon, and Colloquialisms",
    "Gerund and Participle Errors"
]

def get_gemini_evaluation(sentence, model):
    categories_str = "\n".join([f"- {cat}" for cat in OFFICIAL_ERROR_TYPES])
    
    system_prompt = f"""
    You are an expert English grammar evaluator.
    
    Your task is to analyze the user's sentence for grammatical errors.
    
    Step 1: Identify the SINGLE most appropriate error type from the following list of 36 labels. 
    You must copy the label text EXACTLY as shown below:
    {categories_str}
    
    Step 2: Provide the grammatically correct version of the sentence. Maintain the original meaning.

    Input Sentence: "{sentence}"

    You MUST respond ONLY with a valid JSON object in the following format. Do NOT include markdown formatting like ```json.
    {{
        "label": "<The exact error label from the list above>",
        "correction": "<The corrected sentence>"
    }}

    If the sentence is completely correct (which is rare), set "label" to "None" and "correction" to the original sentence.
    """
    
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        temperature=0.1
    )
    
    max_retries = 8
    base_wait_time = 8 
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                system_prompt,
                generation_config=generation_config
            )
            
            raw_text = response.text.strip()
            cleaned_text = re.sub(r"^```json\s*", "", raw_text)
            cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            result_json = json.loads(cleaned_text)
            
            if "label" in result_json and "correction" in result_json:
                return result_json['label'], result_json['correction']
            else:
                print(f"Warning: JSON missing fields (Attempt {attempt+1}/{max_retries})")
                continue
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait_time = base_wait_time * (2 ** attempt)
                print(f"\n!!! Rate Limit Triggered (429). Pausing for {wait_time} seconds before retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            elif "400" in error_str:
                print(f"\n!!! 400 Error (Bad Request): {e}")
                return "Error: Bad Request", "Error: Bad Request"
            else:
                print(f"\n!!! Unknown API Error: {e}. Pausing for 5 seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(5)
    
    return "Error: Failed after max retries", "Error: Failed after max retries"

def main_batch_process():
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File '{INPUT_FILE}' not found.")
        return
    
    df['original_index'] = df.index

    initial_rows = len(df)
    content_cols = [COL_INCORRECT, COL_ERROR_TYPE, COL_CORRECT]
    df.drop_duplicates(subset=content_cols, keep='first', inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    if RUN_MODE == 1:
        df_to_process_master = df.head(100).copy()
        print("Test Mode: Processing first 100 rows.")
    else:
        df_to_process_master = df.copy()
        print(f"Full Mode: Processing all {len(df)} rows.")

    processed_indices = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            if 'original_index' in df_done.columns:
                processed_indices = set(df_done['original_index'].unique())
            print(f"Found processed file, skipping {len(processed_indices)} rows.")
        except:
            print("Failed to read old file, starting over.")

    df_queue = df_to_process_master[~df_to_process_master['original_index'].isin(processed_indices)]
    
    if len(df_queue) == 0:
        print("All data processed!")
        return

    print("Initializing Gemini Model...")
    model = genai.GenerativeModel('models/gemini-2.5-flash')

    print("Starting evaluation...")
    write_header = not os.path.exists(OUTPUT_FILE)

    for index, row in tqdm(df_queue.iterrows(), total=len(df_queue), desc="Processing"):
        
        sentence = row[COL_INCORRECT]
        
        api_label, api_correction = get_gemini_evaluation(sentence, model)
        
        result_row = row.to_dict()
        result_row['api_label'] = api_label
        result_row['api_correction'] = api_correction
        
        out_df = pd.DataFrame([result_row])
        out_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
        
        write_header = False
        
        time.sleep(1.0) 

    print(f"\nDone! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main_batch_process()