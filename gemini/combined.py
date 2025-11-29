import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

INPUT_FILE = 'gemini_evaluation_results_full.csv' 
OUTPUT_IMG = 'confusion_matrix_final_2018.png'

LABEL_MAPPING = {
    "Subject-Verb Agreement": "Verb/Tense Issues",
    "Verb Tense Errors": "Verb/Tense Issues",
    "Incorrect Auxiliaries": "Verb/Tense Issues",
    "Passive Voice Overuse": "Verb/Tense Issues",

    "Gerund and Participle Errors": "Verb Form Errors",
    "Infinitive Errors": "Verb Form Errors",

    "Punctuation Errors": "Mechanics/Punctuation",
    "Contractions Errors": "Mechanics/Punctuation",
    "Ellipsis Errors": "Mechanics/Punctuation",
    "Capitalization Errors": "Mechanics/Punctuation",
    "Abbreviation Errors": "Mechanics/Punctuation",

    "Sentence Structure Errors": "Sentence Structure",
    "Sentence Fragments": "Sentence Structure",
    "Run-on Sentences": "Sentence Structure",
    "Comma Splices": "Sentence Structure",
    "Conjunction Misuse": "Sentence Structure",
    "Ambiguity": "Sentence Structure",

    "Parallelism Errors": "Parallelism Issues",
    "Lack of Parallelism in Lists or Series": "Parallelism Issues",

    "Inappropriate Register": "Style/Register",
    "Slang, Jargon, and Colloquialisms": "Style/Register",
    "Clichés": "Style/Register",
    
    "Redundancy/Repetition": "Redundancy/Word Choice",
    "Tautology": "Redundancy/Word Choice",
    "Word Choice/Usage": "Redundancy/Word Choice",
    "Mixed Metaphors/Idioms": "Redundancy/Word Choice",
    
    "Faulty Comparisons": "Comparison Errors",
    "Agreement in Comparative and Superlative Forms": "Comparison Errors",
    
    "Pronoun Errors": "Pronoun/Relative Clause",
    "Relative Clause Errors": "Pronoun/Relative Clause",
}

def analyze_full_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found {INPUT_FILE}")
        return

    column_names = [
        'id', 'Error Type', 'Ungrammatical Statement', 'Standard English', 
        'original_index', 'api_label', 'api_correction'
    ]
    
    try:
        df = pd.read_csv(INPUT_FILE, header=None, names=column_names)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    df = df[df['Error Type'] != 'Error Type']
    df = df.dropna(subset=['Error Type', 'api_label'])
    
    print(f"Successfully loaded data, valid rows: {len(df)}")
    
    df['mapped_true'] = df['Error Type'].map(LABEL_MAPPING).fillna(df['Error Type'])
    df['mapped_pred'] = df['api_label'].map(LABEL_MAPPING).fillna(df['api_label'])

    print("\nGenerating analysis report...")

    print("\n--- Classification Report (Final Logic) ---")
    print(classification_report(df['mapped_true'], df['mapped_pred'], zero_division=0))

    top_labels = df['mapped_true'].value_counts().head(15).index.tolist()
    
    df_plot = df[df['mapped_true'].isin(top_labels) & df['mapped_pred'].isin(top_labels)]
    
    cm = confusion_matrix(df_plot['mapped_true'], df_plot['mapped_pred'], labels=top_labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=top_labels, yticklabels=top_labels)
    plt.xlabel('Predicted Label (Gemini)')
    plt.ylabel('True Label (Ground Truth)')
    plt.title('Confusion Matrix (Full Dataset - Mapped)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMG)
    print(f"\nSaved confusion matrix image: {OUTPUT_IMG}")

if __name__ == "__main__":
    analyze_full_data()