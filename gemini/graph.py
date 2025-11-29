import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import difflib

file_name = 'gemini_evaluation_results_full.csv'

try:
    df = pd.read_csv(file_name, header=None)
    df.columns = ['id_1', 'target_label', 'incorrect_sentence', 'gt_correction', 
                  'id_2', 'api_label', 'api_correction']
except FileNotFoundError:
    print(f"Error: File not found '{file_name}'")
    exit()

mapping = {
    'Lack of Parallelism in Lists or Series': 'Parallelism Issues',
    'Parallelism Errors': 'Parallelism Issues',
    'Verb Tense Errors': 'Verb/Tense Issues',
    'Subject-Verb Agreement': 'Verb/Tense Issues',
    'Incorrect Auxiliaries': 'Verb/Tense Issues',
    'Verb Form Errors': 'Verb/Tense Issues',
    'Gerund and Participle Errors': 'Verb Form Errors', 
    'Infinitive Errors': 'Verb Form Errors',
    'Redundancy': 'Redundancy/Word Choice',
    'Word Choice Errors': 'Redundancy/Word Choice',
    'Wrong Word Usage': 'Redundancy/Word Choice',
    'Tautology': 'Redundancy/Word Choice',
    'Article Errors': 'Article Usage',
    'Preposition Errors': 'Preposition Usage',
    'Run-on Sentences': 'Sentence Structure',
    'Sentence Fragments': 'Sentence Structure',
    'Dangling Modifiers': 'Modifiers Misplacement',
    'Misplaced Modifiers': 'Modifiers Misplacement',
    'Pronoun-Antecedent Agreement': 'Pronoun/Relative Clause',
    'Pronoun Case Errors': 'Pronoun/Relative Clause',
    'Relative Clause Errors': 'Pronoun/Relative Clause',
    'Vague Pronoun Reference': 'Pronoun/Relative Clause',
    'Spelling': 'Spelling Mistakes',
    'Capitalization Errors': 'Mechanics/Punctuation',
    'Punctuation Errors': 'Mechanics/Punctuation',
    'Typographical Errors': 'Mechanics/Punctuation',
    'Comma Splices': 'Mechanics/Punctuation',
    'Apostrophe Usage': 'Mechanics/Punctuation',
    'Quotation Mark Usage': 'Mechanics/Punctuation',
    'Double Negatives': 'Negation Errors',
    'Incorrect Negative Forms': 'Negation Errors',
    'Inappropriate Register': 'Style/Register',
    'Colloquialisms or Slang': 'Style/Register',
    'Slang, Jargon, and Colloquialisms': 'Style/Register',
    'Clichés': 'Style/Register',
    'Idiom Errors': 'Style/Register',
    'Ambiguity': 'Style/Register',
    'Word Order Errors': 'Sentence Structure',
    'Sentence Structure Errors': 'Sentence Structure',
    'Conditional Sentence Errors': 'Mixed Conditionals',
    'Subjunctive Mood Errors': 'Mixed Conditionals',
    'Comparatives and Superlatives': 'Comparison Errors',
    'Agreement in Comparative and Superlative Forms': 'Comparison Errors',
    'Quantifier Errors': 'Quantifier Errors',
    'Countable and Uncountable Noun Errors': 'Quantifier Errors',
    'Conjunction Errors': 'Mechanics/Punctuation' 
}

df['mapped_target'] = df['target_label'].replace(mapping)
df['mapped_api'] = df['api_label'].replace(mapping)

df_clean = df.dropna(subset=['mapped_target', 'mapped_api'])
df_clean = df_clean[~df_clean['mapped_api'].str.contains("Error", na=False)]

df_clean['is_correct'] = df_clean['mapped_target'] == df_clean['mapped_api']

print(f"Data loading complete, valid rows: {len(df_clean)}")

df_clean['sent_length'] = df_clean['incorrect_sentence'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 6))
sns.boxplot(x='is_correct', y='sent_length', data=df_clean, palette="Set2")
plt.title('Impact of Sentence Length on Classification Accuracy')
plt.xlabel('Is Prediction Correct? (False vs True)')
plt.ylabel('Sentence Length (Words)')
plt.savefig('analysis_length_impact.png')
print("Plot 1 saved: analysis_length_impact.png")
plt.close()

def get_change_ratio(str1, str2):
    return difflib.SequenceMatcher(None, str(str1), str(str2)).ratio()

df_clean['sim_gt'] = df_clean.apply(lambda row: get_change_ratio(row['incorrect_sentence'], row['gt_correction']), axis=1)
df_clean['sim_gemini'] = df_clean.apply(lambda row: get_change_ratio(row['incorrect_sentence'], row['api_correction']), axis=1)

df_clean['change_mag_gt'] = 1 - df_clean['sim_gt']
df_clean['change_mag_gemini'] = 1 - df_clean['sim_gemini']

plt.figure(figsize=(10, 6))
sns.kdeplot(df_clean['change_mag_gt'], label='Ground Truth Changes', shade=True, color='blue', alpha=0.3)
sns.kdeplot(df_clean['change_mag_gemini'], label='Gemini Changes', shade=True, color='orange', alpha=0.3)
plt.title('Distribution of Correction Magnitude (Human vs AI)')
plt.xlabel('Change Magnitude (0 = No Change, 1 = Complete Rewrite)')
plt.ylabel('Density')
plt.legend()
plt.savefig('analysis_edit_magnitude.png')
print("Plot 2 saved: analysis_edit_magnitude.png")
plt.close()

counts_target = df_clean['mapped_target'].value_counts().reset_index()
counts_target.columns = ['Label', 'Count']
counts_target['Source'] = 'Ground Truth'

counts_api = df_clean['mapped_api'].value_counts().reset_index()
counts_api.columns = ['Label', 'Count']
counts_api['Source'] = 'Gemini'

combined_counts = pd.concat([counts_target, counts_api])
top_labels = combined_counts.groupby('Label')['Count'].sum().sort_values(ascending=False).head(10).index
filtered_counts = combined_counts[combined_counts['Label'].isin(top_labels)]

plt.figure(figsize=(12, 8))
sns.barplot(y='Label', x='Count', hue='Source', data=filtered_counts, palette='viridis')
plt.title('Top 10 Error Types: Ground Truth vs Gemini Distribution')
plt.xlabel('Count')
plt.ylabel('Error Type')
plt.legend(title='Source')
plt.tight_layout()
plt.savefig('analysis_label_distribution.png')
print("Plot 3 saved: analysis_label_distribution.png")
plt.close()

print("\nAll analyses complete!")