import os
import time
import json
import glob
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

read_file_path = "Grammar_Correction.csv"
output_path = "Grammar_Correction_with_GPT.csv"
df = pd.read_csv(read_file_path)

prompt = (
    "You are a professional English teacher grading students' sentences.\n"
    "Each input sentence contains exactly one main grammar or usage error "
    "(it may also include a spelling mistake).\n\n"
    "Your tasks:\n"
    "1) Identify the single most appropriate error type from the following 36 labels "
    "(copy the label text exactly):\n"
    "   - Capitalization Errors\n"
    "   - Preposition Usage\n"
    "   - Infinitive Errors\n"
    "   - Faulty Comparisons\n"
    "   - Mixed Conditionals\n"
    "   - Contractions Errors\n"
    "   - Incorrect Auxiliaries\n"
    "   - Clichés\n"
    "   - Ambiguity\n"
    "   - Relative Clause Errors\n"
    "   - Conjunction Misuse\n"
    "   - Verb Tense Errors\n"
    "   - Ellipsis Errors\n"
    "   - Quantifier Errors\n"
    "   - Spelling Mistakes\n"
    "   - Passive Voice Overuse\n"
    "   - Agreement in Comparative and Superlative Forms\n"
    "   - Lack of Parallelism in Lists or Series\n"
    "   - Abbreviation Errors\n"
    "   - Modifiers Misplacement\n"
    "   - Punctuation Errors\n"
    "   - Tautology\n"
    "   - Parallelism Errors\n"
    "   - Run-on Sentences\n"
    "   - Negation Errors\n"
    "   - Word Choice/Usage\n"
    "   - Pronoun Errors\n"
    "   - Inappropriate Register\n"
    "   - Sentence Fragments\n"
    "   - Article Usage\n"
    "   - Subject-Verb Agreement\n"
    "   - Mixed Metaphors/Idioms\n"
    "   - Redundancy/Repetition\n"
    "   - Sentence Structure Errors\n"
    "   - Slang, Jargon, and Colloquialisms\n"
    "   - Gerund and Participle Errors\n\n"
    "2) Rewrite the sentence as a grammatically correct, natural-sounding English sentence "
    "while preserving the original meaning.\n\n"
    "Return ONLY a JSON object (no explanation, no extra text). "
    "The JSON must have exactly this structure:\n"
    "{ \"error_type\": \"...\", \"corrected_sentence\": \"...\" }"
)

gpt_error_types = []
gpt_corrections = []

for i, sentence in enumerate(df["Ungrammatical Statement"]):
    print("processing "f"{i}: {sentence}")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": sentence},
        ],
    )
    content = response.choices[0].message.content
    data = json.loads(content)

    gpt_error_types.append(data["error_type"])
    gpt_corrections.append(data["corrected_sentence"])

    error_type = data["error_type"]
    corrected = data["corrected_sentence"]

    df.loc[i, "GPT_Error_Type"] = error_type
    df.loc[i, "GPT_Correction"] = corrected

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"saved row {i} to {output_path}")

