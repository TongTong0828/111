import difflib

import pandas as pd

df = pd.read_csv("Grammar_Correction_with_GPT.csv")
def norm(s):
    s = str(s).strip().lower()
    return " ".join(s.split())
df["std_norm"] = df["Standard English"].apply(norm)
df["gpt_norm"] = df["GPT_Correction"].apply(norm)

exact_match_acc = (df["std_norm"] == df["gpt_norm"]).mean()
print("Exact correction match accuracy:", exact_match_acc)

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()
df["sim"] = df.apply(lambda row: similarity(row["std_norm"], row["gpt_norm"]), axis=1)
high_quality_ratio = (df["sim"] >= 0.9).mean()

print("High-quality corrections (similarity >= 0.9):", high_quality_ratio)
