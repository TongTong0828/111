import pandas as pd

df = pd.read_csv("Grammar_Correction_with_GPT.csv")
strict_acc = (df["Error Type"] == df["GPT_Error_Type"]).mean()
print("Strict label accuracy:", strict_acc)
per_type_acc = (
    df.assign(correct = df["Error Type"] == df["GPT_Error_Type"])
      .groupby("Error Type")["correct"]
      .mean()
      .sort_values(ascending=False)
)
print(per_type_acc)
