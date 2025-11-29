# 🤖 Gemini API Syntax Evaluation: A Baseline Error Correction Analysis for Smart Models

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![API](https://img.shields.io/badge/Gemini%20Model-2.5%20Flash-yellow)](https://ai.google.dev/models)

This project provides a systematic framework for evaluating the capabilities of the **Google Gemini API (gemini-2.5-flash)** on two key Natural Language Processing tasks, and for reverse-validating the label quality of public datasets.

---
# Workflow
 cd tech-review 

python3.11 -m venv tech
source tech/bin/activate

pip install -r requirements.txt

python -m spacy download en_core_web_sm

export GOOGLE_API_KEY='AIza...'

python geminiApi.py

python combined.py
python graph.py 

python prepare_errant.py

chmod +x run_errant.sh

./run_errant.sh


## 🎯 Project Goals and Evaluation Tasks

This project uses the public [Kaggle Grammatical Error Correction dataset](https://www.kaggle.com/datasets/...) as a benchmark, focusing on evaluating Gemini's following capabilities:

| Task | Goal | Core Metric |
| :--- | :--- | :--- |
| **Task A: Fine-grained Error Classification** | Accurately classify ungrammatical sentences into **30+ specific grammatical error types**. | Confusion Matrix and Classification Report |
| **Task B: Grammatical Error Correction (GEC)** | Generate a **grammatically correct and fluent** corrected version for the ungrammatical sentence. | ERRANT evaluation framework's F0.5 Score |


---


## 🛠️ Technology Stack and Dependencies

This project relies on the following core Python libraries for API connection, data processing, and industry-standard evaluation.

```markdown
google-generativeai
pandas
errant
spacy
scikit-learn
seaborn/