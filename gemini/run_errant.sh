#!/bin/bash

echo "--- Script: run_errant.sh ---"

SPACY_PATH=$(python -m spacy info --path)
if [ ! -L "$SPACY_PATH/en" ]; then
    echo "Creating 'en' model link for ERRANT..."
    ln -s "$SPACY_PATH/en_core_web_sm" "$SPACY_PATH/en"
else
    echo "'en' model link already exists."
fi

echo "\n[1/3] Analyzing 'API Corrections' (api_edits.m2)..."
errant_parallel -orig source.txt -cor hypothesis.txt -out api_edits.m2

echo "\n[2/3] Analyzing 'Ground Truth Reference' (ref_edits.m2)..."
errant_parallel -orig source.txt -cor reference.txt -out ref_edits.m2

echo "\n[3/3] Comparing versions and calculating F0.5 score..."
errant_compare -hyp api_edits.m2 -ref ref_edits.m2

echo "\n--- ERRANT Evaluation Complete ---"