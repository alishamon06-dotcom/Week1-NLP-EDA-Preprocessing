# Week 1 NLP Project: Exploratory Data Analysis and Preprocessing

This project explores the **UCI SMS Spam Collection**. It includes an EDA script, a Jupyter notebook, a student-style report, and generated output folders.

## Dataset
- Source: UCI Machine Learning Repository — SMS Spam Collection
- DOI: https://doi.org/10.24432/C5CC84
- Official page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Size: 5,574 English SMS messages
- Labels: `ham` (legitimate) and `spam`
- License: CC BY 4.0 (cite the original dataset authors)

The dataset is downloaded automatically by the script on first run. The raw dataset is not committed here.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src_eda.py
```

The script writes cleaned data, summary statistics, frequency tables and charts to `outputs/`.

## What is examined
- Dataset shape, labels, missing values and duplicate messages
- Character and word-count distributions
- Frequent unigrams and bigrams
- Lowercasing, URL/number placeholders, tokenization and stop-word filtering
- Optional VADER sentiment polarity (descriptive only)

## Important interpretation note
The original target is **spam vs ham**, not positive vs negative sentiment. The optional VADER output is a separate lexicon-based estimate and must not be treated as the dataset's ground-truth label or as a model evaluation score.

## Reproducibility
All dataset-derived counts, frequency tables and plots are calculated by running the code against the downloaded dataset. The report only quotes the UCI dataset size and published class counts as reference information. Run the script to generate fresh local outputs before submission.

## Citation
Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84
