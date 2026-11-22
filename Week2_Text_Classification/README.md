# Week 2 – Text Classification Strategies

SMS Spam Detection using TF-IDF, Multinomial Naive Bayes and Linear SVM.

## Dataset
UCI SMS Spam Collection: https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
DOI: https://doi.org/10.24432/C5CC84

The dataset contains SMS messages labelled as ham (ordinary) or spam (unwanted). The training script downloads the public UCI archive automatically.

## Run
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python src/train_classifier.py
```

## Pipeline
1. Load the public UCI data and inspect missing/duplicate records.
2. Split into stratified training and test sets (80/20).
3. Convert text into unigram/bigram TF-IDF features.
4. Train Multinomial Naive Bayes and Linear SVM pipelines.
5. Compare accuracy, spam precision, spam recall and spam F1.
6. Save confusion matrices, classification reports, comparison CSV and the selected model.

TF-IDF is fitted inside each scikit-learn Pipeline using training data only, preventing test-set vocabulary leakage. The script generates measured results when run; no unverified score is claimed.

## Files
- `src/train_classifier.py`: training and evaluation code
- `requirements.txt`: dependencies
- `reports/`: generated metrics and plots (created when run)
- `models/`: saved pipeline (created when run)

The accompanying DOCX report and notebook are included in the downloadable project ZIP shared in the conversation.