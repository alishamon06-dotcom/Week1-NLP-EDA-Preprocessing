from pathlib import Path
import io
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for folder in (DATA, REPORTS, MODELS):
    folder.mkdir(exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_data():
    local = DATA / "SMSSpamCollection"
    if local.exists():
        return pd.read_csv(local, sep="\t", header=None, names=["label", "text"])
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
        with archive.open(member) as stream:
            return pd.read_csv(stream, sep="\t", header=None, names=["label", "text"])

def main():
    df = load_data()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df.label.isin(["ham", "spam"])].drop_duplicates(["label", "text"]).copy()
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    print("Rows:", len(df))
    print("Class distribution:\n", df.label.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.target, test_size=0.20, random_state=42, stratify=df.target
    )
    candidates = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("model", MultinomialNB(alpha=0.5))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("model", LinearSVC(class_weight="balanced", random_state=42))
        ])
    }

    rows, best_f1, best_model = [], -1, None
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "spam_precision": precision_score(y_test, pred, zero_division=0),
            "spam_recall": recall_score(y_test, pred, zero_division=0),
            "spam_f1": f1_score(y_test, pred, zero_division=0)
        }
        rows.append(scores)
        print("\n", name, scores)
        print(classification_report(y_test, pred, target_names=["ham", "spam"],
                                    zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        display = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
        display.plot(values_format="d")
        plt.title("Confusion Matrix - " + name)
        plt.tight_layout()
        filename = "naive_bayes" if "Naive" in name else "linear_svm"
        plt.savefig(REPORTS / f"confusion_matrix_{filename}.png", dpi=160)
        plt.close()
        (REPORTS / f"classification_report_{filename}.txt").write_text(
            classification_report(y_test, pred, target_names=["ham", "spam"],
                                  zero_division=0), encoding="utf-8")
        if scores["spam_f1"] > best_f1:
            best_f1, best_model = scores["spam_f1"], model

    pd.DataFrame(rows).to_csv(REPORTS / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODELS / "best_text_classifier.joblib")
    print("\nMeasured outputs saved in", REPORTS)

if __name__ == "__main__":
    main()
