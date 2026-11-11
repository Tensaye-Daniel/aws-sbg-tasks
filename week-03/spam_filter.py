"""Week 03 - Spam Email / Message Detector

Builds, tests, and tries to break a classical machine learning spam filter
(TF-IDF + Multinomial Naive Bayes) on the SMS Spam Collection dataset.

The script first reports baseline accuracy, precision and recall on a held-out
test set, then feeds five hand-written adversarial messages to the trained
filter to see which ones slip through.

The dataset is downloaded from UCI at runtime and is never stored in the repo.

Run it with:
    pip install -r requirements.txt
    python spam_filter.py
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)
ZIP_PATH = BASE_DIR / "smsspamcollection.zip"
DATA_PATH = BASE_DIR / "sms_data" / "SMSSpamCollection"

ADVERSARIAL_MESSAGES: List[Tuple[str, str, str]] = [
    ("Cl@im your fr33 g1ft card now", "spam", "Spelling/obfuscation"),
    ("Not free, just confirming lunch at 1pm", "ham", "Negation"),
    ("URGENT! You have won a free vacation. Click here!", "spam", "Classic spam wording"),
    ("Hey, are we still meeting for our project review tomorrow?", "ham", "Normal email"),
    ("Congratulations, cash prize waiting for you!", "spam", "Direct spam"),
]

EXPLANATIONS: List[str] = [
    "Fooled: 'Cl@im', 'fr33' and 'g1ft' tokenize into unseen tokens ('cl', 'im', 'fr33', 'g1ft') with zero spam evidence, so ordinary words like 'your', 'card' and 'now' decide the vote for ham.",
    "Right answer, wrong reason: bag-of-words cannot parse the negation 'Not free'; everyday words ('just', 'lunch', 'pm') and the strong ham prior simply outweighed the trigger word 'free'.",
    "Fooled: TF-IDF length normalization dilutes the trigger words across 8 tokens, and words like 'urgent', 'you' and 'have' also occur in personal texts, so the ~87% ham prior tips it to ham.",
    "Correct: plain conversational wording with no prize or offer trigger words looks exactly like the ham messages the model learned from.",
    "Caught: 'congratulations' and 'cash prize' are strong, correctly spelled spam triggers that dominate the short message's TF-IDF weights.",
]


def download_dataset(url: str = DATASET_URL) -> Path:
    """Download and extract the SMS Spam Collection from UCI."""
    if not DATA_PATH.exists():
        print("Downloading dataset from UCI...")
        urllib.request.urlretrieve(url, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_PATH.parent)
    print(f"Dataset ready at {DATA_PATH}")
    return DATA_PATH


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the tab-separated (label, message) pairs into a DataFrame."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    print(f"Dataset shape: {df.shape}")
    return df


def train_filter(df: pd.DataFrame) -> Tuple[TfidfVectorizer, MultinomialNB]:
    """Vectorize with TF-IDF, train a Multinomial Naive Bayes filter, report baseline metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    vec = TfidfVectorizer()
    model = MultinomialNB()
    model.fit(vec.fit_transform(X_train), y_train)

    y_pred = model.predict(vec.transform(X_test))
    print("--- Baseline Model Performance ---")
    print(f"Accuracy:         {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (spam):  {precision_score(y_test, y_pred, pos_label='spam'):.4f}")
    print(f"Recall (spam):     {recall_score(y_test, y_pred, pos_label='spam'):.4f}")
    return vec, model


def run_adversarial_tests(vec: TfidfVectorizer, model: MultinomialNB) -> None:
    """Predict the adversarial messages and compare against the correct labels."""
    messages = [msg for msg, _, _ in ADVERSARIAL_MESSAGES]
    predictions = model.predict(vec.transform(messages))

    print("\n--- Adversarial Test Messages ---")
    fooled = 0
    for (msg, correct, trick), predicted, why in zip(
        ADVERSARIAL_MESSAGES, predictions, EXPLANATIONS
    ):
        was_fooled = predicted != correct
        fooled += int(was_fooled)
        print(f"\nMessage:    {msg}")
        print(f"Trick:      {trick}")
        print(f"Model said: {predicted}")
        print(f"Should be:  {correct}")
        print(f"Fooled:     {'Yes' if was_fooled else 'No'}")
        print(f"Why:        {why}")

    print(f"\n{fooled} of {len(messages)} messages fooled the filter.")


def predict_message(vec: TfidfVectorizer, model: MultinomialNB, message: str) -> str:
    """Classify a single custom message."""
    return model.predict(vec.transform([message]))[0]


def main() -> None:
    """Run the baseline training and the adversarial tests."""
    parser = argparse.ArgumentParser(description="Spam Email / Message Detector")
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Optional custom message to classify after the tests",
    )
    args = parser.parse_args()

    df = load_dataset(download_dataset())
    vec, model = train_filter(df)
    run_adversarial_tests(vec, model)

    if args.message:
        label = predict_message(vec, model, args.message)
        print(f"\nYour message: {args.message!r} -> {label}")


if __name__ == "__main__":
    main()
