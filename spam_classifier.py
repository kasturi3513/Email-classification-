
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = Path(__file__).with_name("emails.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "emails.csv not found. Keep it beside this script."
        )

    data = pd.read_csv(DATA_PATH)

    required = {"label", "text"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain label and text columns.")

    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"])]
    data = data[data["text"].str.len() > 0]
    data = data.drop_duplicates(subset=["text"])

    if data["label"].nunique() < 2:
        raise ValueError("Dataset must contain ham and spam examples.")

    print("Dataset examples:", len(data))
    print("\nClass counts:")
    print(data["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", ComplementNB(alpha=0.5))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Model Evaluation ===")
    print("Test examples:", len(X_test))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        zero_division=0
    ))

    print("Confusion matrix (label order: ham, spam):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    ))

    print("\n=== Try your own email text ===")
    while True:
        try:
            message = input(
                "\nPaste email text (or type 'quit' to exit): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nProgram stopped.")
            break

        if message.lower() == "quit":
            break

        
        if not message:
            print("Please enter some text.")
            continue

        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = model.named_steps["classifier"].classes_

        print("Prediction:", prediction.upper())

        for label, probability in zip(classes, probabilities):
            print(f"{label.upper()} probability: {probability:.2%}")

        print("Note: This is a learning demo; verify important emails manually.")




if __name__ == "__main__":
    main()
