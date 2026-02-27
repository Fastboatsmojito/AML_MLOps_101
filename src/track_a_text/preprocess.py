"""
Data cleaning and text preprocessing utilities for the Contoso inspection
classification use case.
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def clean_inspection_text(text: str) -> str:
    """Clean a single inspection comment."""
    if pd.isna(text):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_and_clean_inspections(filepath: str) -> pd.DataFrame:
    """Load the inspections dataset and apply cleaning."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    df["comment_clean"] = df["comment"].apply(clean_inspection_text)
    df = df[df["comment_clean"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["is_lead_opportunity"].astype(int)
    return df


def build_features(
    df: pd.DataFrame,
    text_column: str = "comment_clean",
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    vectorizer: TfidfVectorizer = None,
    fit: bool = True,
):
    """
    Build TF-IDF features from cleaned text.
    Returns feature matrix, labels, and the fitted vectorizer.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )

    if fit:
        X = vectorizer.fit_transform(df[text_column])
    else:
        X = vectorizer.transform(df[text_column])

    y = df["label"].values if "label" in df.columns else None
    return X, y, vectorizer


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
):
    """End-to-end: clean, split, featurize. Returns train/test splits + vectorizer."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    X_train, y_train, vectorizer = build_features(
        train_df, max_features=max_features, fit=True
    )
    X_test, y_test, _ = build_features(
        test_df, vectorizer=vectorizer, fit=False
    )

    return X_train, X_test, y_train, y_test, vectorizer, train_df, test_df
