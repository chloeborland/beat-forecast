import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# ============================================================
# Paths (edit HIT_TRAIN_CSV + TARGET_COL to match your data)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # adjust if needed
MODELS_DIR = BASE_DIR / "models"      # or BASE_DIR if that's where you store hit model

HIT_TRAIN_CSV = DATA_DIR / "hit_training.csv"  # <-- YOU set this
OUT_PATH = BASE_DIR / "hit_gb_pipeline_wade_compat.joblib"  # overwrite existing if you want

TARGET_COL = "is_hit"  # <-- YOU set this (e.g., "hit", "hit_label", etc.)

# ============================================================
# Feature columns expected by your APP row
# (these match your streamlit inputs + extractor output)
# ============================================================
AUDIO_COLS = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "duration_ms",
    "key",
    "mode",
]

NUMERIC_COLS = ["total_artist_followers", "year"] + AUDIO_COLS
CAT_COLS = ["genre"]  # this will become genre_* one-hots

def main():
    if not HIT_TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training file not found: {HIT_TRAIN_CSV}")

    df = pd.read_csv(HIT_TRAIN_CSV)

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{TARGET_COL}' not found in CSV columns: {list(df.columns)}")

    # ---- Rename followers column if needed (common cases)
    if "total_artist_followers" not in df.columns:
        for cand in ["spotify_followers", "followers", "artist_followers", "total_followers"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "total_artist_followers"})
                break

    # ---- Basic sanity: ensure required columns exist
    missing = [c for c in (NUMERIC_COLS + CAT_COLS) if c not in df.columns]
    if missing:
        raise ValueError(
            "Training CSV is missing required columns.\n"
            f"Missing: {missing}\n"
            "Fix by renaming columns or adding them to the training table."
        )

    X = df[NUMERIC_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].astype(int).copy()

    # ---- Preprocess
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, NUMERIC_COLS),
            ("cat", categorical_tf, CAT_COLS),
        ],
        remainder="drop"
    )

    # ---- Classifier (keep it simple/stable)
    clf = GradientBoostingClassifier(random_state=42)

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", clf),
    ])

    # ---- Train / evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    # ---- Metrics
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"ROC AUC: {auc:.3f}")
    print(f"Avg Precision (PR AUC): {ap:.3f}")
    print(f"Saved model -> {OUT_PATH}")

    joblib.dump(pipe, OUT_PATH)

if __name__ == "__main__":
    main()