import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

# ---------------- Page Config ----------------
st.set_page_config(page_title="Beat Forecast", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# --- Model / asset paths (standardize everything) ---
MODELS_DIR = BASE_DIR / "models"
POP_MODEL_PATH = MODELS_DIR / "pop_rf_pipeline.joblib"
HIT_MODEL_PATH = MODELS_DIR / "hit_gb_pipeline_wade.joblib"

CLUSTER_SCALER_PATH = BASE_DIR / "cluster_scaler.pkl"
KMEANS_PATH = BASE_DIR / "kmeans_k6.pkl"
CLUSTER_FEATS_PATH = BASE_DIR / "cluster_feature_list.json"
CLUSTER_LABELS_PATH = BASE_DIR / "cluster_labels.csv"
CLUSTER_HIT_SUMMARY_PATH = BASE_DIR / "cluster_hit_summary.csv"
CLUSTER_TOP2024_PATH = BASE_DIR / "cluster_top2024_lift.csv"

# ---------------- Cached Loaders ----------------
@st.cache_resource
def load_regression_pipeline():
    return joblib.load(POP_MODEL_PATH.as_posix())

@st.cache_resource
def load_gb_hit_model():
    return joblib.load(HIT_MODEL_PATH.as_posix())

@st.cache_resource
def load_clustering_assets():
    scaler = joblib.load(CLUSTER_SCALER_PATH.as_posix())
    kmeans = joblib.load(KMEANS_PATH.as_posix())
    with open(CLUSTER_FEATS_PATH.as_posix(), "r") as f:
        feats = json.load(f)

    labels = pd.read_csv(CLUSTER_LABELS_PATH.as_posix())          # cluster,name,description
    hit_summary = pd.read_csv(CLUSTER_HIT_SUMMARY_PATH.as_posix())# cluster,name,hit_rate,...
    try:
        top2024 = pd.read_csv(CLUSTER_TOP2024_PATH.as_posix())    # name, lift, etc
    except Exception:
        top2024 = None

    return scaler, kmeans, feats, labels, hit_summary, top2024

# ---------------- Utility: schema + importance (Chloe) ----------------
def safe_get_feature_names(pipeline):
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def safe_get_feature_importance(pipeline):
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
    return None

# ---------------- Clustering inference ----------------
def predict_archetype(user_inputs: dict):
    scaler, kmeans, feats, labels, hit_summary, top2024 = load_clustering_assets()

    X = pd.DataFrame([{f: user_inputs.get(f) for f in feats}]).reindex(columns=feats)
    Xs = scaler.transform(X)
    cid = int(kmeans.predict(Xs)[0])

    lab = labels[labels["cluster"] == cid].head(1)
    name = lab["name"].iloc[0] if len(lab) else f"Cluster {cid}"
    desc = lab["description"].iloc[0] if len(lab) else ""

    hs = hit_summary[hit_summary["cluster"] == cid].head(1)
    t24 = top2024[top2024["name"] == name].head(1) if top2024 is not None else None

    return cid, name, desc, hs, t24

# ---------------- Helper: Build model-ready input for regression ----------------
def build_regression_input(df_row: pd.Series, expected_cols: list[str]) -> pd.DataFrame:
    """
    Builds a 1-row dataframe that matches the regression model's expected feature columns.
    """
    X = pd.DataFrame([{c: 0 for c in expected_cols}])

    # Map app fields to model fields
    if "total_artist_followers" in expected_cols:
        X.loc[0, "total_artist_followers"] = float(df_row["followers"])
    if "avg_artist_popularity" in expected_cols:
        X.loc[0, "avg_artist_popularity"] = float(df_row["artist_popularity"])

    direct_map = {
        "danceability": "danceability",
        "energy": "energy",
        "loudness": "loudness",
        "tempo": "tempo",
        "valence": "valence",
        "speechiness": "speechiness",
        "acousticness": "acousticness",
        "instrumentalness": "instrumentalness",
        "liveness": "liveness",
        "duration_ms": "duration_ms",
        "year": "year",
        "key": "key",
        "mode": "mode",
    }

    for app_k, model_k in direct_map.items():
        if model_k in expected_cols:
            X.loc[0, model_k] = float(df_row[app_k])

    # Optional: genre one-hot if regression model expects it
    if "primary_genre" in df_row.index:
        genre_col = f"genre_{df_row['primary_genre']}"
        if genre_col in expected_cols:
            X.loc[0, genre_col] = 1

    return X[expected_cols]

# ---------------- Header ----------------
st.title("Beat Forecast")
st.caption("Data-driven decision support for estimating Spotify performance prior to release.")
st.divider()

# ---------------- Market Context ----------------
col_mc1, col_mc2 = st.columns([2, 1])
with col_mc1:
    st.subheader("Market Context")
    st.write(
        "Spotify breakout success is structurally rare. "
        "Only a small fraction of tracks achieve top-tier performance. "
        "This tool evaluates whether a song aligns with structural success patterns, "
        "predicts hit likelihood, and forecasts expected popularity prior to release."
    )
with col_mc2:
    st.metric("Observed Hit Rate (2024)", "0.22%")

st.divider()

# ---------------- Sidebar Inputs (unified) ----------------
st.sidebar.header("Model Inputs")

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Spotify Followers", min_value=0, value=5000, step=100)
    artist_pop = st.slider("Artist Popularity (0–100)", 0, 100, 35)

with st.sidebar.expander("Audio Features", expanded=True):
    danceability = st.slider("Danceability", 0.0, 1.0, 0.60)
    energy = st.slider("Energy", 0.0, 1.0, 0.70)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -7.0)
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.10)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.10)
    key = st.selectbox("Key (0–11)", list(range(12)), index=5)

    mode_label = st.selectbox("Mode", ["Minor (0)", "Major (1)"], index=1)
    mode = 0 if mode_label.startswith("Minor") else 1

    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)
    duration_ms = st.number_input("Duration (ms)", min_value=30_000, value=210_000, step=1_000)

with st.sidebar.expander("Regression Defaults (if needed)", expanded=False):
    year = st.slider("Release Year", 1950, 2026, 2024)

with st.sidebar.expander("Genre (optional)", expanded=False):
    primary_genre = st.selectbox(
        "Primary Genre",
        ["Classical", "Country", "Electronic", "Folk", "Hip-Hop", "Jazz", "Pop", "R&B", "Rock"],
        index=6
    )

show_debug = st.sidebar.checkbox("Show debug (feature alignment)", value=False)

st.sidebar.divider()

# Persist run state so results stay visible after click
if "run_forecast" not in st.session_state:
    st.session_state["run_forecast"] = False

def do_run():
    st.session_state["run_forecast"] = True

def do_reset():
    st.session_state["run_forecast"] = False

st.sidebar.button("Run Forecast", type="primary", on_click=do_run)
st.sidebar.button("Reset", on_click=do_reset)

run = st.session_state["run_forecast"]

# ---------------- Build unified display inputs ----------------
inputs_display = pd.DataFrame([{
    "followers": followers,
    "artist_popularity": artist_pop,
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "tempo": tempo,
    "valence": valence,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "duration_ms": duration_ms,
    "key": key,
    "mode": mode,
    "year": year,
    "primary_genre": primary_genre,
}])

# ---------------- Main Layout ----------------
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Breakout Probability")
    hit_metric = st.empty()
with col2:
    st.subheader("Popularity Forecast")
    pop_metric = st.empty()
with col3:
    st.subheader("Executive Recommendation")
    rec_box = st.empty()

st.divider()

# ---------------- Input Summary ----------------
st.subheader("Input Summary")
st.dataframe(
    inputs_display.drop(columns=["primary_genre"]),
    use_container_width=True,
    hide_index=True
)
st.divider()

# ---------------- Scenario Comparison ----------------
st.subheader("Scenario Comparison")
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []

colS1, colS2 = st.columns([1, 1])
with colS1:
    if st.button("Save Current Scenario"):
        st.session_state["scenarios"].append(inputs_display.iloc[0].to_dict())
with colS2:
    if st.button("Clear Scenarios"):
        st.session_state["scenarios"] = []

if st.session_state["scenarios"]:
    st.dataframe(pd.DataFrame(st.session_state["scenarios"]), use_container_width=True, hide_index=True)
else:
    st.caption("Save multiple input sets to compare tradeoffs across songs or mixes.")

st.divider()

# ---------------- Run Forecast ----------------
if run:
    # --- Sanity checks for required artifacts ---
    missing = []
    if not HIT_MODEL_PATH.exists(): missing.append(HIT_MODEL_PATH.as_posix())
    if not POP_MODEL_PATH.exists(): missing.append(POP_MODEL_PATH.as_posix())
    for p in [CLUSTER_SCALER_PATH, KMEANS_PATH, CLUSTER_FEATS_PATH, CLUSTER_LABELS_PATH, CLUSTER_HIT_SUMMARY_PATH]:
        if not p.exists():
            missing.append(p.as_posix())

    if missing:
        st.error("Missing required files:\n- " + "\n- ".join(missing))
        st.stop()

    row = inputs_display.iloc[0]

    # ---------------- 1) GB Hit Probability (classification) ----------------
    gb_model = load_gb_hit_model()

    # Build model-ready X for GB model using its expected columns
    if not hasattr(gb_model, "feature_names_in_"):
        st.error("GB model is missing feature_names_in_. Re-train with pandas DataFrame to preserve schema.")
        st.stop()

    gb_expected = gb_model.feature_names_in_.tolist()

    # Map fields into the GB schema
    gb_base = {
        "danceability": row["danceability"],
        "energy": row["energy"],
        "loudness": row["loudness"],
        "speechiness": row["speechiness"],
        "acousticness": row["acousticness"],
        "instrumentalness": row["instrumentalness"],
        "liveness": row["liveness"],
        "valence": row["valence"],
        "tempo": row["tempo"],
        "duration_ms": row["duration_ms"],
        "key": row["key"],
        "mode": row["mode"],
        "total_artist_followers": row["followers"],
        "avg_artist_popularity": row["artist_popularity"],
    }

    X_gb = pd.DataFrame([gb_base]).reindex(columns=gb_expected)
    breakout_prob = float(gb_model.predict_proba(X_gb)[:, 1][0])
    hit_metric.metric("Hit Likelihood", f"{round(breakout_prob * 100, 1)}%")

    # ---------------- 2) Regression Popularity Forecast ----------------
    reg_pipeline = load_regression_pipeline()
    reg_expected = safe_get_feature_names(reg_pipeline)

    if reg_expected is None:
        st.error("Regression pipeline missing feature schema (feature_names_in_).")
        st.stop()

    X_reg = build_regression_input(row, reg_expected)

    try:
        pred_pop = float(reg_pipeline.predict(X_reg)[0])
        pred_pop_display = max(0.0, min(100.0, pred_pop))
        pop_metric.metric("Predicted Popularity (0–100)", round(pred_pop_display, 1))
    except Exception as e:
        st.error(f"Regression prediction failed: {e}")
        st.stop()

    # ---------------- 3) Executive Recommendation ----------------
    if pred_pop_display >= 70:
        pop_band = "Strong"
    elif pred_pop_display >= 45:
        pop_band = "Moderate"
    else:
        pop_band = "Low"

    if breakout_prob >= 0.40 and pred_pop_display >= 55:
        rec_box.success("Strong profile: high breakout signal + solid popularity forecast. Greenlight release + promotion.")
    elif breakout_prob >= 0.25 and pred_pop_display >= 45:
        rec_box.info(f"Promising: breakout signal moderate and popularity forecast is {pop_band.lower()}. Consider tweaks + stronger marketing plan.")
    elif pred_pop_display >= 60 and breakout_prob < 0.25:
        rec_box.info("Popularity forecast solid, breakout signal weak. Consider collabs/playlist strategy to raise breakout odds.")
    else:
        rec_box.warning("Caution: projected breakout probability and popularity are low. Consider revising production/positioning/promo.")

    # ---------------- Debug (optional) ----------------
    if show_debug:
        st.divider()
        st.subheader("Debug: Feature Alignment")
        st.write("GB expects:", gb_expected)
        st.dataframe(X_gb, use_container_width=True, hide_index=True)
        st.write("Regression expects:", reg_expected)
        st.dataframe(X_reg, use_container_width=True, hide_index=True)

    st.divider()

    # ---------------- Song Archetype (Clustering) ----------------
    st.subheader("Song Archetype (Clustering)")
    cluster_inputs = {
        "danceability": danceability,
        "energy": energy,
        "valence": valence,
        "tempo": tempo,
        "loudness": loudness,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "duration_ms": duration_ms,
    }

    try:
        cid, archetype, archetype_desc, hs, t24 = predict_archetype(cluster_inputs)
        st.write(f"**{archetype}**")
        if archetype_desc:
            st.caption(archetype_desc)

        if len(hs):
            c1, c2, c3 = st.columns(3)
            c1.metric("Hit rate (pop≥70)", f"{hs['hit_rate'].iloc[0]:.3f}")

            rel_col = (
                "relative_to_overall"
                if "relative_to_overall" in hs.columns
                else ("enrichment_ratio" if "enrichment_ratio" in hs.columns else None)
            )
            c2.metric("Relative to overall", f"{hs[rel_col].iloc[0]:.2f}×" if rel_col else "—")

            c3.metric("Avg popularity", f"{hs['avg_popularity'].iloc[0]:.1f}" if "avg_popularity" in hs.columns else "—")

        if t24 is not None and len(t24) and "lift" in t24.columns:
            st.metric("Top-2024 lift", f"{t24['lift'].iloc[0]:.2f}×")

    except Exception as e:
        st.error(f"Clustering could not run. Details: {e}")

    st.divider()

    # ---------------- Production Assessment ----------------
    st.subheader("Production Assessment")
    strengths, weaknesses = [], []

    if energy > 0.65: strengths.append("Energy aligns with high-performing tracks.")
    else: weaknesses.append("Energy below common hit threshold (0.65).")

    if danceability > 0.60: strengths.append("Danceability within competitive streaming range.")
    else: weaknesses.append("Danceability below common engagement range (0.60).")

    if loudness > -8: strengths.append("Loudness consistent with commercial production standards.")
    else: weaknesses.append("Loudness below competitive streaming levels (-8 dB threshold).")

    if followers > 50000: strengths.append("Strong baseline artist reach.")
    else: weaknesses.append("Limited artist reach may constrain exposure.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Strengths**")
        if strengths:
            for s in strengths:
                st.write("-", s)
        else:
            st.write("No structural strengths identified.")

    with colB:
        st.markdown("**Areas for Improvement**")
        if weaknesses:
            for w in weaknesses:
                st.write("-", w)
        else:
            st.write("No material weaknesses identified.")

    st.divider()

    # ---------------- Top Drivers (Regression) ----------------
    st.subheader("Top Model Drivers (Regression)")
    importances = safe_get_feature_importance(reg_pipeline)
    if importances is None:
        st.caption("Feature importance unavailable for this model type.")
    else:
        fi = pd.DataFrame({"feature": reg_expected, "importance": importances}).sort_values("importance", ascending=False)
        top_n = 10
        st.dataframe(fi.head(top_n), use_container_width=True, hide_index=True)
        st.bar_chart(fi.head(top_n).set_index("feature")["importance"])

    st.divider()

    # ---------------- Signal Strength ----------------
    st.subheader("Signal Strength")
    st.write("Breakout probability visualization based on model probability.")
    st.progress(breakout_prob)

else:
    hit_metric.metric("Hit Likelihood", "—")
    pop_metric.metric("Predicted Popularity (0–100)", "—")
    rec_box.info("Run the forecast to generate predictions and recommendations.")
