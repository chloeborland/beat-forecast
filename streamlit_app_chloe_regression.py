import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- Page Config ----------------
st.set_page_config(page_title="Beat Forecast", layout="wide")

st.title("Beat Forecast")
st.caption("Data-driven decision support for estimating Spotify performance prior to release.")
st.divider()

# ---------------- Load Model (cached) ----------------
MODEL_PATH = Path("models/pop_rf_pipeline.joblib")

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

def safe_get_feature_names(pipeline):
    # Works if model was trained with a pandas DataFrame and sklearn stored feature_names_in_
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    # Try to reach inside pipeline steps (sometimes stored there)
    if hasattr(pipeline, "named_steps"):
        for step in pipeline.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def safe_get_feature_importance(pipeline):
    # Assumes pipeline has a "model" step that is a RandomForestRegressor (or similar)
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
    return None

# ---------------- Market Context ----------------
col_mc1, col_mc2 = st.columns([2, 1])
with col_mc1:
    st.subheader("Market Context")
    st.write(
        "Spotify breakout success is structurally rare. "
        "Only a small fraction of tracks achieve top-tier performance. "
        "This tool evaluates whether a song aligns with structural success patterns "
        "and predicts expected popularity prior to release."
    )
with col_mc2:
    st.metric("Observed Hit Rate (2024)", "0.22%")

st.divider()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Model Inputs")

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Artist Followers", min_value=0, value=5000, step=100, help="Total followers for the artist.")
    artist_pop = st.slider("Artist Popularity (0–100)", 0, 100, 35, help="Spotify-style popularity score for the artist.")

with st.sidebar.expander("Audio Features", expanded=True):
    danceability = st.slider("Danceability", 0.0, 1.0, 0.60)
    energy = st.slider("Energy", 0.0, 1.0, 0.70)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -7.0)
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.10)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.10)

with st.sidebar.expander("Defaults (needed by model)", expanded=False):
    # These are required by your model’s training schema
    year = st.slider("Release Year", 1950, 2026, 2024, help="If unknown, keep current year.")
    duration_ms = st.number_input("Duration (ms)", min_value=30_000, value=210_000, step=1_000, help="Typical song ~180k–240k ms.")
    key = st.selectbox("Key", list(range(12)), index=5, help="0=C, 1=C#, … 11=B (Spotify encoding).")
    mode = st.selectbox("Mode", [0, 1], index=1, help="0=minor, 1=major.")
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)

with st.sidebar.expander("Genre (one-hot)", expanded=True):
    primary_genre = st.selectbox(
        "Primary Genre",
        ["Classical", "Country", "Electronic", "Folk", "Hip-Hop", "Jazz", "Pop", "R&B", "Rock"],
        index=6
    )

show_debug = st.sidebar.checkbox("Show debug (feature alignment)", value=False)

st.sidebar.divider()
run = st.sidebar.button("Run Forecast", type="primary")

# ---------------- Build Input DataFrame (app-level) ----------------
# These are your app inputs (nice for display)
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
    "year": year,
    "duration_ms": duration_ms,
    "key": key,
    "mode": mode,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "primary_genre": primary_genre
}])

# ---------------- Main Layout (top KPIs) ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Breakout Probability")
    hit_likelihood_placeholder = st.empty()

with col2:
    st.subheader("Popularity Forecast")
    pop_forecast_placeholder = st.empty()

with col3:
    st.subheader("Executive Recommendation")
    rec_placeholder = st.empty()

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

# ---------------- Performance Drivers ----------------
st.subheader("Performance Drivers")
st.write("Model-based feature importance is shown after you run a forecast.")

# ---------------- Helper: Build model-ready X_input ----------------
def build_model_input(df_row: pd.Series, expected_cols: list[str]) -> pd.DataFrame:
    """
    Builds a 1-row dataframe that matches the model's expected feature columns.
    """
    # Start with all expected columns set to 0/None; fill numerical defaults
    X = pd.DataFrame([{c: 0 for c in expected_cols}])

    # Map app fields to model fields
    # Model expects: total_artist_followers, avg_artist_popularity
    if "total_artist_followers" in expected_cols:
        X.loc[0, "total_artist_followers"] = float(df_row["followers"])
    if "avg_artist_popularity" in expected_cols:
        X.loc[0, "avg_artist_popularity"] = float(df_row["artist_popularity"])

    # Direct audio features
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

    # Genre one-hot
    # expected: genre_Classical ... genre_Rock
    genre_col = f"genre_{df_row['primary_genre']}"
    if genre_col in expected_cols:
        X.loc[0, genre_col] = 1

    # Ensure column order exactly matches model expectation
    X = X[expected_cols]
    return X

# ---------------- Run Forecast ----------------
if run:
    # --- Breakout (rule-based, kept simple) ---
    score = 0
    if energy > 0.65: score += 1
    if danceability > 0.60: score += 1
    if loudness > -8: score += 1
    if followers > 50000: score += 1
    if artist_pop > 60: score += 1

    breakout_prob = min(score * 0.12, 0.60)  # cap at 60%
    hit_likelihood_placeholder.metric("Hit Likelihood", f"{round(breakout_prob * 100, 1)}%")

    # --- Regression prediction ---
    if not MODEL_PATH.exists():
        pop_forecast_placeholder.error(f"Missing model file: {MODEL_PATH.as_posix()}")
        rec_placeholder.info("Upload/commit the model artifact into /models and rerun.")
        st.stop()

    reg_pipeline = load_model(MODEL_PATH.as_posix())
    expected_cols = safe_get_feature_names(reg_pipeline)

    if expected_cols is None:
        pop_forecast_placeholder.error("Could not read model feature schema (feature_names_in_ missing).")
        rec_placeholder.info("Re-train using a pandas DataFrame so feature names are preserved.")
        st.stop()

    row = inputs_display.iloc[0]
    X_input = build_model_input(row, expected_cols)

    # Debug output (optional)
    if show_debug:
        st.divider()
        st.subheader("Debug: Feature Alignment")
        st.write("Model expects:", expected_cols)
        st.write("X_input columns:", list(X_input.columns))
        st.dataframe(X_input, use_container_width=True, hide_index=True)

    # Predict
    try:
        pred_pop = float(reg_pipeline.predict(X_input)[0])
        # Clamp just for display sanity
        pred_pop_display = max(0.0, min(100.0, pred_pop))
        pop_forecast_placeholder.metric("Predicted Popularity (0–100)", round(pred_pop_display, 1))
    except Exception as e:
        pop_forecast_placeholder.error(f"Prediction failed: {e}")
        rec_placeholder.info("Fix inputs / schema mismatch, then rerun.")
        st.stop()

    # --- Interpret popularity (human-friendly) ---
    if pred_pop_display >= 70:
        pop_band = "Strong"
    elif pred_pop_display >= 45:
        pop_band = "Moderate"
    else:
        pop_band = "Low"

    # --- Executive Recommendation (blends both signals) ---
    if breakout_prob >= 0.40 and pred_pop_display >= 55:
        recommendation = "Strong profile: high breakout signal + solid popularity forecast. Consider greenlighting release + promotion."
        rec_placeholder.success(recommendation)
    elif breakout_prob >= 0.25 and pred_pop_display >= 45:
        recommendation = f"Promising: breakout signal is moderate and popularity forecast is {pop_band.lower()}. Consider small production tweaks + stronger marketing plan."
        rec_placeholder.info(recommendation)
    elif pred_pop_display >= 60 and breakout_prob < 0.25:
        recommendation = "Popularity forecast is solid, but breakout signal is weak. Consider features/collabs/playlist strategy to improve breakout likelihood."
        rec_placeholder.info(recommendation)
    else:
        recommendation = "Caution: projected breakout probability and popularity are low under current inputs. Consider revising production, positioning, or promotion before release."
        rec_placeholder.warning(recommendation)

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

    # ---------------- Feature Importance ----------------
    st.subheader("Top Model Drivers (Regression)")
    importances = safe_get_feature_importance(reg_pipeline)

    if importances is None:
        st.caption("Feature importance unavailable for this model type.")
    else:
        fi = pd.DataFrame({
            "feature": expected_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)

        top_n = 10
        st.write(f"Top {top_n} features influencing the popularity prediction:")
        st.dataframe(fi.head(top_n), use_container_width=True, hide_index=True)

        # Streamlit will pick default styling; no custom colors needed
        st.bar_chart(fi.head(top_n).set_index("feature")["importance"])

    st.divider()

    # ---------------- Signal Strength ----------------
    st.subheader("Signal Strength")
    st.write("Breakout probability visualization based on structural alignment.")
    st.progress(breakout_prob)

else:
    # Before run: keep placeholders neutral
    hit_likelihood_placeholder.metric("Hit Likelihood", "—")
    pop_forecast_placeholder.metric("Predicted Popularity (0–100)", "—")
    rec_placeholder.info("Run the forecast to generate a recommendation and model drivers.")