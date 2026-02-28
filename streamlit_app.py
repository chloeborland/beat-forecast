

import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ============================================================
# Page config MUST be the first Streamlit call
# ============================================================
st.set_page_config(
    page_title="Beat Forecast",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Paths + CSS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

def load_css(rel_path: str):
    css_path = BASE_DIR / rel_path
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css("assets/theme.css")

# Optional sidebar logo (won’t crash if missing)
logo_path = BASE_DIR / "assets" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)
    st.sidebar.markdown("---")
else:
    st.sidebar.caption("Logo not found: assets/logo.png")

# Audio extraction (optional)
try:
    import librosa
except Exception:
    librosa = None


# ============================================================
# Model / asset paths (standardized)
# ============================================================
MODELS_DIR = BASE_DIR / "models"
POP_MODEL_PATH = MODELS_DIR / "pop_rf_pipeline.joblib"
HIT_MODEL_PATH = MODELS_DIR / "hit_gb_pipeline_wade.joblib"

CLUSTER_SCALER_PATH = BASE_DIR / "cluster_scaler.pkl"
KMEANS_PATH = BASE_DIR / "kmeans_k6.pkl"
CLUSTER_FEATS_PATH = BASE_DIR / "cluster_feature_list.json"
CLUSTER_LABELS_PATH = BASE_DIR / "cluster_labels.csv"
CLUSTER_HIT_SUMMARY_PATH = BASE_DIR / "cluster_hit_summary.csv"
CLUSTER_TOP2024_PATH = BASE_DIR / "cluster_top2024_lift.csv"


# ============================================================
# Cached loaders
# ============================================================
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

    labels = pd.read_csv(CLUSTER_LABELS_PATH.as_posix())
    hit_summary = pd.read_csv(CLUSTER_HIT_SUMMARY_PATH.as_posix())
    try:
        top2024 = pd.read_csv(CLUSTER_TOP2024_PATH.as_posix())
    except Exception:
        top2024 = None

    return scaler, kmeans, feats, labels, hit_summary, top2024


# ============================================================
# Utilities
# ============================================================
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

def build_regression_input(df_row: pd.Series, expected_cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame([{c: 0 for c in expected_cols}])

    # Artist context mapping
    if "total_artist_followers" in expected_cols and df_row.get("followers") is not None:
        X.loc[0, "total_artist_followers"] = float(df_row["followers"])
    if "avg_artist_popularity" in expected_cols and df_row.get("artist_popularity") is not None:
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
        if model_k in expected_cols and df_row.get(app_k) is not None:
            X.loc[0, model_k] = float(df_row[app_k])

    # Optional genre one-hot if the regression model expects it
    if "genre" in df_row.index:
        genre_col = f"genre_{df_row['genre']}"
        if genre_col in expected_cols:
            X.loc[0, genre_col] = 1

    return X[expected_cols]

def recommendation_from_models(breakout_prob: float, pred_pop: float) -> tuple[str, str]:
    if pred_pop >= 70:
        pop_band = "strong"
    elif pred_pop >= 45:
        pop_band = "moderate"
    else:
        pop_band = "low"

    if breakout_prob >= 0.40 and pred_pop >= 55:
        return "good", "Strong profile: high breakout probability + solid popularity forecast. Greenlight release + promotion."
    if breakout_prob >= 0.25 and pred_pop >= 45:
        return "mid", f"Promising: breakout signal moderate and popularity forecast is {pop_band}. Consider tweaks + stronger marketing plan."
    if pred_pop >= 60 and breakout_prob < 0.25:
        return "mid", "Popularity forecast is solid but breakout signal is weak. Consider collabs/playlist strategy to lift breakout odds."
    return "low", "Caution: projected breakout probability and popularity are low. Consider revising production/positioning/promo."

def score_badges(audio_feats: dict) -> list[str]:
    if audio_feats is None:
        return []
    badges = []
    if 100 <= audio_feats["tempo"] <= 140:
        badges.append("Tempo in pop range")
    if audio_feats["loudness"] >= -9:
        badges.append("Commercial loudness")
    if audio_feats["danceability"] >= 0.55:
        badges.append("Rhythm strong")
    if audio_feats["energy"] >= 0.60:
        badges.append("High energy")
    return badges[:4]


# ============================================================
# Audio feature extraction
# ============================================================
def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def normalize_minmax(x: float, xmin: float, xmax: float) -> float:
    if xmax <= xmin:
        return 0.0
    return clip01((x - xmin) / (xmax - xmin))

def seconds_to_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(round(seconds % 60))
    return f"{m}:{s:02d}"

def estimate_key_mode(chroma_mean: np.ndarray) -> tuple[int, int]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    chroma = chroma_mean / (np.sum(chroma_mean) + 1e-9)

    best_key, best_mode, best_score = 0, 1, -1e9
    for k in range(12):
        maj = np.roll(major_profile, k)
        minr = np.roll(minor_profile, k)
        maj_score = np.corrcoef(chroma, maj / maj.sum())[0, 1]
        min_score = np.corrcoef(chroma, minr / minr.sum())[0, 1]
        if np.isnan(maj_score): maj_score = -1e9
        if np.isnan(min_score): min_score = -1e9
        if maj_score > best_score:
            best_score = maj_score
            best_key = k
            best_mode = 1
        if min_score > best_score:
            best_score = min_score
            best_key = k
            best_mode = 0
    return int(best_key), int(best_mode)

def extract_audio_features(file_bytes: bytes) -> dict:
    if librosa is None:
        raise RuntimeError("librosa is not installed. Audio extraction unavailable.")

    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)

    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    duration_ms = int(round(duration_sec * 1000))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isfinite(tempo) else 120.0
    tempo = float(np.clip(tempo, 40.0, 220.0))

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    loudness_db = float(20.0 * np.log10(rms_mean + 1e-9))
    loudness_db = float(np.clip(loudness_db, -60.0, 0.0))

    energy = normalize_minmax(rms_mean, 0.01, 0.20)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(np.mean(onset_env)) if len(onset_env) else 0.0
    onset_std = float(np.std(onset_env)) if len(onset_env) else 0.0
    rhythm_strength = normalize_minmax(onset_mean, 0.10, 2.50)
    rhythm_stability = 1.0 - normalize_minmax(onset_std, 0.20, 2.00)
    danceability = clip01(rhythm_strength * rhythm_stability)

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))
    speechiness = normalize_minmax(zcr_mean, 0.02, 0.20)

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    acousticness = normalize_minmax(float(np.mean(flatness)), 0.01, 0.25)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    valence = normalize_minmax(centroid_mean, 800.0, 3500.0)

    y_harm, y_perc = librosa.effects.hpss(y)
    harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)[0]))
    perc_rms = float(np.mean(librosa.feature.rms(y=y_perc)[0]))
    harm_ratio = harm_rms / (harm_rms + perc_rms + 1e-9)
    instrumentalness = normalize_minmax(harm_ratio, 0.40, 0.90)

    liveness = normalize_minmax(float(np.std(rms)), 0.005, 0.08)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key, mode = estimate_key_mode(chroma_mean)

    return {
        "sr": int(sr),
        "duration_sec": duration_sec,
        "duration_ms": duration_ms,
        "tempo": tempo,
        "loudness": loudness_db,
        "energy": float(energy),
        "danceability": float(danceability),
        "valence": float(valence),
        "speechiness": float(speechiness),
        "acousticness": float(acousticness),
        "instrumentalness": float(instrumentalness),
        "liveness": float(liveness),
        "key": int(np.clip(key, 0, 11)),
        "mode": int(mode),
    }


# ============================================================
# State
# ============================================================
if "scenario_bank" not in st.session_state:
    st.session_state["scenario_bank"] = []


# ============================================================
# Sidebar (song-driven)
# ============================================================
st.sidebar.markdown("## Inputs")
st.sidebar.caption("Upload a song, set artist context + genre, then run.")

uploaded_audio = st.sidebar.file_uploader("Upload MP3/WAV", type=["mp3", "wav"], accept_multiple_files=False)

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Artist Followers", min_value=0, value=5000, step=100)
    artist_pop = st.slider("Artist Popularity (0–100)", 0, 100, 35)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2024, step=1)

with st.sidebar.expander("Genre", expanded=True):
    genre = st.selectbox(
        "Primary Genre",
        ["Pop", "Rock", "Hip-Hop", "R&B", "Electronic", "Country", "Jazz", "Folk", "Classical"],
        index=0,
    )

show_debug = st.sidebar.checkbox("Show debug (feature alignment)", value=False)

st.sidebar.markdown("---")
run = st.sidebar.button("Run Forecast", type="primary", use_container_width=True)


# ============================================================
# Tabs
# ============================================================
tab_overview, tab_features, tab_compare, tab_debug = st.tabs(
    ["Overview", "Song Features", "Compare Scenarios", "Debug"]
)


# ============================================================
# Extract features on upload
# ============================================================
audio_feats = None
audio_bytes = None
filename = None

if uploaded_audio is not None:
    filename = uploaded_audio.name
    if librosa is None:
        st.sidebar.error("librosa is not installed, so audio extraction can't run.")
    else:
        try:
            audio_bytes = uploaded_audio.read()
            audio_feats = extract_audio_features(audio_bytes)
            with st.sidebar.expander("Extracted (from upload)", expanded=False):
                st.write(
                    {
                        "file": filename,
                        "duration": seconds_to_mmss(audio_feats["duration_sec"]),
                        "tempo": round(audio_feats["tempo"], 1),
                        "loudness_db": round(audio_feats["loudness"], 2),
                        "energy": round(audio_feats["energy"], 3),
                        "danceability": round(audio_feats["danceability"], 3),
                    }
                )
        except Exception as e:
            st.sidebar.error(f"Audio extraction failed: {e}")
            audio_feats = None


# ============================================================
# Build inputs row
# ============================================================
inputs = pd.DataFrame(
    [
        {
            "file": filename,
            "followers": followers,
            "artist_popularity": artist_pop,
            "year": year,
            "genre": genre,
            "danceability": None if audio_feats is None else audio_feats["danceability"],
            "energy": None if audio_feats is None else audio_feats["energy"],
            "loudness": None if audio_feats is None else audio_feats["loudness"],
            "tempo": None if audio_feats is None else audio_feats["tempo"],
            "valence": None if audio_feats is None else audio_feats["valence"],
            "speechiness": None if audio_feats is None else audio_feats["speechiness"],
            "acousticness": None if audio_feats is None else audio_feats["acousticness"],
            "instrumentalness": None if audio_feats is None else audio_feats["instrumentalness"],
            "liveness": None if audio_feats is None else audio_feats["liveness"],
            "duration_ms": None if audio_feats is None else audio_feats["duration_ms"],
            "key": None if audio_feats is None else audio_feats["key"],
            "mode": None if audio_feats is None else audio_feats["mode"],
        }
    ]
)


# ============================================================
# Run models
# ============================================================
pred_pop = None
breakout_prob = None
rec_bucket = None
rec_text = None
archetype_out = None
X_reg = None
X_gb = None

if run:
    if uploaded_audio is None or audio_feats is None:
        st.error("Upload a song (MP3/WAV) first — this version is song-driven.")
    else:
        missing = []
        if not POP_MODEL_PATH.exists(): missing.append(POP_MODEL_PATH.as_posix())
        if not HIT_MODEL_PATH.exists(): missing.append(HIT_MODEL_PATH.as_posix())
        for p in [CLUSTER_SCALER_PATH, KMEANS_PATH, CLUSTER_FEATS_PATH, CLUSTER_LABELS_PATH, CLUSTER_HIT_SUMMARY_PATH]:
            if not p.exists():
                missing.append(p.as_posix())
        if missing:
            st.error("Missing required files:\n- " + "\n- ".join(missing))
            st.stop()

        row = inputs.iloc[0]

        # 1) Breakout prob (GB classifier)
        gb_model = load_gb_hit_model()
        if not hasattr(gb_model, "feature_names_in_"):
            st.error("GB model missing feature_names_in_. Re-train with pandas DataFrame to preserve schema.")
            st.stop()

        gb_expected = gb_model.feature_names_in_.tolist()
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

        # 2) Popularity (regression)
        reg_pipeline = load_regression_pipeline()
        reg_expected = safe_get_feature_names(reg_pipeline)
        if reg_expected is None:
            st.error("Regression pipeline missing feature schema (feature_names_in_).")
            st.stop()

        X_reg = build_regression_input(row, reg_expected)
        pred_val = float(reg_pipeline.predict(X_reg)[0])
        pred_pop = float(np.clip(pred_val, 0.0, 100.0))

        # 3) Archetype (clustering)
        cluster_inputs = {
            "danceability": row["danceability"],
            "energy": row["energy"],
            "valence": row["valence"],
            "tempo": row["tempo"],
            "loudness": row["loudness"],
            "acousticness": row["acousticness"],
            "speechiness": row["speechiness"],
            "instrumentalness": row["instrumentalness"],
            "liveness": row["liveness"],
            "duration_ms": row["duration_ms"],
        }
        try:
            archetype_out = predict_archetype(cluster_inputs)
        except Exception as e:
            archetype_out = ("—", "Archetype unavailable", str(e), pd.DataFrame(), None)

        # 4) Recommendation
        rec_bucket, rec_text = recommendation_from_models(breakout_prob, pred_pop)


# ============================================================
# OVERVIEW TAB
# ============================================================
with tab_overview:
    h1, h2 = st.columns([2.5, 1])
    with h1:
        st.title("BeatForecast")
        st.markdown('<div class="kicker">INSIGHTS THAT MOVE MUSIC FORWARD</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Song upload → audio features → models → recommendation</div>', unsafe_allow_html=True)
    with h2:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">Modules</div>
              <div class="card-sub">Clustering · Hit Prob · Popularity Forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1.2])

    hit_val = "—" if breakout_prob is None else f"{breakout_prob*100:.1f}%"
    pop_val = "—" if pred_pop is None else f"{pred_pop:.1f}"

    with c1:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Hit Likelihood (GB Model)</div>
              <div class="card-value">{hit_val}</div>
              <div class="card-sub">Gradient boosting classifier probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Predicted Popularity</div>
              <div class="card-value">{pop_val}</div>
              <div class="card-sub">Regression pipeline output (0–100)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        if rec_text is None:
            st.markdown(
                """
                <div class="rec">
                  <div class="rec-title">Executive Recommendation</div>
                  <p class="rec-text">Upload a song and click <b>Run Forecast</b> to generate results.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="rec {rec_bucket}">
                  <div class="rec-title">Executive Recommendation</div>
                  <p class="rec-text">{rec_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.8])
    with left:
        st.subheader("What this app does")
        st.markdown(
            """
            - Extracts **audio features** from your upload
            - Combines with **artist context** + **genre**
            - Predicts **hit likelihood** (GB classifier)
            - Predicts **popularity** (regression)
            - Assigns a **song archetype** (clustering)
            """.strip()
        )

    with right:
        st.subheader("Positive signals")
        if audio_feats is None:
            st.info("Upload a song to see signals.")
        else:
            badges = score_badges(audio_feats)
            if not badges:
                st.caption("No positive signals detected based on current extraction.")
            else:
                for b in badges:
                    st.markdown(f'<span class="chip">✓ {b}</span>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    s1, s2 = st.columns([1, 1])
    with s1:
        if st.button("Save Current Scenario", use_container_width=True, disabled=(pred_pop is None)):
            row2 = inputs.iloc[0].to_dict()
            row2["predicted_popularity"] = None if pred_pop is None else float(pred_pop)
            row2["hit_likelihood"] = None if breakout_prob is None else float(breakout_prob)
            st.session_state["scenario_bank"].append(row2)
            st.success("Scenario saved.")
    with s2:
        if st.button("Clear Saved Scenarios", use_container_width=True):
            st.session_state["scenario_bank"] = []
            st.success("Cleared.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Song Archetype (Clustering)")
    if archetype_out is None:
        st.caption("Run forecast to see archetype.")
    else:
        cid, archetype, archetype_desc, hs, t24 = archetype_out
        st.write(f"**{archetype}**")
        if archetype_desc:
            st.caption(archetype_desc)


# ============================================================
# FEATURES TAB
# ============================================================
with tab_features:
    st.subheader("Song Features (from upload)")
    if uploaded_audio is None:
        st.info("Upload a song to view extracted features.")
    elif audio_feats is None:
        st.warning("Audio uploaded, but extraction failed. Check librosa install or file type.")
    else:
        a, b = st.columns([1, 1])
        with a:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">File</div>
                  <div class="card-value" style="font-size:18px;">{filename}</div>
                  <div class="card-sub">Duration {seconds_to_mmss(audio_feats["duration_sec"])} · {audio_feats["duration_sec"]:.1f}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">Tempo</div>
                  <div class="card-value">{audio_feats["tempo"]:.1f} BPM</div>
                  <div class="card-sub">Beat tracking</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">Loudness</div>
                  <div class="card-value">{audio_feats["loudness"]:.1f} dB</div>
                  <div class="card-sub">RMS-based proxy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if audio_bytes is not None:
                st.write("")
                st.audio(audio_bytes)

        with b:
            st.markdown("### Model Inputs (audio-driven)")
            model_inputs_view = {
                "danceability": audio_feats["danceability"],
                "energy": audio_feats["energy"],
                "valence": audio_feats["valence"],
                "speechiness": audio_feats["speechiness"],
                "acousticness": audio_feats["acousticness"],
                "instrumentalness": audio_feats["instrumentalness"],
                "liveness": audio_feats["liveness"],
                "tempo": audio_feats["tempo"],
                "loudness": audio_feats["loudness"],
                "duration_ms": audio_feats["duration_ms"],
                "key": audio_feats["key"],
                "mode": audio_feats["mode"],
            }
            st.dataframe(pd.DataFrame([model_inputs_view]), use_container_width=True, hide_index=True)
            st.markdown("### Artist Context")
            st.dataframe(inputs[["followers", "artist_popularity", "year", "genre"]], use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Top Model Drivers (Regression)")
    if run and pred_pop is not None:
        reg_pipeline = load_regression_pipeline()
        reg_expected = safe_get_feature_names(reg_pipeline)
        importances = safe_get_feature_importance(reg_pipeline)
        if importances is None or reg_expected is None:
            st.caption("Feature importance unavailable for this model.")
        else:
            fi = pd.DataFrame({"feature": reg_expected, "importance": importances}).sort_values("importance", ascending=False)
            st.dataframe(fi.head(10), use_container_width=True, hide_index=True)
            st.bar_chart(fi.head(10).set_index("feature")["importance"])
    else:
        st.caption("Run forecast to see model drivers.")


# ============================================================
# COMPARE TAB
# ============================================================
with tab_compare:
    st.subheader("Compare scenarios")
    st.caption("Save multiple runs to compare songs/mixes.")

    bank = st.session_state["scenario_bank"]
    if not bank:
        st.info("No scenarios saved yet. Run a forecast and click **Save Current Scenario**.")
    else:
        df = pd.DataFrame(bank)
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Scenarios (CSV)",
            data=csv_bytes,
            file_name="beat_forecast_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# DEBUG TAB
# ============================================================
with tab_debug:
    st.subheader("Debug")
    st.caption("Shows extraction + model input rows used in prediction.")

    d1, d2 = st.columns([1, 1])
    with d1:
        st.markdown("### Extraction diagnostics")
        if audio_feats is None:
            st.info("Upload a song to populate diagnostics.")
        else:
            st.write(audio_feats)

    with d2:
        st.markdown("### Model input rows")
        if X_reg is None and X_gb is None:
            st.info("Run Forecast to generate model inputs.")
        else:
            if X_gb is not None:
                st.markdown("**GB input (X_gb)**")
                st.dataframe(X_gb, use_container_width=True, hide_index=True)
            if X_reg is not None:
                st.markdown("**Regression input (X_reg)**")
                st.dataframe(X_reg, use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Inputs used")
    st.dataframe(inputs, use_container_width=True, hide_index=True)