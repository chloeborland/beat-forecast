import io
import json
import warnings
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
# Silence noisy (harmless) sklearn warning
# ============================================================
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but StandardScaler was fitted without feature names",
)

# ============================================================
# Paths + CSS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


def load_css(rel_path: str):
    css_path = BASE_DIR / rel_path
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css("assets/theme.css")

# Optional sidebar logo (won’t crash if missing)
logo_path = BASE_DIR / "assets" / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=220)
    st.sidebar.markdown("---")
else:
    st.sidebar.caption("Logo not found: assets/logo.png")

# Audio extraction (optional)
try:
    import librosa
except Exception:
    librosa = None

# ============================================================
# Model / asset paths
# ============================================================
MODELS_DIR = BASE_DIR / "models"

# Regression (popularity)
POP_MODEL_PATH = MODELS_DIR / "pop_rf_pipeline.joblib"

# Classification (hit likelihood)
# Prefer models/ path; fallback to repo root if needed
HIT_MODEL_PATH_PRIMARY = MODELS_DIR / "hit_gb_pipeline_wade_compat.joblib"
HIT_MODEL_PATH_FALLBACK = BASE_DIR / "hit_gb_pipeline_wade_compat.joblib"
HIT_MODEL_PATH = (
    HIT_MODEL_PATH_PRIMARY if HIT_MODEL_PATH_PRIMARY.exists() else HIT_MODEL_PATH_FALLBACK
)

# Clustering assets (repo root)
CLUSTER_SCALER_PATH = BASE_DIR / "cluster_scaler.pkl"
KMEANS_PATH = BASE_DIR / "kmeans_k6.pkl"
CLUSTER_FEATS_PATH = BASE_DIR / "cluster_feature_list.json"
CLUSTER_LABELS_PATH = BASE_DIR / "cluster_labels.csv"
CLUSTER_HIT_SUMMARY_PATH = BASE_DIR / "cluster_hit_summary.csv"
CLUSTER_TOP2024_PATH = BASE_DIR / "cluster_top2024_lift.csv"

# ============================================================
# Interpretation constants (NO model changes)
# ============================================================
BASE_HIT_RATE = 0.00224   # 0.224% baseline
DOC_THRESHOLD_T = 0.20    # report-aligned threshold


# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource
def load_regression_pipeline():
    return joblib.load(POP_MODEL_PATH.as_posix())


@st.cache_resource
def load_hit_pipeline():
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


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def estimate_artist_popularity(followers: float) -> float:
    """
    Estimate Spotify-style artist popularity (0-100) from follower count.
    This keeps the UI simple while providing the regression model with
    the artist popularity feature it appears to expect.

    Rough mapping:
    - 1,000 followers  -> ~10
    - 10,000 followers -> ~26
    - 1,000,000        -> ~58
    - 50,000,000       -> ~85
    """
    followers = max(float(followers), 1.0)
    popularity = np.interp(np.log10(followers), [3.0, 8.0], [10.0, 90.0])
    return float(np.clip(popularity, 0.0, 100.0))


def predict_archetype(user_inputs: dict):
    """
    Robust archetype lookup that won't crash if cluster_labels.csv has different column names.
    """
    scaler, kmeans, feats, labels, hit_summary, top2024 = load_clustering_assets()

    X = pd.DataFrame([{f: user_inputs.get(f) for f in feats}]).reindex(columns=feats)
    Xs = scaler.transform(X)
    cid = int(kmeans.predict(Xs)[0])

    cluster_col = pick_col(labels, ["cluster", "cluster_id", "clusterId", "k", "label"])
    if cluster_col is None:
        name = f"Cluster {cid}"
        desc = ""
        hs_cluster_col = pick_col(hit_summary, ["cluster", "cluster_id", "clusterId"])
        hs = hit_summary[hit_summary[hs_cluster_col] == cid].head(1) if hs_cluster_col else hit_summary.head(0)
        return cid, name, desc, hs, None

    lab = labels[labels[cluster_col] == cid].head(1)

    name_col = pick_col(labels, ["name", "archetype", "cluster_name", "label_name", "title"])
    desc_col = pick_col(labels, ["description", "desc", "summary", "details"])

    if len(lab):
        name = str(lab[name_col].iloc[0]) if name_col else f"Cluster {cid}"
        desc = str(lab[desc_col].iloc[0]) if desc_col else ""
    else:
        name = f"Cluster {cid}"
        desc = ""

    hs_cluster_col = pick_col(hit_summary, ["cluster", "cluster_id", "clusterId"])
    hs = hit_summary[hit_summary[hs_cluster_col] == cid].head(1) if hs_cluster_col else hit_summary.head(0)

    t24 = None
    if top2024 is not None:
        t24_name_col = pick_col(top2024, ["name", "archetype", "cluster_name", "label_name", "title"])
        if t24_name_col:
            t24 = top2024[top2024[t24_name_col] == name].head(1)

    return cid, name, desc, hs, t24


def build_aligned_input(row: pd.Series, expected_cols: list[str]) -> pd.DataFrame:
    """
    Align app inputs to whatever the model expects, without retraining.

    IMPORTANT FIX:
    - now injects estimated artist popularity from followers
    - supports raw + log variants for followers and popularity
    - keeps existing year/audio/genre mapping
    """
    X = pd.DataFrame([{c: 0 for c in expected_cols}])

    followers_val = row.get("followers")
    year_val = row.get("year")
    est_artist_popularity = row.get("estimated_artist_popularity")

    # ----------------------------
    # FOLLOWERS (raw + log variants)
    # ----------------------------
    if followers_val is not None:
        f = float(followers_val)

        follower_targets = [
            "total_artist_followers",
            "spotify_followers",
            "followers",
            "artist_followers",
            "artist_followers_total",
            "artist_followers_count",
            "artist_total_followers",
            "total_followers",
            "followers_total",
            "total_artist_followers_count",
            "total_followers_count",
            "artist_followers_total_count",
        ]
        for col in follower_targets:
            if col in expected_cols:
                X.loc[0, col] = f

        flog = float(np.log1p(max(f, 0.0)))
        follower_log_targets = [
            "log_followers",
            "log_artist_followers",
            "followers_log",
            "ln_followers",
            "log_total_artist_followers",
            "log_spotify_followers",
            "log_artist_followers_total",
            "log_followers_total",
            "artist_followers_log",
            "artist_followers_ln",
        ]
        for col in follower_log_targets:
            if col in expected_cols:
                X.loc[0, col] = flog

    # ----------------------------
    # ESTIMATED ARTIST POPULARITY
    # ----------------------------
    if est_artist_popularity is not None:
        ap = float(est_artist_popularity)

        popularity_targets = [
            "avg_artist_popularity",
            "artist_popularity",
            "spotify_artist_popularity",
            "avg_popularity",
            "mean_artist_popularity",
            "artist_popularity_avg",
            "average_artist_popularity",
        ]
        for col in popularity_targets:
            if col in expected_cols:
                X.loc[0, col] = ap

        popularity_log_targets = [
            "log_artist_popularity",
            "artist_popularity_log",
            "ln_artist_popularity",
            "log_avg_artist_popularity",
        ]
        aplog = float(np.log1p(max(ap, 0.0)))
        for col in popularity_log_targets:
            if col in expected_cols:
                X.loc[0, col] = aplog

    # ----------------------------
    # YEAR (raw + derived variants)
    # ----------------------------
    if year_val is not None:
        y = float(year_val)

        for col in ["year", "release_year"]:
            if col in expected_cols:
                X.loc[0, col] = y

        current_year = float(pd.Timestamp.now().year)
        age = max(0.0, current_year - y)

        for col in ["years_since_release", "song_age_years", "track_age_years", "age_years"]:
            if col in expected_cols:
                X.loc[0, col] = age

    # ----------------------------
    # AUDIO FEATURES
    # ----------------------------
    direct_map = {
        "danceability": ["danceability"],
        "energy": ["energy"],
        "loudness": ["loudness"],
        "tempo": ["tempo"],
        "valence": ["valence"],
        "speechiness": ["speechiness"],
        "acousticness": ["acousticness"],
        "instrumentalness": ["instrumentalness"],
        "liveness": ["liveness"],
        "duration_ms": ["duration_ms", "duration_ms_rounded", "duration", "track_duration_ms"],
        "key": ["key"],
        "mode": ["mode"],
    }

    for app_k, model_keys in direct_map.items():
        if row.get(app_k) is None:
            continue
        for mk in model_keys:
            if mk in expected_cols:
                X.loc[0, mk] = float(row[app_k])

    # ----------------------------
    # GENRE one-hot mapping
    # ----------------------------
    if "genre" in row.index and row.get("genre") is not None:
        g = str(row["genre"]).strip()
        possible = [
            f"genre_{g}",
            f"primary_genre_{g}",
            f"genre__{g}",
            f"genre_{g.lower()}",
            f"primary_genre_{g.lower()}",
            f"genre__{g.lower()}",
        ]
        for gc in possible:
            if gc in expected_cols:
                X.loc[0, gc] = 1

    return X[expected_cols]


def safe_predict_hit_probability(hit_pipeline, X_hit: pd.DataFrame) -> float:
    """
    Robustly compute p(hit)=P(y=1) using correct proba column based on classes_.
    """
    if hasattr(hit_pipeline, "predict_proba"):
        proba = hit_pipeline.predict_proba(X_hit)

        classes = getattr(hit_pipeline, "classes_", None)
        if classes is None and hasattr(hit_pipeline, "named_steps"):
            last = list(hit_pipeline.named_steps.values())[-1]
            classes = getattr(last, "classes_", None)

        if classes is not None:
            classes = list(classes)
            if 1 in classes:
                idx = classes.index(1)
                return float(proba[0, idx])

        return float(proba[0, 1])

    if hasattr(hit_pipeline, "decision_function"):
        score = float(hit_pipeline.decision_function(X_hit)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    pred = int(hit_pipeline.predict(X_hit)[0])
    return 1.0 if pred == 1 else 0.0


def hit_signal_tier(p_hit: float | None) -> tuple[str, str]:
    if p_hit is None:
        return "na", "—"
    if p_hit >= DOC_THRESHOLD_T:
        return "strong", "Strong"
    if p_hit >= 0.08:
        return "promising", "Promising"
    if p_hit >= 0.02:
        return "emerging", "Emerging"
    return "low", "Low"


def lift_vs_baseline(p_hit: float | None, baseline: float = BASE_HIT_RATE) -> float | None:
    if p_hit is None:
        return None
    b = float(baseline) if baseline and baseline > 0 else 1e-9
    return float(p_hit) / b


def recommendation_from_outputs(pred_pop: float | None, p_hit: float | None) -> tuple[str, str]:
    if pred_pop is None and p_hit is None:
        return "low", "Upload a song and click Run Forecast to generate results."

    tier_key, _ = hit_signal_tier(p_hit)
    lift = lift_vs_baseline(p_hit)

    if p_hit is not None and tier_key == "strong":
        return "good", "Strong breakout signal. Consider greenlighting release and promotion."

    if lift is not None and lift >= 10:
        return "good", "Above-average breakout signal versus baseline. Consider a focused release + targeted marketing test."

    if pred_pop is not None and float(pred_pop) >= 65:
        return "mid", "High expected popularity. Consider release with disciplined marketing and monitor early performance."

    if lift is not None and lift >= 4:
        return "mid", "Moderate breakout signal versus baseline. Consider refining and running a small-scale rollout."

    return "low", "Cautious outlook. Consider revising production/positioning before investing heavily."


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
        if np.isnan(maj_score):
            maj_score = -1e9
        if np.isnan(min_score):
            min_score = -1e9
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
    tempo_val = np.atleast_1d(tempo)[0]
    tempo = float(tempo_val) if np.isfinite(tempo_val) else 120.0
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
# Sidebar
# ============================================================
st.sidebar.markdown("## Inputs")
st.sidebar.caption("Upload a song, set artist context + genre, then run.")

uploaded_audio = st.sidebar.file_uploader(
    "Upload MP3/WAV", type=["mp3", "wav"], accept_multiple_files=False
)

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Spotify Followers", min_value=0, value=5000, step=100)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2024, step=1)

with st.sidebar.expander("Genre", expanded=True):
    genre = st.selectbox(
        "Primary Genre",
        ["Pop", "Rock", "Hip-Hop", "R&B", "Electronic", "Country", "Jazz", "Folk", "Classical"],
        index=0,
    )

estimated_artist_popularity = estimate_artist_popularity(followers)


st.sidebar.markdown("---")
run = st.sidebar.button("Run Forecast", type="primary", use_container_width=True)

# ============================================================
# Tabs
# ============================================================
tab_overview, tab_features, tab_compare = st.tabs(
    ["Overview", "Song Features", "Compare Scenarios"]
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
            "estimated_artist_popularity": estimated_artist_popularity,
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
p_hit = None
hit_lift = None
hit_tier_label = None

rec_bucket = None
rec_text = None

archetype_out = None
X_reg = None
X_hit = None

if run:
    if uploaded_audio is None or audio_feats is None:
        st.error("Upload a song (MP3/WAV) first — this version is song-driven.")
    else:
        missing = []

        if not POP_MODEL_PATH.exists():
            missing.append(POP_MODEL_PATH.as_posix())
        if not HIT_MODEL_PATH.exists():
            missing.append(HIT_MODEL_PATH.as_posix())

        for p in [
            CLUSTER_SCALER_PATH,
            KMEANS_PATH,
            CLUSTER_FEATS_PATH,
            CLUSTER_LABELS_PATH,
            CLUSTER_HIT_SUMMARY_PATH,
        ]:
            if not p.exists():
                missing.append(p.as_posix())

        if missing:
            st.error("Missing required files:\n- " + "\n- ".join(missing))
            st.stop()

        row = inputs.iloc[0]

        # Regression
        reg_pipeline = load_regression_pipeline()
        reg_expected = safe_get_feature_names(reg_pipeline)
        if reg_expected is None:
            st.error("Regression pipeline missing feature schema (feature_names_in_).")
            st.stop()

        X_reg = build_aligned_input(row, reg_expected)
        pred_pop = float(np.clip(float(reg_pipeline.predict(X_reg)[0]), 0.0, 100.0))

        # Classification
        hit_pipeline = load_hit_pipeline()
        hit_expected = safe_get_feature_names(hit_pipeline)
        if hit_expected is None:
            st.error("Hit pipeline missing feature schema (feature_names_in_).")
            st.stop()

        X_hit = build_aligned_input(row, hit_expected)
        p_hit = safe_predict_hit_probability(hit_pipeline, X_hit)
        hit_lift = lift_vs_baseline(p_hit)
        _, hit_tier_label = hit_signal_tier(p_hit)

        # Clustering
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
            archetype_out = None
            st.warning(f"Archetype unavailable: {e}")

        # Recommendation
        rec_bucket, rec_text = recommendation_from_outputs(pred_pop, p_hit)


# ============================================================
# OVERVIEW TAB
# ============================================================
with tab_overview:
    st.title("BeatForecast")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1.2])

    pop_val = "—" if pred_pop is None else f"{pred_pop:.1f}"
    hit_pct = "—" if p_hit is None else f"{100.0 * float(p_hit):.2f}%"
    lift_val = "—" if hit_lift is None else f"{hit_lift:.1f}×"
    baseline_txt = f"{100.0 * BASE_HIT_RATE:.3f}%"
    hit_signal = "—" if hit_tier_label is None else hit_tier_label

    with c1:
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

    with c2:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Hit Likelihood</div>
              <div class="card-value">{hit_pct}</div>
              <div class="card-sub">Model probability (hits are rare)</div>
              <div class="card-sub"><b>Baseline hit rate:</b> {baseline_txt}</div>
              <div class="card-sub"><b>Lift vs baseline:</b> {lift_val}</div>
              <div class="card-sub"><b>Hit Signal:</b> {hit_signal} (t={DOC_THRESHOLD_T:.2f})</div>
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
            - Estimates **artist popularity** from followers
            - Predicts **popularity** (regression)
            - Predicts **hit likelihood** (classification)
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
        if st.button("Save Current Scenario", use_container_width=True, disabled=(pred_pop is None and p_hit is None)):
            row2 = inputs.iloc[0].to_dict()
            row2["predicted_popularity"] = None if pred_pop is None else float(pred_pop)
            row2["hit_probability"] = None if p_hit is None else float(p_hit)
            row2["hit_lift_vs_baseline"] = None if hit_lift is None else float(hit_lift)
            row2["hit_signal"] = hit_tier_label
            st.session_state["scenario_bank"].append(row2)
            st.success("Scenario saved.")
    with s2:
        if st.button("Clear Saved Scenarios", use_container_width=True):
            st.session_state["scenario_bank"] = []
            st.success("Cleared.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.subheader("Song Archetype")
    if archetype_out is None:
        st.caption("Run forecast to see archetype.")
    else:
        cid, archetype, archetype_desc, hs, t24 = archetype_out
        st.write(f"**{archetype}**")
        if archetype_desc and archetype_desc != "nan":
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
            artist_context_view = inputs[["followers", "estimated_artist_popularity", "year", "genre"]].copy()
            artist_context_view.rename(
                columns={"estimated_artist_popularity": "artist_popularity_est"},
                inplace=True,
            )
            st.dataframe(artist_context_view, use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.subheader("Top Model Drivers (Regression)")
    if run and pred_pop is not None:
        reg_pipeline = load_regression_pipeline()
        reg_expected = safe_get_feature_names(reg_pipeline)
        importances = safe_get_feature_importance(reg_pipeline)
        if importances is None or reg_expected is None:
            st.caption("Feature importance unavailable for this model.")
        else:
            fi = (
                pd.DataFrame({"feature": reg_expected, "importance": importances})
                .sort_values("importance", ascending=False)
            )
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
        st.info("No scenarios saved yet. Run a forecast and click Save Current Scenario.")
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