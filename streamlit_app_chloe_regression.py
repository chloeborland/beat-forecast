import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Audio extraction
try:
    import librosa
except Exception:
    librosa = None


# ============================================================
# Page config + Green theme CSS
# ============================================================
st.set_page_config(
    page_title="Beat Forecast",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root{
        --green-900:#0b3d2e;
        --green-800:#0f5a3f;
        --green-700:#137a53;
        --green-100:#e7f5ee;
        --green-050:#f2fbf6;

        --text:#1f2328;
        --muted:rgba(31,35,40,0.65);
        --border:rgba(31,35,40,0.10);
        --shadow:0 10px 24px rgba(0,0,0,0.06);
      }

      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
      [data-testid="stSidebar"] { border-right: 1px solid var(--border); }

      h1, h2, h3 { letter-spacing: -0.02em; }
      .muted { color: var(--muted); }

      /* Primary button */
      div.stButton > button[kind="primary"]{
        background: linear-gradient(180deg, var(--green-700), var(--green-800));
        border: 1px solid rgba(19,122,83,0.35);
        color: white;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        box-shadow: var(--shadow);
      }
      div.stButton > button[kind="primary"]:hover{
        filter: brightness(1.02);
      }

      /* Cards */
      .card{
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        background: rgba(255,255,255,0.85);
        box-shadow: var(--shadow);
      }
      .card-title{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }
      .card-value{ font-size: 32px; font-weight: 800; line-height: 1.05; color: var(--text); }
      .card-sub{ font-size: 12px; color: rgba(31,35,40,0.55); margin-top: 6px; }

      /* Chips */
      .chip{
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: var(--green-100);
        border: 1px solid rgba(19,122,83,0.18);
        color: var(--green-900);
        font-weight: 750;
        font-size: 12px;
        margin: 0 8px 8px 0;
      }

      /* Recommendation */
      .rec{
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(19,122,83,0.18);
        box-shadow: var(--shadow);
      }
      .rec.good{ background: rgba(19,122,83,0.14); }
      .rec.mid{ background: rgba(242, 192, 86, 0.18); border-color: rgba(242,192,86,0.35); }
      .rec.low{ background: rgba(220, 53, 69, 0.10); border-color: rgba(220,53,69,0.20); }
      .rec-title{ font-size: 14px; font-weight: 850; margin-bottom: 6px; color: var(--text); }
      .rec-text{ margin: 0; color: rgba(31,35,40,0.78); }

      /* Section divider */
      .hr{ height: 1px; background: rgba(31,35,40,0.08); margin: 12px 0 14px 0; }

      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model loader
# ============================================================
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

MODEL_PATH = "models/pop_rf_pipeline.joblib"


# ============================================================
# Helpers + feature extraction
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

        # diagnostics
        "rms_mean": rms_mean,
        "onset_mean": onset_mean,
        "onset_std": onset_std,
        "centroid_mean": centroid_mean,
        "flatness_mean": float(np.mean(flatness)),
        "zcr_mean": zcr_mean,
    }

def build_model_input(feature_names, values: dict) -> pd.DataFrame:
    X = pd.DataFrame([{c: 0 for c in feature_names}], columns=feature_names)

    mapping = {
        "danceability": "danceability",
        "energy": "energy",
        "loudness": "loudness",
        "tempo": "tempo",
        "valence": "valence",
        "speechiness": "speechiness",
        "acousticness": "acousticness",
        "instrumentalness": "instrumentalness",
        "liveness": "liveness",
        "key": "key",
        "mode": "mode",
        "duration_ms": "duration_ms",
        "year": "year",
        "followers": "total_artist_followers",
        "artist_popularity": "avg_artist_popularity",
    }

    for src_k, model_k in mapping.items():
        if model_k in X.columns and values.get(src_k) is not None:
            X.loc[0, model_k] = float(values[src_k])

    genre = values.get("genre", "Pop")
    genre_col = f"genre_{genre}"
    if genre_col in X.columns:
        X.loc[0, genre_col] = 1.0

    return X

def recommendation_from_popularity(pred_pop: float) -> tuple[str, str]:
    if pred_pop >= 70:
        return "good", "High projected popularity. Strong release candidate with current profile."
    if pred_pop >= 50:
        return "mid", "Moderate projected popularity. Tighten production + boost promotion to improve odds."
    if pred_pop >= 30:
        return "mid", "Lower projected popularity. Consider adjusting energy/tempo mix or genre-positioning."
    return "low", "Very low projected popularity under current inputs. Treat as experimental or revise key drivers."

def hit_likelihood_from_popularity(pred_pop: float) -> float:
    """
    FIXED: Sigmoid mapping so it’s never “stuck” at 0%.
    - Popularity ~50 gives ~50% likelihood
    - Popularity ~30 gives low but non-zero
    - Popularity ~80+ approaches high likelihood
    """
    # Center and steepness (tune if desired)
    center = 50.0
    steep = 10.0
    p = 1.0 / (1.0 + np.exp(-(pred_pop - center) / steep))

    # Keep it readable (avoid exact 0 or 1)
    return float(np.clip(p, 0.01, 0.99))

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
# App state
# ============================================================
if "scenario_bank" not in st.session_state:
    st.session_state["scenario_bank"] = []


# ============================================================
# Header
# ============================================================
l, r = st.columns([2.4, 1])
with l:
    st.title("Beat Forecast")
    st.markdown('<span class="muted">Song-driven forecasting using your trained regression pipeline.</span>', unsafe_allow_html=True)
with r:
    st.markdown(
        """
        <div class="card">
          <div class="card-title">Theme</div>
          <div class="card-value" style="font-size:18px; color: var(--green-900);">Green Executive Dashboard</div>
          <div class="card-sub">Audio → features → model → recommendation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown("## Inputs")
st.sidebar.caption("Upload a song, set artist context, choose genre, then run the forecast.")

uploaded_audio = st.sidebar.file_uploader("Upload MP3/WAV", type=["mp3", "wav"], accept_multiple_files=False)

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Artist Followers", min_value=0, value=5000, step=100)
    artist_pop = st.slider("Artist Popularity (0–100)", 0, 100, 35)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2024, step=1)

with st.sidebar.expander("Genre", expanded=True):
    genre = st.selectbox("Primary Genre",
                         ["Pop", "Rock", "Hip-Hop", "R&B", "Electronic", "Country", "Jazz", "Folk", "Classical"],
                         index=0)

st.sidebar.markdown("---")
run = st.sidebar.button("Run Forecast", type="primary", use_container_width=True)


# ============================================================
# Tabs
# ============================================================
tab_overview, tab_features, tab_compare, tab_debug = st.tabs(
    ["Overview", "Song Features", "Compare Scenarios", "Debug"]
)

# ============================================================
# Extract features immediately (so UI updates on upload)
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
                st.write({
                    "file": filename,
                    "duration": seconds_to_mmss(audio_feats["duration_sec"]),
                    "tempo": round(audio_feats["tempo"], 1),
                    "loudness_db": round(audio_feats["loudness"], 2),
                    "energy": round(audio_feats["energy"], 3),
                    "danceability": round(audio_feats["danceability"], 3),
                })
        except Exception as e:
            st.sidebar.error(f"Audio extraction failed: {e}")
            audio_feats = None


# ============================================================
# Build inputs row (song-driven)
# ============================================================
inputs = pd.DataFrame([{
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
}])


# ============================================================
# Run model
# ============================================================
pred_pop = None
hit_like = None
rec_bucket = None
rec_text = None
X = None

if run:
    if uploaded_audio is None or audio_feats is None:
        st.error("Upload a song (MP3/WAV) first — the app is designed to be song-driven.")
    else:
        try:
            reg_pipeline = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Could not load model at {MODEL_PATH}: {e}")
            st.stop()

        if not hasattr(reg_pipeline, "feature_names_in_"):
            st.error(
                "Your saved pipeline does not expose feature_names_in_. "
                "Re-save the pipeline with a newer sklearn or store the feature list separately."
            )
            st.stop()

        feature_names = list(reg_pipeline.feature_names_in_)
        ui_vals = inputs.iloc[0].to_dict()
        X = build_model_input(feature_names, ui_vals)

        try:
            pred_val = float(reg_pipeline.predict(X)[0])
            pred_val = float(np.clip(pred_val, 0.0, 100.0))
            pred_pop = pred_val
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        hit_like = hit_likelihood_from_popularity(pred_pop)
        rec_bucket, rec_text = recommendation_from_popularity(pred_pop)


# ============================================================
# OVERVIEW TAB
# ============================================================
with tab_overview:
    c1, c2, c3 = st.columns([1, 1, 1.2])

    with c1:
        val = "—" if hit_like is None else f"{hit_like*100:.1f}%"
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Hit Likelihood</div>
              <div class="card-value">{val}</div>
              <div class="card-sub">Sigmoid mapping from model prediction (not stuck at 0%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        val = "—" if pred_pop is None else f"{pred_pop:.1f}"
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Predicted Popularity</div>
              <div class="card-value">{val}</div>
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
                  <p class="rec-text">Upload a song and click <b>Run Forecast</b> to generate a recommendation.</p>
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
            - Extracts **song-derived audio features** from your upload
            - Combines them with **artist context** + **genre**
            - Runs your trained **regression model** to predict popularity (0–100)
            - Converts that into a clear **decision-ready recommendation**
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
            row = inputs.iloc[0].to_dict()
            row["predicted_popularity"] = None if pred_pop is None else float(pred_pop)
            row["hit_likelihood"] = None if hit_like is None else float(hit_like)
            st.session_state["scenario_bank"].append(row)
            st.success("Scenario saved.")
    with s2:
        if st.button("Clear Saved Scenarios", use_container_width=True):
            st.session_state["scenario_bank"] = []
            st.success("Cleared.")


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
        order = [
            "file", "predicted_popularity", "hit_likelihood",
            "followers", "artist_popularity", "year", "genre",
            "tempo", "loudness", "energy", "danceability", "valence",
            "speechiness", "acousticness", "instrumentalness", "liveness",
            "duration_ms", "key", "mode"
        ]
        order = [c for c in order if c in df.columns]
        df = df[order + [c for c in df.columns if c not in order]]

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
    st.subheader("Debug / Proof")
    st.caption("This tab proves the app is using your uploaded song AND your regression model.")

    d1, d2 = st.columns([1, 1])

    with d1:
        st.markdown("### Extraction diagnostics")
        if audio_feats is None:
            st.info("Upload a song to populate diagnostics.")
        else:
            st.write({
                "rms_mean": audio_feats["rms_mean"],
                "onset_mean": audio_feats["onset_mean"],
                "onset_std": audio_feats["onset_std"],
                "centroid_mean": audio_feats["centroid_mean"],
                "flatness_mean": audio_feats["flatness_mean"],
                "zcr_mean": audio_feats["zcr_mean"],
            })

    with d2:
        st.markdown("### Model input row (X)")
        if X is None:
            st.info("Run Forecast to generate X.")
        else:
            st.dataframe(X, use_container_width=True, hide_index=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Inputs used for prediction")
    st.dataframe(inputs, use_container_width=True, hide_index=True)

    csv_bytes = inputs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Current Inputs (CSV)",
        data=csv_bytes,
        file_name="beat_forecast_inputs.csv",
        mime="text/csv",
        use_container_width=True,
    )