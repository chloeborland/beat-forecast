import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Beat Forecast",
    layout="wide"
)

st.title("Beat Forecast")
st.caption("Data-driven decision support for estimating Spotify performance prior to release.")

# ---------------- Sidebar ----------------
st.sidebar.header("Model Inputs")

with st.sidebar.expander("Artist Context", expanded=True):
    followers = st.number_input("Artist Followers", min_value=0, value=5000, step=100)
    artist_pop = st.slider("Artist Popularity (0–100)", 0, 100, 35)

with st.sidebar.expander("Audio Features", expanded=True):
    danceability = st.slider("Danceability", 0.0, 1.0, 0.60)
    energy = st.slider("Energy", 0.0, 1.0, 0.70)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -7.0)
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.10)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.10)

st.sidebar.divider()
run = st.sidebar.button("Run Forecast", type="primary")

# ---------------- Main Layout ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Breakout Probability")
    st.metric("Hit Likelihood", "—")

with col2:
    st.subheader("Popularity Forecast")
    st.metric("Predicted Popularity (0–100)", "—")

with col3:
    st.subheader("Executive Recommendation")
    st.info("Model connection pending. Recommendation will appear here.")

st.divider()

st.subheader("Input Summary")

inputs = pd.DataFrame([{
    "followers": followers,
    "artist_popularity": artist_pop,
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "tempo": tempo,
    "valence": valence,
    "speechiness": speechiness,
    "acousticness": acousticness,
}])

st.dataframe(inputs, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Model Drivers")
st.write("Feature importance and directional guidance will appear here once models are connected.")

if run:

    # --- Simple Rule-Based Scoring (Temporary Logic) ---

    score = 0

    # Production thresholds (aligned with report findings)
    if energy > 0.65:
        score += 1
    if danceability > 0.60:
        score += 1
    if loudness > -8:
        score += 1

    # Artist leverage
    if followers > 50000:
        score += 1
    if artist_pop > 60:
        score += 1

    breakout_prob = min(score * 0.12, 0.60)  # cap at 60%
    predicted_popularity = 20 + (score * 8)

    # --- Update Metrics ---
    col1.metric("Hit Likelihood", f"{round(breakout_prob*100, 1)}%")
    col2.metric("Predicted Popularity (0–100)", round(predicted_popularity, 1))

    # --- Recommendation Logic ---
    if breakout_prob > 0.40:
        recommendation = "Release-ready under current production and exposure profile."
    elif breakout_prob > 0.25:
        recommendation = "Moderate breakout potential. Consider production refinements or stronger promotion."
    else:
        recommendation = "Low projected breakout probability under current inputs."

    col3.success(recommendation)


