"""Streamlit dashboard — Role 4's deliverable (plan §2).

Skeleton only: proves the container starts and can reach the API. The
explainability visuals (3-ROI waveforms, MAR/envelope overlay, confidence gauge)
are the actual product and are built out in Phase 1–2.

API_URL comes from the environment because the address differs by run mode:
  docker compose -> http://api:8000   (compose network)
  make dev       -> http://localhost:8000
Hardcoding either one breaks the other. See docker-compose.yml.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DeepGuard", page_icon="🔎", layout="wide")
st.title("DeepGuard — Multi-Modal Manipulation Detection")
st.caption("Upload-only. Cross-region pulse consistency + speech-to-lip alignment.")

with st.sidebar:
    st.subheader("Service")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API reachable at {API_URL}")
        st.json(health)
    except requests.RequestException as exc:
        st.error(f"API unreachable at {API_URL}")
        st.caption(str(exc))

uploaded = st.file_uploader(
    "Video file",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="4–60s, clear frontal face. Longer clips are truncated to the first 20s.",
)

if uploaded is not None:
    st.info(
        "Ingestion pipeline lands in Phase 1 (plan §5). "
        "This build confirms the UI, the API, and the contracts line up."
    )

st.divider()
st.caption(
    "A verdict of UNCERTAIN or INSUFFICIENT_EVIDENCE is the system working as "
    "designed, not a failure — see plan §6.4."
)
