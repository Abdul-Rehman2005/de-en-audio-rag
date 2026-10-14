import json
import requests
import streamlit as st

# FastAPI backend endpoint
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="German-English RAG Tutor",
    page_icon="🇩🇪",
    layout="wide"
)

st.title("🇩🇪 German-English Speech RAG Tutor 🇬🇧")
st.markdown("Practice speaking German or English, retrieve Wikipedia facts, and receive real-time grammar lessons.")

# -------------------------------------------------------------------
# Sidebar: Background Context Ingestion
# -------------------------------------------------------------------
with st.sidebar:
    st.header("1. Index Wikipedia Facts")
    st.markdown("Seed Qdrant with facts from Wikipedia before testing your voice.")

    topic = st.text_input("Wikipedia Topic", value="Deutsches Reich")
    language = st.selectbox("Wikipedia Language", options=["de", "en"], index=0)

    if st.button("Index Topic into Qdrant", use_container_width=True):
        with st.spinner("Fetching and vectorizing Wikipedia article..."):
            try:
                response = requests.post(
                    f"{API_URL}/index-wiki",
                    json={"topic": topic, "language": language}
                )
                if response.status_code == 201:
                    st.success(f"Successfully indexed '{topic}' into Qdrant!")
                else:
                    st.error(f"Error {response.status_code}: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}")

# -------------------------------------------------------------------
# Main Layout: Audio Input & RAG Lesson
# -------------------------------------------------------------------
st.header("2. Speak & Get Your Lesson")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Choose Input Method")
    input_method = st.radio("Audio Source", ["🎙️ Live Microphone", "📁 File Upload"], horizontal=True)

    audio_bytes = None
    file_name = "audio.wav"

    if input_method == "🎙️ Live Microphone":
        # Native browser microphone recorder widget
        recorded_audio = st.audio_input("Record your voice")
        if recorded_audio:
            audio_bytes = recorded_audio.getvalue()
            file_name = "live_recording.wav"
    else:
        uploaded_file = st.file_uploader(
            "Upload an audio file (.wav, .mp3, .m4a)",
            type=["wav", "mp3", "m4a", "ogg"]
        )
        if uploaded_file:
            audio_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

    optional_topic = st.text_input(
        "Optional Dynamic Topic (Indexes on the fly)",
        value="",
        placeholder="e.g., German grammar"
    )

    submit_button = st.button("Generate Lesson", type="primary", use_container_width=True)

with col2:
    st.subheader("Lesson Output")

    if submit_button:
        if not audio_bytes:
            st.warning("Please record live audio or upload an audio file first.")
        else:
            # Placeholders filled in once the SSE [DONE] sentinel arrives
            meta_placeholder = st.empty()
            context_placeholder = st.empty()

            st.markdown("### Grammar & Translation Lesson")

            try:
                files = {"file": (file_name, audio_bytes, "audio/wav")}
                data = {"wiki_topic": optional_topic} if optional_topic else {}

                # ── SSE streaming via /process-audio-stream ────────────────
                def _token_stream():
                    """Yield lesson tokens from the SSE stream; capture the
                    final metadata event and store it in session_state."""
                    with requests.post(
                        f"{API_URL}/process-audio-stream",
                        files=files,
                        data=data,
                        stream=True,
                        timeout=300,
                    ) as r:
                        if r.status_code != 200:
                            st.error(f"Error {r.status_code}: {r.text}")
                            return

                        for raw_line in r.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            if not line.startswith("data:"):
                                continue

                            payload = json.loads(line[len("data:"):].strip())

                            if "error" in payload:
                                st.error(payload["error"])
                                return

                            if payload.get("event") == "done":
                                # Store metadata for post-stream display
                                st.session_state["_rag_meta"] = payload
                                return

                            if "token" in payload:
                                yield payload["token"]

                # Stream tokens directly into the lesson area
                st.write_stream(_token_stream())

                # ── Post-stream: render metadata ───────────────────────────
                meta = st.session_state.pop("_rag_meta", None)
                if meta:
                    with meta_placeholder.container():
                        st.info(
                            f"**Transcription:** {meta.get('transcription')}\n\n"
                            f"**Detected Language:** `{meta.get('detected_language')}`"
                        )

                    with context_placeholder.container():
                        with st.expander("View Retrieved Vector Context (Qdrant)"):
                            for chunk in meta.get("retrieved_context", []):
                                score = chunk.get("score", 0)
                                topic_name = chunk.get("topic", "—")
                                url = chunk.get("url")
                                topic_display = (
                                    f"[{topic_name}]({url})" if url else topic_name
                                )
                                st.markdown(
                                    f"- **Score:** `{score:.3f}` | **Topic:** {topic_display}"
                                )
                                st.caption(chunk.get("text"))
                                st.markdown("---")

            except Exception as e:
                st.error(f"Failed to process audio: {e}")
