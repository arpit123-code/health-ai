import io
import os
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import streamlit as st

# Add project root, src/, and data/ to sys.path so all internal imports resolve
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src", ROOT / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config.settings import get_settings
from audio.conversion import (
    record_voice,
    send_to_stt,
    translate_to_english,
    convert_transcript_to_perfect_query,
    EARBUD_DEVICE_INDEX,
)
from rag.workflow import ask

settings = get_settings()

# --- Configuration & Constants ---
TARGET_SR = 16000
AUDIO_CACHE_DIR = ROOT / "data" / "audio_cache" / "input"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = {
    "Hindi": "hi-IN",
    "English": "en-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Punjabi": "pa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
}

st.set_page_config(
    page_title=settings.app_name,
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Multilingual Voice Healthcare RAG")
st.caption(
    "Record your question in an Indic language or English. The pipeline transcribes via Gnani STT, "
    "translates to English, refines the query, and runs the LangGraph Agentic RAG workflow."
)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_language = st.selectbox(
        "Spoken / Input Language",
        options=list(LANGUAGES.keys()),
        index=0,
    )
    refine_query = st.toggle(
        "Refine Search Query",
        value=True,
        help="Uses Groq LLM to convert conversational speech into a precise retrieval query.",
    )
    device_idx = st.number_input(
        "Local Mic Device Index (for sounddevice)",
        min_value=0,
        value=EARBUD_DEVICE_INDEX,
        step=1,
    )

    st.divider()
    st.markdown(f"**Groq Model:** `{settings.groq_model}`")
    st.markdown(f"**Pinecone Index:** `{settings.pinecone_index_name}`")
    st.markdown(f"**Top-K Chunks:** `{settings.top_k}`")

    if st.button("🗑️ Clear Session History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current = None
        st.session_state.last_audio_hash = None
        st.rerun()

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "current" not in st.session_state:
    st.session_state.current = None
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


# --- Helper Functions ---
def save_browser_audio_as_16k_wav(audio_buffer, out_path: Path) -> Path:
    """Converts browser-recorded audio to 16kHz Mono PCM_16 WAV for Gnani STT."""
    if hasattr(audio_buffer, "seek"):
        audio_buffer.seek(0)

    data, sr = sf.read(audio_buffer, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)

    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, int(sr) // g).astype("float32")

    mono = np.clip(mono, -1.0, 1.0)
    sf.write(str(out_path), mono, TARGET_SR, subtype="PCM_16")
    return out_path


def execute_voice_pipeline(wav_path: Path, language_name: str, refine: bool) -> dict:
    """Runs STT -> Translation -> Query Refinement -> Agentic RAG."""
    lang_code = LANGUAGES[language_name]
    output = {"language": language_name, "audio_path": str(wav_path)}

    with st.status("Running Voice & Agentic RAG Pipeline...", expanded=True) as status:
        # 1. Speech-to-Text
        st.write(f"🎙️ Transcribing audio (`{lang_code}`) via Gnani STT...")
        transcript = send_to_stt(str(wav_path), language_code=lang_code)
        output["transcript"] = transcript

        if not transcript or not transcript.strip():
            status.update(label="No speech detected", state="error")
            output["error"] = "Gnani STT returned an empty transcript. Please speak clearly and try again."
            return output

        st.write(f"**Raw Transcript:** {transcript}")

        # 2. Translation to English
        if lang_code.startswith("en"):
            english_text = transcript
        else:
            st.write(f"🌐 Translating {language_name} → English...")
            english_text = translate_to_english(transcript, source_language=language_name)
            st.write(f"**Translated:** {english_text}")
        output["english"] = english_text

        # 3. Query Refinement
        if refine:
            st.write("✍️ Refining into a search query...")
            refined = convert_transcript_to_perfect_query(
                {"translated_transcript": english_text}
            )
            final_question = refined["question"]
            st.write(f"**Search Query:** `{final_question}`")
        else:
            final_question = english_text
        output["question"] = final_question

        # 4. LangGraph Agentic RAG Workflow
        st.write("🔎 Executing LangGraph workflow (`ask`)...")
        rag_result = ask(final_question)
        output["result"] = rag_result

        status.update(label="Pipeline Complete!", state="complete", expanded=False)

    return output


def render_result(data: dict):
    """Displays transcription, translation, RAG answer, citations, trace, and KB chunks."""
    if not data:
        return

    if data.get("error"):
        st.error(data["error"])
        return

    st.subheader("🗣️ Recognized Input")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Original ({data['language']})**")
        st.info(data.get("transcript", ""))
    with col2:
        st.markdown("**English Translation**")
        st.info(data.get("english", ""))

    if data.get("question") and data.get("question") != data.get("english"):
        st.caption(f"Refined Query sent to Agent: `{data['question']}`")

    result = data.get("result", {})
    st.subheader("💡 Assistant Response")
    st.markdown(result.get("answer", "No answer generated."))
    st.caption(f"Source Used: `{result.get('source_used', 'unknown').upper()}`")

    # Citations
    citations = result.get("citations", [])
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})", expanded=True):
            for c in citations:
                if c.get("url"):
                    st.markdown(f"- [{c.get('title')}]({c.get('url')}) `({c.get('type')})`")
                else:
                    st.markdown(f"- {c.get('title')} `({c.get('type')})`")

    # LangGraph Execution Trace
    trace = result.get("trace", [])
    if trace:
        with st.expander("🛠️ LangGraph Execution Trace", expanded=True):
            for idx, step in enumerate(trace, 1):
                st.markdown(f"`{idx}.` {step}")

    # Retrieved Private KB Documents
    kb_docs = result.get("kb_docs", [])
    if kb_docs:
        with st.expander(f"📄 Retrieved KB Chunks ({len(kb_docs)})"):
            for d in kb_docs:
                st.markdown(f"**Source:** `{d.metadata.get('source', 'unknown')}`")
                st.text(d.page_content)
                st.divider()


# --- Input Tabs ---
tab_browser, tab_local, tab_text = st.tabs([
    "🎙️ Browser Mic",
    "🎧 Local Device Mic (sounddevice)",
    "⌨️ Text Input (Skip STT)",
])

with tab_browser:
    st.markdown("### 🎙️ Voice Recording not works.")

    st.info("""
    1. Open your browser microphone and ensure microphone access is allowed.
    2. Click the microphone button to start recording.
    3. Allow microphone permission if your browser asks.
    4. Speak your health-related question clearly.
    5. Click stop when you finish recording.
    6. Your audio will automatically be sent for transcription.
    """)
    audio_input = st.audio_input(f"Record your question in {selected_language}")
    if audio_input is not None:
        audio_bytes = audio_input.getvalue()
        audio_hash = hash(audio_bytes)
        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            try:
                wav_file = save_browser_audio_as_16k_wav(
                    io.BytesIO(audio_bytes),
                    AUDIO_CACHE_DIR / f"browser_{int(time.time())}.wav",
                )
                st.session_state.current = execute_voice_pipeline(
                    wav_file, selected_language, refine_query
                )
                st.session_state.history.append(st.session_state.current)
            except Exception as e:
                st.session_state.current = {"error": f"{type(e).__name__}: {e}"}

with tab_local:
    st.info("""
    1. Ensure you have a working microphone connected to your local device.
    2. Enter the device index of your microphone (default is 0).""")
    st.caption(
        f"(Device Index `{device_idx}`, "
        "stops automatically after 5 seconds of silence)."
    )
    if st.button("🔴 Start Recording on Local Device"):
        out_wav = AUDIO_CACHE_DIR / f"local_{int(time.time())}.wav"
        try:
            with st.spinner("Listening... Speak now (stops after 5s of silence)..."):
                rec_res = record_voice(output_file=str(out_wav), device_index=int(device_idx))
                saved_path = Path(rec_res["audio_file_path"]) if isinstance(rec_res, dict) else Path(rec_res)

            st.audio(str(saved_path))
            st.session_state.current = execute_voice_pipeline(
                saved_path, selected_language, refine_query
            )
            st.session_state.history.append(st.session_state.current)
        except Exception as e:
            st.session_state.current = {"error": f"{type(e).__name__}: {e}"}

with tab_text:
    typed_query = st.text_input(f"Type your question in {selected_language}:")
    if st.button("Run RAG Pipeline") and typed_query.strip():
        try:
            with st.status("Running Translation & Agentic RAG...", expanded=True) as status:
                if selected_language == "English":
                    eng_text = typed_query
                else:
                    st.write(f"🌐 Translating {selected_language} → English...")
                    eng_text = translate_to_english(typed_query, source_language=selected_language)

                if refine_query:
                    st.write("✍️ Refining search query...")
                    q = convert_transcript_to_perfect_query(
                        {"translated_transcript": eng_text}
                    )["question"]
                else:
                    q = eng_text

                st.write("🔎 Running LangGraph workflow...")
                res = ask(q)
                status.update(label="Pipeline Complete!", state="complete", expanded=False)

            st.session_state.current = {
                "language": selected_language,
                "transcript": typed_query,
                "english": eng_text,
                "question": q,
                "result": res,
            }
            st.session_state.history.append(st.session_state.current)
        except Exception as e:
            st.session_state.current = {"error": f"{type(e).__name__}: {e}"}

# --- Display Active Output ---
if st.session_state.current:
    st.divider()
    render_result(st.session_state.current)

# --- Display Session History ---
if st.session_state.history:
    st.divider()
    with st.expander(f"🕒 Session History ({len(st.session_state.history)})"):
        for item in reversed(st.session_state.history):
            if item.get("error"):
                continue
            st.markdown(f"**Q ({item['language']}):** {item.get('transcript', '')}")
            st.markdown(f"**Refined Query:** `{item.get('question', '')}`")
            st.markdown(f"**A:** {item.get('result', {}).get('answer', '')[:300]}...")
            st.divider()