
import html
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Glassmorphism Custom CSS ---
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Deep Mesh Gradient Background for Glass Contrast */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(16, 185, 129, 0.16) 0%, transparent 45%),
                radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.18) 0%, transparent 45%),
                radial-gradient(circle at 50% 85%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
                linear-gradient(135deg, #070b14 0%, #0f172a 50%, #090d16 100%);
    background-attachment: fixed;
    color: #f8fafc;
}

/* Hide default Streamlit top bar background */
header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1120px !important;
}

/* Frosted Glass Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.45) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Reusable Glass Card */
.glass-card {
    background: rgba(255, 255, 255, 0.045);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
    margin-bottom: 1.25rem;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}

.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 12px 40px 0 rgba(14, 165, 233, 0.12);
}

/* Hero Banner */
.hero-glass {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 24px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff 0%, #7dd3fc 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.4rem 0 0.6rem 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 780px;
    margin: 0;
}

/* Glass Pill Badges */
.glass-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #7dd3fc;
}

.glass-badge-emerald {
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #6ee7b7;
}

/* Sidebar Spec Pills */
.spec-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
}
.spec-label { color: #94a3b8; }
.spec-value {
    color: #38bdf8;
    font-family: monospace;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.1);
    padding: 2px 8px;
    border-radius: 6px;
}

/* Tabs Styling */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.25rem !important;
    margin-right: 0.5rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(16, 185, 129, 0.15)) !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2) !important;
}

/* Inputs & Audio Widget Glass Styling */
div[data-testid="stAudioInput"],
div[data-testid="stTextInput"] > div > div,
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] > div > div {
    background: rgba(15, 23, 42, 0.55) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
    color: #f8fafc !important;
}

/* Glowing Glass Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(16, 185, 129, 0.25) 100%) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    border-color: #38bdf8 !important;
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3) !important;
}

/* Glass Expanders */
details[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    overflow: hidden;
    margin-bottom: 0.85rem;
}

/* Step Guidance List */
.step-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0 1.25rem 0;
}
.step-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-size: 0.86rem;
    color: #cbd5e1;
}
.step-num {
    color: #38bdf8;
    font-weight: 700;
    margin-right: 6px;
}

.card-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.45rem;
    font-weight: 600;
}

.card-body-text {
    font-size: 1.02rem;
    color: #f1f5f9;
    line-height: 1.6;
}
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-glass">
        <span class="glass-badge">✨ Agentic Healthcare Intelligence</span>
        <h1 class="hero-title">🩺 Multilingual Voice Healthcare RAG</h1>
        <p class="hero-subtitle">
            Speak naturally in an Indic language or English. Your voice streams through 
            <strong>Gnani STT</strong>, neural translation, and <strong>Groq LLM query refinement</strong> 
            before executing a multi-step <strong>LangGraph Agentic RAG</strong> workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
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
        "Local Mic Device Index (sounddevice)",
        min_value=0,
        value=EARBUD_DEVICE_INDEX,
        step=1,
    )

    st.markdown("---")
    st.markdown("### 🧬 System Telemetry")
    st.markdown(
        f"""
        <div class="spec-pill">
            <span class="spec-label">Groq Model</span>
            <span class="spec-value">{html.escape(str(settings.groq_model))}</span>
        </div>
        <div class="spec-pill">
            <span class="spec-label">Pinecone Index</span>
            <span class="spec-value">{html.escape(str(settings.pinecone_index_name))}</span>
        </div>
        <div class="spec-pill">
            <span class="spec-label">Top-K Chunks</span>
            <span class="spec-value">{html.escape(str(settings.top_k))}</span>
        </div>
        <div class="spec-pill">
            <span class="spec-label">Audio Sample Rate</span>
            <span class="spec-value">16 kHz PCM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
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
    """Displays transcription, translation, RAG answer, citations, trace, and KB chunks in glass panels."""
    if not data:
        return

    if data.get("error"):
        st.error(data["error"])
        return

    st.markdown("### 🗣️ Recognized Voice & Query Transformation")
    col1, col2 = st.columns(2)

    raw_transcript = html.escape(str(data.get("transcript", "")))
    eng_translation = html.escape(str(data.get("english", "")))
    lang_label = html.escape(str(data.get("language", "")))

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">🎙️ Original Input ({lang_label})</div>
                <div class="card-body-text">{raw_transcript}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-label">🌐 English Translation</div>
                <div class="card-body-text">{eng_translation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if data.get("question") and data.get("question") != data.get("english"):
        refined_q = html.escape(str(data["question"]))
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 1rem 1.4rem; border-color: rgba(56, 189, 248, 0.3);">
                <span class="glass-badge" style="margin-right: 10px;">🔎 Refined Retrieval Query</span>
                <span style="color: #e2e8f0; font-family: monospace; font-size: 0.95rem;">{refined_q}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    result = data.get("result", {})
    source_used = html.escape(str(result.get("source_used", "unknown")).upper())
    answer_text = result.get("answer", "No answer generated.")

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin: 1.2rem 0 0.6rem 0;">
            <h3 style="margin: 0;">💡 Clinical Assistant Synthesis</h3>
            <span class="glass-badge glass-badge-emerald">Source: {source_used}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render answer inside a styled container
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(answer_text)
        st.markdown("</div>", unsafe_allow_html=True)

    # Citations, Execution Trace, and KB Docs in clean columns/expanders
    citations = result.get("citations", [])
    trace = result.get("trace", [])
    kb_docs = result.get("kb_docs", [])

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        if citations:
            with st.expander(f"📚 Verified Citations ({len(citations)})", expanded=True):
                for c in citations:
                    if c.get("url"):
                        st.markdown(f"- [{c.get('title')}]({c.get('url')}) `({c.get('type')})`")
                    else:
                        st.markdown(f"- {c.get('title')} `({c.get('type')})`")
    with c_col2:
        if trace:
            with st.expander("🛠️ LangGraph Execution Trace", expanded=True):
                for idx, step in enumerate(trace, 1):
                    st.markdown(f"`{idx}.` {step}")

    if kb_docs:
        with st.expander(f"📄 Retrieved Knowledge Base Chunks ({len(kb_docs)})"):
            for d in kb_docs:
                st.markdown(f"**Source:** `{d.metadata.get('source', 'unknown')}`")
                st.text(d.page_content)
                st.divider()


# --- Input Tabs ---
tab_browser, tab_local, tab_text = st.tabs([
    "🎙️ Browser Microphone",
    "🎧 Local Device Mic (sounddevice)",
    "⌨️ Direct Text Input",
])

with tab_browser:
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; font-size: 1.05rem; color: #f8fafc;">Browser Voice Capture</span>
                <span class="glass-badge">Active Language: {html.escape(selected_language)}</span>
            </div>
            <div class="step-grid">
                <div class="step-item"><span class="step-num">01</span>Allow browser mic permissions</div>
                <div class="step-item"><span class="step-num">02</span>Click the mic icon & speak clearly</div>
                <div class="step-item"><span class="step-num">03</span>Stop recording to auto-trigger RAG</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;">
                <span style="font-weight: 600; font-size: 1.05rem; color: #f8fafc;">Local Hardware Stream (sounddevice)</span>
                <span class="glass-badge">Device Index: {device_idx}</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
                Captures audio directly from your host machine's microphone and automatically stops after <strong>5 seconds of silence</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
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
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.4rem;">
                <span style="font-weight: 600; font-size: 1.05rem; color: #f8fafc;">Text-Only Pipeline Bypass</span>
                <span class="glass-badge">Skips Gnani STT</span>
            </div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">
                Type your query in <strong>{html.escape(selected_language)}</strong> to test translation, query refinement, and LangGraph retrieval directly.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    typed_query = st.text_input(
        f"Type your question in {selected_language}:",
        placeholder="e.g., मुझे पिछले तीन दिनों से बुखार और सिरदर्द है, मुझे क्या करना चाहिए?",
    )
    if st.button("🚀 Run Agentic RAG Pipeline") and typed_query.strip():
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
    st.markdown("<br>", unsafe_allow_html=True)
    render_result(st.session_state.current)

# --- Display Session History ---
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(f"🕒 Session History ({len(st.session_state.history)} queries)"):
        for item in reversed(st.session_state.history):
            if item.get("error"):
                continue
            q_lang = html.escape(str(item.get("language", "")))
            q_raw = html.escape(str(item.get("transcript", "")))
            q_ref = html.escape(str(item.get("question", "")))
            ans_preview = html.escape(str(item.get("result", {}).get("answer", ""))[:320])
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 1.1rem; margin-bottom: 0.8rem;">
                    <div style="display:flex; justify-content:space-between; margin-bottom: 0.4rem;">
                        <span class="glass-badge">{q_lang}</span>
                        <span style="color:#38bdf8; font-family:monospace; font-size:0.82rem;">Refined: {q_ref}</span>
                    </div>
                    <div style="font-weight:600; color:#f8fafc; margin-bottom:0.35rem;">Q: {q_raw}</div>
                    <div style="color:#94a3b8; font-size:0.9rem;">{ans_preview}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )