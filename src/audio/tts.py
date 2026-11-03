import io
import os
import time
from pathlib import Path
from typing import Optional
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from gnani.tts import GnaniTTSClient, AudioConfig
from config.settings import BASE_DIR, get_settings
from pipeline.state import AgentState

load_dotenv()
settings = get_settings()

OUTPUT_CACHE_DIR = BASE_DIR / "data" / "audio_cache" / "output"
OUTPUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Gnani TTS Client using official SDK
tts_client = GnaniTTSClient(api_key=os.getenv("GNANI_ACCESS_KEY"))

# Mapping language codes to recommended Timbre voices
DEFAULT_VOICES = {
    "hi-IN": "Nalini",
    "en-IN": "Pranav",
    "ta-IN": "Kaveri",
    "te-IN": "Suhana",
    "kn-IN": "Poorvi",
    "mr-IN": "Asmita",
    "bn-IN": "Nalini",
    "gu-IN": "Nalini",
    "pa-IN": "Nalini",
    "ml-IN": "Kaveri",
}


def play_audio(audio_bytes: bytes, device_index: Optional[int] = None) -> None:
    """Reads audio bytes directly in-memory and outputs playback via sounddevice."""
    with io.BytesIO(audio_bytes) as audio_io:
        data, sample_rate = sf.read(audio_io, dtype="float32")
    sd.play(data, samplerate=sample_rate, device=device_index)
    sd.wait()


def text_to_speech_gnani_sdk(
    text: str,
    language_code: str = "hi-IN",
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
    play_realtime: bool = True,
    device_index: Optional[int] = None,
) -> str:
    """
    Synthesizes speech using the Gnani Python SDK (GnaniTTSClient),
    saves the WAV output, and plays it aloud in real time.
    """
    if not text or not text.strip():
        return ""

    selected_voice = voice or DEFAULT_VOICES.get(language_code, "Nalini")

    if not output_path:
        filename = f"tts_{int(time.time())}.wav"
        output_path = str(OUTPUT_CACHE_DIR / filename)

    # Request uncompressed WAV via SDK
    audio_bytes = tts_client.synthesize(
        text=text,
        voice=selected_voice,
        model="timbre-v2.5",
        language=language_code,
        audio_config=AudioConfig(
            sample_rate=16000,
            encoding="linear_pcm",
            container="wav",
        ),
    )

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    if play_realtime:
        play_audio(audio_bytes, device_index=device_index)

    return output_path


def tts_node(state: AgentState, device_index: Optional[int] = None) -> dict:
    """
    LangGraph node that vocalizes the generated or summarized answer
    in the patient's original language.
    """
    text = state.get("translated_answer") or state.get("answer", "")
    lang = state.get("source_language", "hi-IN")

    output_path = text_to_speech_gnani_sdk(
        text=text,
        language_code=lang,
        device_index=device_index,
    )

    return {
        "output_audio_path": output_path,
        "trace": [*state.get("trace", []), f"Gnani TTS ({lang}) → Audio Played"],
    }