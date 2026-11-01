import os
from langchain.agents import AgentState
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient
import numpy as np
from config.settings import Settings


import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

settings=Settings()

# Audio recording configuration
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1         # Mono
SILENCE_THRESHOLD = 500  # RMS amplitude threshold to detect silence
SILENCE_LIMIT = 5.0     # Stop recording after 5 seconds of continuous silence
AUDIO_FILE = "temp_input.wav"
EARBUD_DEVICE_INDEX = 22

# Initialize Gnani STT Client using SDK
client = GnaniSTTClient(api_key=os.getenv("GNANI_ACCESS_KEY"))


def record_voice(output_file: str = AUDIO_FILE,device_index: int = EARBUD_DEVICE_INDEX) -> str:
    """Records audio continuously until 5 seconds of silence is detected."""
    print("Listening... Speak now.")
    
    audio_frames = []
    silent_duration = 0.0
    block_duration = 0.2  # Process audio in 200ms blocks
    block_size = int(SAMPLE_RATE * block_duration)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", device=device_index) as stream:
        while True:
            data, _ = stream.read(block_size)
            audio_frames.append(data)
            
            # Calculate Root Mean Square (RMS) volume
            rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))

            if rms < SILENCE_THRESHOLD:
                silent_duration += block_duration
            else:
                silent_duration = 0.0  # Reset silence counter if user speaks

            if silent_duration >= SILENCE_LIMIT:
                print("5 seconds of silence detected. Stopping recording.")
                break

    recorded_audio = np.concatenate(audio_frames, axis=0)
    sf.write(output_file, recorded_audio, SAMPLE_RATE)
    print(f"Audio saved to: {output_file}")
    return {"audio_file_path": output_file}


def send_to_stt(audio_file_path: str, language_code: str = "hi-IN") -> str:
    """Sends recorded audio to Gnani STT using the Gnani Python SDK."""
    result = client.transcribe(audio_file_path, language_code=language_code)
    return result.get("transcript", "")


def translate_to_english(transcript: str, source_language: str) -> str:
    """Translates transcript from an Indic language (Hindi, Bengali, Gujarati,

    Kannada, Malayalam, Marathi, Punjabi, Tamil) to English using Groq LLM.
    """
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0.0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert translator. Translate the given text from {source_language} to clear, natural English. "
                "Output ONLY the translated text without explanations, greetings, or quotation marks.",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {"source_language": source_language, "text": transcript}
    )

    return response.content.strip()


def stt_node(state: AgentState) -> dict:
    transcript = send_to_stt(
        audio_file_path=state.get("input_audio_path", "temp_input.wav"),
        language_code=state.get("source_language", "hi-IN"),
    )
    return {"raw_transcript": transcript}

def translation_node(state: AgentState) -> dict:
    translated_transcript = translate_to_english(
        transcript=state.get("raw_transcript", ""),
        source_language=state.get("source_language", "hi-IN"),
    )
    return {"translated_transcript": translated_transcript}

def convert_transcript_to_perfect_query(state: AgentState) -> dict:
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=1.0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at converting natural language queries into precise search queries. "
                "Output ONLY the refined query without explanations, greetings, or quotation marks.",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {"text": state.get("translated_transcript", "")}
    )

    return {"question": response.content.strip()}

# if __name__ == "__main__":
#     audio_path = record_voice()
#     transcription = send_to_stt(audio_path, language_code="hi-IN")
#     print(f"Transcription: {transcription}")