import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import numpy as np
import os
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="🎙️ Whisper Audio Transcriber", layout="centered")
st.title("🎧 Record and Transcribe Audio (Cloud-Friendly)")

# ---------------- Load Whisper once ----------------
@st.cache_resource
def load_model():
    return WhisperModel("small", device="cpu", compute_type="int8")  # small is enough for CPU

model = load_model()

# ---------------- Step 1: Record Audio ----------------
st.subheader("Step 1: Record your voice")
audio_data = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
)

if audio_data:
    st.audio(audio_data['bytes'], format="audio/wav")
    
    # ---------------- Step 2: Save temporarily ----------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data['bytes'])
        temp_path = tmp_file.name

    # Optional: convert to mono 16kHz if needed
    sound = AudioSegment.from_file(temp_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(temp_path, format="wav")

    # ---------------- Step 3: Transcribe ----------------
    st.subheader("Step 2: Transcribe")
    if st.button("📝 Transcribe Now"):
        with st.spinner("Transcribing... ⏳"):
            segments, info = model.transcribe(temp_path)
            transcript = " ".join([seg.text for seg in segments])
            st.success("✅ Transcription Complete!")
            st.text_area("📝 Transcribed Text:", transcript, height=200)

        # Cleanup temp file
        os.remove(temp_path)
