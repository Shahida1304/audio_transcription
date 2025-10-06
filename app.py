import streamlit as st
import whisper
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="🎙️ Whisper Audio Transcriber", layout="centered")
st.title("🎧 Live Audio Transcription (Record then Transcribe)")

# Load model from Hugging Face or local (whisper small)
@st.cache_resource
def load_model():
    return whisper.load_model("small")  # choose 'base', 'small', 'medium', etc.

model = load_model()

# Record audio
st.subheader("Step 1: Record your voice")
audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True,
)

if audio:
    st.audio(audio['bytes'], format="audio/wav")

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio['bytes'])
        temp_audio_path = temp_audio.name

    st.subheader("Step 2: Transcribe your audio")
    if st.button(" Transcribe Now"):
        with st.spinner("Transcribing... please wait ⏳"):
            result = model.transcribe(temp_audio_path)
            st.success(" Transcription Complete!")
            st.text_area("Transcribed Text:", result["text"], height=200)
        
        # cleanup
        os.remove(temp_audio_path)
