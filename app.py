import streamlit as st
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import av
import queue
import threading
import time

st.set_page_config(page_title="🎙️ Live Whisper Transcription", layout="centered")
st.title("🗣️ Real-Time Speech-to-Text using Whisper")

# ---------------- Load Whisper ----------------
@st.cache_resource
def load_model():
    return WhisperModel("small.en", device="cpu", compute_type="int8")

model = load_model()


# ---------------- Audio Processor ----------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = queue.Queue()
        self.transcript = ""
        self.running = True
        self.thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self.thread.start()

    def recv_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().astype(np.float32).flatten() / 32768.0
        self.audio_buffer.put(audio)
        return frame

    def _transcribe_loop(self):
        chunk_size = 16000 * 4  # 4 seconds
        audio_accum = np.array([], dtype=np.float32)

        while self.running:
            try:
                block = self.audio_buffer.get(timeout=1)
                audio_accum = np.concatenate((audio_accum, block))
                if len(audio_accum) >= chunk_size:
                    chunk = audio_accum[:chunk_size]
                    audio_accum = audio_accum[chunk_size:]
                    segments, _ = model.transcribe(chunk, beam_size=1)
                    text = " ".join([s.text for s in segments])
                    if text.strip():
                        self.transcript += " " + text
            except queue.Empty:
                continue

    def get_text(self):
        return self.transcript


# ---------------- Streamlit UI ----------------
ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

# Avoid running before context initializes
if ctx.state.playing and ctx.audio_processor:
    st.markdown("🎤 **Listening... start speaking!**")
    placeholder = st.empty()

    # Safe continuous update loop using Streamlit events
    while ctx.state.playing:
        time.sleep(1.5)
        transcript = ctx.audio_processor.get_text()
        placeholder.markdown(f"**Transcript:** {transcript}")
else:
    st.info("Click **Start** above and allow microphone access.")
