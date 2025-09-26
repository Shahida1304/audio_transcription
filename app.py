import streamlit as st
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# ---------------- Configuration ----------------
samplerate = 16000
block_duration = 0.5
chunk_duration = 3.0
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

stop_flag = threading.Event()

# ---------------- Load Whisper ----------------
model = WhisperModel(
    "medium.en", device="cpu", compute_type="int8"
)

def is_speech(audio_chunk, threshold=0.01):
    """Return True if chunk contains speech, False if mostly silence."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > threshold

# ---------------- Audio Callback ----------------
class AudioProcessor(AudioProcessorBase):
    def recv_audio_frame(self, frame):
        audio = frame.to_ndarray().astype(np.float32) / 32768.0
        audio_queue.put(audio[:, 0].copy())  # mono
        return frame

# ---------------- Recorder ----------------
def recorder():
    # Browser mic → queue (continuous)
    webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

# ---------------- Transcriber ----------------
def transcriber():
    global audio_buffer
    last_text = ""

    while not stop_flag.is_set():
        try:
            block = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)
            audio_chunk = audio_data[:frames_per_chunk]
            leftover = audio_data[frames_per_chunk:]
            audio_buffer = [leftover] if len(leftover) > 0 else []

            # convert to float32
            audio_chunk = audio_chunk.astype(np.float32)

            # Transcribe
            if is_speech(audio_chunk):
                segments, _ = model.transcribe(
                    audio_chunk, language="en", beam_size=3, temperature=0.0
                )
                for segment in segments:
                    if segment.text != last_text:
                        st.write(segment.text)
                        last_text = segment.text

            # Keep overlap buffer
            keep_frames = int(samplerate * 0.25)
            if len(audio_chunk) > keep_frames:
                audio_buffer = [audio_chunk[-keep_frames:]]
            else:
                audio_buffer = [audio_chunk]

# ---------------- Main ----------------
def main():
    st.title("🎙️ Live Speech-to-Text with Whisper")
    st.write("Speak into your mic and see live transcription below:")

    recorder()

    if st.button("Start Transcription"):
        stop_flag.clear()
        threading.Thread(target=transcriber, daemon=True).start()

    if st.button("Stop"):
        stop_flag.set()
        st.write("🛑 Transcription stopped.")

if __name__ == "__main__":
    main()
