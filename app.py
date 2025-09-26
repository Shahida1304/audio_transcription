import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel


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
    "tiny.en", device="cpu", compute_type="int8"
)  # smaller model for CPU


def is_speech(audio_chunk, threshold=0.01):
    """Return True if chunk contains speech, False if mostly silence."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms > threshold


# ---------------- Audio Callback ----------------
def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ {status}")
    # convert block to 1D immediately
    audio_queue.put(indata[:, 0].copy())


# ---------------- Recorder ----------------
def recorder():
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=audio_callback,
        blocksize=frames_per_block,
    ):
        print("🎙️ Listening... Press Ctrl+C to stop.")
        while not stop_flag.is_set():
            sd.sleep(100)


# ---------------- Transcriber ----------------
def transcriber():
    global audio_buffer
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
                last_text = ""
                for segment in segments:
                    if segment.text != last_text:
                        print(segment.text, end=" ", flush=True)
                        last_text = segment.text
            else:
                pass

            keep_frames = int(samplerate * 0.25)
            if len(audio_chunk) > keep_frames:
                audio_buffer = [audio_chunk[-keep_frames:]]
            else:
                audio_buffer = [audio_chunk]


# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        threading.Thread(target=recorder, daemon=True).start()
        transcriber()
    except KeyboardInterrupt:
        stop_flag.set()
        print("\n🛑 Stopped by user.")
