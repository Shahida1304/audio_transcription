import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

class AudioProcessor(AudioProcessorBase):
    def recv_audio_frame(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        st.write(audio.shape)
        return frame

st.title("Mic Test")
webrtc_streamer(
    key="mic-test",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)
