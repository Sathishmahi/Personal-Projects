import streamlit as st
import tempfile
from asr_backend import ASR   # make sure your backend file is named asr_backend.py

# Page config
st.set_page_config(page_title="Speech to Text App", page_icon="🎤")

st.title("🎤 Speech to Text & Translator")
st.write("Upload an audio file and choose an action.")

# Initialize model (load once)
@st.cache_resource
def load_model():
    return ASR()

asr = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "flac", "ogg"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_audio_path = tmp_file.name

    st.markdown("### Choose an option:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Transcribe"):
            with st.spinner("Transcribing..."):
                result = asr.transcribe_text(temp_audio_path)
                st.success("Transcription Complete!")
                st.text_area("Transcribed Text", result, height=150)

    with col2:
        if st.button("🌍 Translate to English"):
            with st.spinner("Translating..."):
                result = asr.translate_to_english(temp_audio_path)
                st.success("Translation Complete!")
                st.text_area("Translated Text", result, height=150)
