import streamlit as st
import os
import tempfile
import yt_dlp
from moviepy.editor import AudioFileClip
import whisper

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙️ Accent Detector from YouTube Audio")

# Get OpenAI API key from environment variable set in Streamlit Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Please add your OPENAI_API_KEY in Streamlit Secrets to use this app.")

# Input: YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:")

def download_audio(youtube_url):
    # Download audio to a temp file using yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'outtmpl': 'audio.%(ext)s',
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        # Filename for audio
        audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        ydl_opts['outtmpl'] = audio_file.name
        ydl.download([youtube_url])
        return audio_file.name

if st.button("Detect Accent"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL!")
    else:
        with st.spinner("Downloading and processing audio..."):
            try:
                audio_path = download_audio(youtube_url)
            except Exception as e:
                st.error(f"Failed to download audio: {e}")
                st.stop()
            
            try:
                # Load audio with moviepy to convert or check format
                audio_clip = AudioFileClip(audio_path)
                # Whisper requires WAV or other audio, convert if necessary
                wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                audio_clip.write_audiofile(wav_path, logger=None)
                audio_clip.close()

                # Transcribe with Whisper
                result = model.transcribe(wav_path)

                text = result.get("text", "")
                st.subheader("🎧 Transcribed Text:")
                st.write(text)

                # For accent detection - placeholder logic (expand with your own model/logic)
                # Example: Just print language detected
                lang = result.get("language", "unknown")
                st.subheader("🌍 Detected Language:")
                st.write(lang)

                # You can integrate your custom accent detection here
                st.info("Accent detection logic to be implemented here.")

            except Exception as e:
                st.error(f"Error processing audio: {e}")

