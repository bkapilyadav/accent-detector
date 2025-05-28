import streamlit as st
import yt_dlp
import tempfile
import whisper
import imageio_ffmpeg as ffmpeg
import os

st.title("🎤 Accent Detector - YouTube Audio Transcription")

# Optional: Use OpenAI Whisper API with your API key stored in Streamlit secrets
use_openai_api = st.checkbox("Use OpenAI Whisper API (needs API key in Streamlit secrets)")

if use_openai_api:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        st.error("Please add your OpenAI API key to Streamlit secrets as OPENAI_API_KEY")
        st.stop()

youtube_url = st.text_input("Enter YouTube Video URL")

def download_audio(youtube_url):
    audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': audio_file.name,
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': ffmpeg.get_ffmpeg_exe()
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return audio_file.name

def transcribe_local(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

def transcribe_openai(audio_path, api_key):
    import openai
    openai.api_key = api_key
    audio_file = open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript

if st.button("Detect Accent"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL!")
    else:
        try:
            with st.spinner("Downloading audio..."):
                mp3_path = download_audio(youtube_url)

            with st.spinner("Transcribing audio..."):
                if use_openai_api:
                    transcription = transcribe_openai(mp3_path, OPENAI_API_KEY)
                    text = transcription['text']
                else:
                    transcription = transcribe_local(mp3_path)
                    text = transcription['text']

            st.subheader("Transcription:")
            st.write(text)

            if not use_openai_api:
                st.subheader("Detected Language:")
                st.write(transcription['language'])

        except Exception as e:
            st.error(f"Error: {e}")
