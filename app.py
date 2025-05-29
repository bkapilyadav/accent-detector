import streamlit as st
import os
import yt_dlp
from tempfile import NamedTemporaryFile
import openai
import re

# Set your OpenAI API key (will be loaded at runtime on Streamlit Cloud)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Title
st.title("🎙️ Accent Detection Tool")
st.write("Upload a video/audio URL (e.g. MP4, Loom, direct link) to detect English accents and estimate confidence.")

# Input URL
video_url = st.text_input("🔗 Enter public video/audio URL:")

# Helper to download audio
def download_audio(url):
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': tmp_file.name,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return tmp_file.name, None
    except Exception as e:
        return None, f"Download error: {e}"

# Transcription
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript["text"]

# Accent Classification (very simple placeholder)
def classify_accent(transcription):
    transcription = transcription.lower()
    # Placeholder logic
    if re.search(r"\bmate\b|\bg'day\b|\baussie\b", transcription):
        return "Australian", 90
    elif re.search(r"\blorry\b|\bqueue\b|\bflat\b", transcription):
        return "British", 85
    elif re.search(r"\bcolor\b|\btruck\b|\bsidewalk\b", transcription):
        return "American", 88
    else:
        return "Unknown English Accent", 60

if st.button("Analyze Accent") and video_url:
    with st.spinner("Downloading and analyzing..."):
        audio_path, error = download_audio(video_url)
        if error:
            st.error(error)
        else:
            st.audio(audio_path)  # Playback
            transcript = transcribe_audio(audio_path)
            st.subheader("Transcription:")
            st.write(transcript)

            accent, confidence = classify_accent(transcript)
            st.subheader("Accent Classification:")
            st.write(f"🎤 Accent Detected: **{accent}**")
            st.write(f"🔍 Confidence: **{confidence}%**")
            st.write("This is a prototype tool using Whisper transcription and basic keyword detection.")

            # Cleanup
            os.remove(audio_path)
