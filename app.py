import streamlit as st
import tempfile
import os
import re
import yt_dlp
from google.cloud import speech

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your-google-cloud-key.json'

def is_valid_youtube_url(url):
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+'
    )
    return youtube_regex.match(url) is not None

def extract_video_id(input_str):
    if is_valid_youtube_url(input_str):
        # Extract video ID from URL
        match = re.search(r'(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})', input_str)
        if match:
            return match.group(1)
        else:
            return None
    elif len(input_str) == 11:
        return input_str
    else:
        return None

def download_youtube_audio(video_id):
    """Download YouTube audio and convert to WAV"""
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': ['-ar', '16000', '-ac', '1'],
            'quiet': True,
            'no_warnings': True,
        }
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
            if duration > 600:
                return None, f"Video too long ({duration//60} min). Max: 10 min."
        
        downloaded_file = os.path.join(temp_dir, f"{video_id}.wav")
        if not os.path.exists(downloaded_file):
            return None, "Download failed. File not found."
        
        return downloaded_file, None
    except Exception as e:
        return None, f"Download error: {str(e)}"

def transcribe_audio(audio_path):
    client = speech.SpeechClient()
    with open(audio_path, 'rb') as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ' '.join([res.alternatives[0].transcript for res in response.results])
    return transcript

def detect_accent(transcript):
    if "colour" in transcript.lower():
        return "British (heuristic)"
    elif "gonna" in transcript.lower():
        return "American (heuristic)"
    return "Neutral/Unknown"

# Streamlit App
st.title("🎤 YouTube Accent Detector 🌐")
st.write("Enter a YouTube URL or Video ID to analyze its accent.")

user_input = st.text_input("YouTube Video URL or Video ID")
if st.button("Analyze"):
    video_id = extract_video_id(user_input)
    if not video_id:
        st.error("❌ Invalid YouTube URL or Video ID. Please check your input.")
    else:
        with st.spinner("Downloading audio..."):
            audio_path, error = download_youtube_audio(video_id)
            if error:
                st.error(f"❌ {error}")
            else:
                st.audio(audio_path)
                st.info("Transcribing...")
                transcript = transcribe_audio(audio_path)
                st.subheader("📝 Transcript:")
                st.write(transcript)
                accent = detect_accent(transcript)
                st.subheader("🌍 Detected Accent:")
                st.write(accent)

st.caption("Powered by yt-dlp and Google Cloud Speech-to-Text")
