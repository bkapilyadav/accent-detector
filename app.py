import streamlit as st
import tempfile
import os
import requests
import yt_dlp
from google.cloud import speech

# Set Google Cloud credentials (make sure to upload key file to Streamlit and replace path)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your-google-cloud-key.json'

def download_youtube_audio(video_id):
    """Download YouTube audio using yt-dlp and convert to WAV format"""
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
            'postprocessor_args': [
                '-ar', '16000',  # Resample to 16 kHz
                '-ac', '1',      # Mono channel
            ],
            'quiet': True,
            'no_warnings': True,
        }
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
            if duration > 600:
                return None, f"Video too long ({duration//60} minutes). Limit: 10 minutes."
        
        downloaded_file = os.path.join(temp_dir, f"{video_id}.wav")
        if not os.path.exists(downloaded_file):
            return None, "Audio file not found after download."
        
        return downloaded_file, None
        
    except Exception as e:
        return None, f"Download error: {str(e)}"

def transcribe_audio(audio_path):
    """Transcribe audio using Google Speech-to-Text"""
    client = speech.SpeechClient()
    with open(audio_path, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="default"
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ' '.join([result.alternatives[0].transcript for result in response.results])
    return transcript

def detect_accent(transcript):
    """Very simple heuristic-based accent detection"""
    if "colour" in transcript.lower() or "favour" in transcript.lower():
        return "British English (heuristic)"
    elif "gonna" in transcript.lower() or "wanna" in transcript.lower():
        return "American English (heuristic)"
    else:
        return "Neutral/Unknown (needs better detection)"

# Streamlit UI
st.title("🎤 YouTube Accent Detector 🌐")
st.write("Enter a YouTube video ID to analyze its accent.")

video_id = st.text_input("YouTube Video ID (e.g., dQw4w9WgXcQ)")
if st.button("Analyze"):
    if not video_id:
        st.warning("Please enter a video ID.")
    else:
        with st.spinner("Downloading and processing audio..."):
            audio_file_path, error = download_youtube_audio(video_id)
            if error:
                st.error(f"❌ {error}")
            elif audio_file_path and os.path.exists(audio_file_path):
                st.audio(audio_file_path)
                st.info("Transcribing audio...")
                transcript = transcribe_audio(audio_file_path)
                if transcript:
                    st.subheader("📝 Transcript:")
                    st.write(transcript)
                    accent = detect_accent(transcript)
                    st.subheader("🌍 Detected Accent:")
                    st.write(accent)
                else:
                    st.error("❌ No transcription available.")
            else:
                st.error("❌ Audio file not found or failed to download.")

st.markdown("---")
st.caption("Powered by Google Speech-to-Text and yt-dlp")
