import streamlit as st
import yt_dlp
import tempfile
import subprocess
import whisper

# Load Whisper model once (this may take a while on first run)
model = whisper.load_model("base")

st.title("Accent Detector - YouTube Audio Transcription")

youtube_url = st.text_input("Enter YouTube Video URL")

def download_audio(youtube_url):
    # Temporary file for mp3
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
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Temporary file for wav output
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    # Convert mp3 to wav using ffmpeg command-line (ffmpeg must be available)
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_file.name, wav_file.name
    ], check=True)

    return wav_file.name

if st.button("Detect Accent"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL!")
    else:
        try:
            with st.spinner("Downloading and converting audio..."):
                wav_path = download_audio(youtube_url)
            
            with st.spinner("Transcribing audio with Whisper..."):
                result = model.transcribe(wav_path)
            
            st.subheader("Transcription:")
            st.write(result["text"])
            
            st.subheader("Detected Language:")
            st.write(result["language"])
        
        except Exception as e:
            st.error(f"Error: {e}")
