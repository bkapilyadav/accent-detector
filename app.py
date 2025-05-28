import streamlit as st
import yt_dlp
import tempfile
import whisper
import imageio_ffmpeg
import os
import subprocess
import sys

# Function to install ffmpeg if not available
def install_ffmpeg():
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            st.info("Installing ffmpeg... This may take a moment.")
            # Try to install via apt-get (for Linux systems like Streamlit Cloud)
            subprocess.run(['apt-get', 'update'], check=True, capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

st.title("🎤 Accent Detector - YouTube Audio Transcription")

# Install ffmpeg if needed
if not install_ffmpeg():
    st.error("Could not install ffmpeg. Please contact support.")
    st.stop()

youtube_url = st.text_input("Enter YouTube Video URL")

def download_audio(youtube_url):
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    
    # Get ffmpeg and ffprobe paths using imageio_ffmpeg as fallback
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
    except:
        # Use system ffmpeg if imageio_ffmpeg fails
        ffmpeg_path = 'ffmpeg'
        ffprobe_path = 'ffprobe'
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_audio_file.name,
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Try to set ffmpeg location if we have a specific path
        'postprocessor_args': ['-hide_banner', '-loglevel', 'error'],
    }
    
    # Only set ffmpeg_location if we have a specific path
    if ffmpeg_path != 'ffmpeg':
        ydl_opts['ffmpeg_location'] = os.path.dirname(ffmpeg_path)
        os.environ['FFMPEG_BINARY'] = ffmpeg_path
        os.environ['FFPROBE_BINARY'] = ffprobe_path
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return temp_audio_file.name

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

if st.button("Detect Accent"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL!")
    else:
        try:
            with st.spinner("Downloading audio..."):
                mp3_path = download_audio(youtube_url)
            
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(mp3_path)
            
            st.subheader("Transcription:")
            st.write(transcription['text'])
            
            st.subheader("Detected Language:")
            st.write(transcription['language'])
            
            # Clean up temporary file
            try:
                os.unlink(mp3_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("If this persists, ffmpeg may not be properly installed on the server.")
