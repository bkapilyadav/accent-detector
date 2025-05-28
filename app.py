import streamlit as st
import yt_dlp
import tempfile
import whisper
import imageio_ffmpeg
import os

st.title("🎤 Accent Detector - YouTube Audio Transcription")

youtube_url = st.text_input("Enter YouTube Video URL")

def download_audio(youtube_url):
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    
    # Get ffmpeg and ffprobe paths using imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')

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
        # Explicit ffmpeg and ffprobe paths
        'ffmpeg_location': os.path.dirname(ffmpeg_path),
        'postprocessor_args': ['-hide_banner', '-loglevel', 'error'],
        'exec_cmd': f'"{ffmpeg_path}"',
    }

    # Set environment variables so yt_dlp finds ffmpeg and ffprobe
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

        except Exception as e:
            st.error(f"Error: {e}")
