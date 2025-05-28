import streamlit as st
import yt_dlp
import tempfile
import whisper
import os
import shutil

st.title("🎤 Accent Detector - YouTube Audio Transcription")

# Check if ffmpeg is available
def check_ffmpeg():
    return shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None

youtube_url = st.text_input("Enter YouTube Video URL")

def download_audio(youtube_url):
    """Download audio from YouTube video"""
    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_audio_file.name.replace('.wav', '.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Find the downloaded file
        base_name = temp_audio_file.name.replace('.wav', '')
        for ext in ['.wav', '.webm', '.m4a', '.mp3']:
            potential_file = base_name + ext
            if os.path.exists(potential_file):
                return potential_file
        
        return temp_audio_file.name
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        raise e

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        raise e

# Main app logic
if st.button("Detect Accent"):
    if not youtube_url:
        st.error("Please enter a valid YouTube URL!")
    elif not check_ffmpeg():
        st.error("FFmpeg is not installed on this system. Please check the deployment configuration.")
        st.info("Make sure 'packages.txt' file contains 'ffmpeg' in your repository.")
    else:
        try:
            with st.spinner("Downloading audio from YouTube..."):
                audio_path = download_audio(youtube_url)
            
            with st.spinner("Transcribing audio (this may take a few minutes)..."):
                transcription = transcribe_audio(audio_path)
            
            # Display results
            st.success("Transcription completed!")
            
            st.subheader("📝 Transcription:")
            st.write(transcription['text'])
            
            st.subheader("🌍 Detected Language:")
            st.write(f"**{transcription['language'].upper()}**")
            
            # Additional info if available
            if 'segments' in transcription and len(transcription['segments']) > 0:
                st.subheader("📊 Additional Information:")
                total_duration = transcription['segments'][-1]['end']
                st.write(f"**Duration:** {total_duration:.2f} seconds")
                st.write(f"**Segments:** {len(transcription['segments'])}")
            
            # Clean up temporary files
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try with a different YouTube URL or check if the video is accessible.")

# Add some helpful information
with st.expander("ℹ️ How to use"):
    st.write("""
    1. **Paste a YouTube URL** in the input field above
    2. **Click 'Detect Accent'** to start the process
    3. **Wait for processing** - this may take a few minutes depending on video length
    4. **View results** - you'll see the transcription and detected language
    
    **Note:** This app works best with videos that have clear speech. Very long videos may take more time to process.
    """)

with st.expander("🔧 Troubleshooting"):
    st.write("""
    If you encounter errors:
    - Make sure the YouTube URL is valid and publicly accessible
    - Try with shorter videos (under 10 minutes) for faster processing
    - Some videos may be restricted and cannot be downloaded
    - If ffmpeg errors persist, the deployment may need configuration updates
    """)
