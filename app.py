import streamlit as st
import yt_dlp
import tempfile
import whisper
import os

st.set_page_config(page_title="Accent Detector", page_icon="🎤")

st.title("🎤 Accent Detector - YouTube Audio Transcription")

youtube_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching"""
    return whisper.load_model("base")

def download_audio(youtube_url):
    """Download audio from YouTube video"""
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "audio")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file + '.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'extractaudio': True,
        'audioformat': 'wav',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        title = info.get('title', 'Unknown')
        duration = info.get('duration', 0)
        
        # Check if video is too long (over 20 minutes)
        if duration and duration > 1200:
            raise Exception(f"Video is too long ({duration//60} minutes). Please use videos under 20 minutes.")
        
        # Download the audio
        ydl.download([youtube_url])
    
    # Find the downloaded file
    for file in os.listdir(temp_dir):
        if file.startswith("audio"):
            return os.path.join(temp_dir, file), title
    
    raise Exception("Failed to download audio file")

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    model = load_whisper_model()
    result = model.transcribe(audio_path)
    return result

# Main interface
if youtube_url:
    if st.button("🎯 Detect Accent", type="primary"):
        try:
            # Validate URL
            if not any(domain in youtube_url.lower() for domain in ['youtube.com', 'youtu.be']):
                st.error("Please enter a valid YouTube URL")
                st.stop()
            
            # Download audio
            with st.spinner("📥 Downloading audio from YouTube..."):
                audio_path, title = download_audio(youtube_url)
                st.success(f"✅ Downloaded: {title}")
            
            # Transcribe
            with st.spinner("🎤 Transcribing audio (this may take a few minutes)..."):
                result = transcribe_audio(audio_path)
            
            # Display results
            st.success("🎉 Transcription completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌍 Detected Language")
                st.markdown(f"**{result['language'].upper()}**")
            
            with col2:
                st.subheader("📊 Confidence")
                # Calculate average confidence from segments if available
                if 'segments' in result:
                    avg_confidence = sum(seg.get('avg_logprob', 0) for seg in result['segments']) / len(result['segments'])
                    confidence_percent = max(0, min(100, (avg_confidence + 1) * 100))
                    st.markdown(f"**{confidence_percent:.1f}%**")
            
            st.subheader("📝 Full Transcription")
            st.text_area("Transcribed Text", result['text'], height=200)
            
            # Clean up
            try:
                os.remove(audio_path)
                os.rmdir(os.path.dirname(audio_path))
            except:
                pass
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            if "ffmpeg" in str(e).lower():
                st.info("💡 FFmpeg is required but not available. Please check deployment configuration.")
            elif "private" in str(e).lower() or "unavailable" in str(e).lower():
                st.info("💡 This video might be private or unavailable. Try a different video.")
            else:
                st.info("💡 Please try with a different YouTube video URL.")

else:
    st.info("👆 Enter a YouTube URL above to get started")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Paste YouTube URL** in the input field
    2. **Click 'Detect Accent'** to start
    3. **Wait for processing** (1-5 minutes)
    4. **View results** below
    
    **Tips:**
    - Use videos with clear speech
    - Shorter videos process faster
    - Works best with videos under 20 minutes
    """)
    
    st.header("⚠️ Limitations")
    st.markdown("""
    - Only public YouTube videos
    - Audio quality affects accuracy
    - Processing time varies by length
    - Some accents may not be detected accurately
    """)
    
    st.header("🛠️ Troubleshooting")
    st.markdown("""
    **Common issues:**
    - Private videos: Use public videos only
    - Long videos: Try shorter clips
    - Poor audio: Use videos with clear speech
    - Network issues: Check internet connection
    """)
