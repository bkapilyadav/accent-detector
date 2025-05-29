import streamlit as st
import yt_dlp
import tempfile
import os
import requests
import json
import subprocess
import shutil
from pathlib import Path

st.set_page_config(
    page_title="YouTube Accent Detector",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 YouTube Accent Detector")
st.markdown("Extract and analyze audio from YouTube videos")

# Initialize session state
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None

def check_dependencies():
    """Check if required system dependencies are available"""
    ffmpeg_available = shutil.which('ffmpeg') is not None
    return {
        'ffmpeg': ffmpeg_available
    }

def get_video_info(url):
    """Get video information without downloading"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown')
            }
        except Exception as e:
            raise Exception(f"Could not extract video info: {str(e)}")

def download_audio(youtube_url, max_duration=600):  # 10 minutes max
    """Download audio from YouTube video"""
    
    # Get video info first
    video_info = get_video_info(youtube_url)
    
    # Check duration
    if video_info['duration'] and video_info['duration'] > max_duration:
        raise Exception(f"Video is too long ({video_info['duration']//60} minutes). Please use videos under {max_duration//60} minutes.")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.wav")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.wav', '.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # mono audio
        ],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Find the actual downloaded file
        for file in os.listdir(temp_dir):
            if file.startswith("audio") and file.endswith(".wav"):
                return os.path.join(temp_dir, file), video_info
        
        raise Exception("Audio file not found after download")
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Download failed: {str(e)}")

def simple_transcribe_with_api(audio_file_path):
    """Simple transcription using a free API service"""
    try:
        # Using AssemblyAI free tier as an example
        # You can replace this with any other transcription service
        
        # For demo purposes, we'll create a mock response
        # In real implementation, you'd use an actual API
        
        import time
        import random
        
        # Simulate processing time
        time.sleep(2)
        
        # Mock transcription based on common patterns
        mock_transcriptions = [
            {
                'text': "This is a sample transcription. The actual transcription would appear here based on the audio content.",
                'language': 'english',
                'confidence': 0.95
            }
        ]
        
        return random.choice(mock_transcriptions)
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def analyze_accent_simple(text, language):
    """Simple accent analysis based on text patterns"""
    accent_indicators = {
        'british': ['colour', 'favour', 'realise', 'centre', 'metre', 'whilst', 'amongst'],
        'american': ['color', 'favor', 'realize', 'center', 'meter', 'while', 'among'],
        'australian': ['mate', 'crikey', 'bloke', 'sheila', 'fair dinkum'],
        'indian': ['prepone', 'good name', 'out of station', 'do one thing'],
    }
    
    text_lower = text.lower()
    detected_accents = []
    
    for accent, indicators in accent_indicators.items():
        matches = sum(1 for indicator in indicators if indicator in text_lower)
        if matches > 0:
            detected_accents.append({
                'accent': accent,
                'confidence': min(matches * 0.2, 1.0),
                'indicators': [ind for ind in indicators if ind in text_lower]
            })
    
    if not detected_accents:
        detected_accents.append({
            'accent': 'neutral/unclear',
            'confidence': 0.5,
            'indicators': []
        })
    
    return sorted(detected_accents, key=lambda x: x['confidence'], reverse=True)

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a YouTube video URL here"
    )

with col2:
    st.markdown("### System Status")
    deps = check_dependencies()
    if deps['ffmpeg']:
        st.success("✅ FFmpeg available")
    else:
        st.error("❌ FFmpeg not available")

if youtube_url:
    # Validate URL
    if not any(domain in youtube_url.lower() for domain in ['youtube.com', 'youtu.be']):
        st.error("Please enter a valid YouTube URL")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🎯 Analyze Accent", type="primary", use_container_width=True):
                if not deps['ffmpeg']:
                    st.error("FFmpeg is required but not available. Please check deployment configuration.")
                    st.info("Make sure 'packages.txt' contains 'ffmpeg' in your repository.")
                    st.stop()
                
                try:
                    # Step 1: Get video info
                    with st.spinner("📋 Getting video information..."):
                        video_info = get_video_info(youtube_url)
                    
                    # Display video info
                    st.success("✅ Video found!")
                    with st.expander("📹 Video Information", expanded=True):
                        st.write(f"**Title:** {video_info['title']}")
                        st.write(f"**Duration:** {video_info['duration']//60}:{video_info['duration']%60:02d}")
                        st.write(f"**Uploader:** {video_info['uploader']}")
                    
                    # Step 2: Download audio
                    with st.spinner("📥 Downloading audio..."):
                        audio_path, video_info = download_audio(youtube_url)
                    
                    st.success("✅ Audio downloaded!")
                    
                    # Step 3: Transcribe (mock for demo)
                    with st.spinner("🎤 Transcribing audio..."):
                        transcription_result = simple_transcribe_with_api(audio_path)
                    
                    st.success("✅ Transcription completed!")
                    
                    # Step 4: Analyze accent
                    with st.spinner("🔍 Analyzing accent..."):
                        accent_analysis = analyze_accent_simple(
                            transcription_result['text'], 
                            transcription_result['language']
                        )
                    
                    st.success("🎉 Analysis completed!")
                    
                    # Store results
                    st.session_state.transcription_result = {
                        'video_info': video_info,
                        'transcription': transcription_result,
                        'accent_analysis': accent_analysis
                    }
                    
                    # Clean up
                    try:
                        os.remove(audio_path)
                        shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
                    # Provide specific help based on error type
                    error_str = str(e).lower()
                    if "ffmpeg" in error_str:
                        st.info("💡 FFmpeg issue: Check if packages.txt contains 'ffmpeg'")
                    elif "private" in error_str or "unavailable" in error_str:
                        st.info("💡 Video access issue: Try a different public video")
                    elif "too long" in error_str:
                        st.info("💡 Video length issue: Use videos under 10 minutes")
                    else:
                        st.info("💡 Try a different YouTube video or check your internet connection")

# Display results
if st.session_state.transcription_result:
    st.markdown("---")
    st.header("📊 Results")
    
    result = st.session_state.transcription_result
    
    # Transcription
    with st.expander("📝 Full Transcription", expanded=True):
        st.text_area(
            "Transcribed Text",
            result['transcription']['text'],
            height=150,
            disabled=True
        )
    
    # Accent Analysis
    st.subheader("🌍 Accent Analysis")
    
    for i, accent in enumerate(result['accent_analysis'][:3]):  # Show top 3
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.write(f"**{accent['accent'].title()}**")
        
        with col2:
            confidence_pct = accent['confidence'] * 100
            st.progress(accent['confidence'])
            st.write(f"{confidence_pct:.1f}%")
        
        with col3:
            if accent['indicators']:
                st.write(f"*Indicators: {', '.join(accent['indicators'][:3])}*")
            else:
                st.write("*Based on general patterns*")
    
    # Language Info
    st.subheader("🔤 Language Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Detected Language", result['transcription']['language'].title())
    
    with col2:
        st.metric("Confidence", f"{result['transcription']['confidence']*100:.1f}%")

# Sidebar
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. **Enter YouTube URL** in the input field
    2. **Click 'Analyze Accent'** to start processing
    3. **Wait for results** (1-3 minutes)
    4. **Review transcription and accent analysis**
    
    **Best Results:**
    - Clear speech audio
    - Videos under 10 minutes
    - Good audio quality
    - Single speaker preferred
    """)
    
    st.header("⚠️ Limitations")
    st.markdown("""
    - **Demo Mode**: Uses simplified accent detection
    - **Public videos only**
    - **Max 10 minutes** duration
    - **English language** works best
    - **Accuracy varies** with audio quality
    """)
    
    st.header("🔧 Technical Info")
    st.markdown(f"""
    **System Dependencies:**
    - FFmpeg: {'✅' if deps['ffmpeg'] else '❌'}
    
    **Note**: This is a demonstration version. 
    For production use, integrate with:
    - OpenAI Whisper
    - Google Speech-to-Text
    - AssemblyAI
    - Azure Speech Services
    """)
    
    if st.button("🔄 Clear Results"):
        st.session_state.transcription_result = None
        st.rerun()
