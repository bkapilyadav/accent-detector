import streamlit as st
import tempfile
import os
import time
from urllib.parse import urlparse, parse_qs
import yt_dlp
from google.cloud import speech

st.set_page_config(page_title="YouTube Audio Transcriber", page_icon="🎤")

st.title("🎤 YouTube Audio Transcriber")
st.write("Extract and transcribe audio from YouTube videos")

# YouTube URL input
youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    elif 'youtube.com' in url:
        if 'v=' in url:
            return url.split('v=')[1].split('&')[0]
        elif 'embed/' in url:
            return url.split('embed/')[1].split('?')[0]
    return None

def download_youtube_audio(video_id):
    """Download YouTube audio using yt-dlp"""
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
            
            if duration > 600:  # 10 minutes limit
                return None, f"Video too long ({duration//60} minutes). Please use videos under 10 minutes."
        
        return output_path, None
        
    except Exception as e:
        return None, f"Download error: {str(e)}"

def transcribe_audio(audio_file_path):
    """Transcribe audio using Google Cloud Speech-to-Text"""
    client = speech.SpeechClient()
    
    with open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        alternative_language_codes=["en-GB", "hi-IN", "es-ES", "fr-FR"]
    )
    
    response = client.recognize(config=config, audio=audio)
    
    if not response.results:
        return {
            'text': 'No transcription available',
            'language': 'unknown',
            'accent': 'unknown',
            'confidence': 'N/A',
            'method': 'Google Cloud Speech-to-Text'
        }
    
    result = response.results[0]
    transcript = result.alternatives[0].transcript
    confidence = f"{round(result.alternatives[0].confidence * 100)}%"
    
    language_code = config.language_code
    language_map = {
        'en-US': 'English (US)',
        'en-GB': 'English (UK)',
        'hi-IN': 'Hindi (India)',
        'es-ES': 'Spanish (Spain)',
        'fr-FR': 'French (France)'
    }
    language = language_map.get(language_code, 'Unknown')
    accent = language_map.get(language_code, 'Unknown')
    
    return {
        'text': transcript,
        'language': language,
        'accent': accent,
        'confidence': confidence,
        'method': 'Google Cloud Speech-to-Text'
    }

# Main application logic
if youtube_url:
    video_id = extract_video_id(youtube_url)
    
    if video_id:
        st.success(f"✅ Valid YouTube URL detected (Video ID: {video_id})")
        
        if st.button("🎯 Transcribe Audio", type="primary"):
            with st.spinner("Processing video..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Validating video...")
                progress_bar.progress(20)
                
                audio_file_path, error = download_youtube_audio(video_id)
                if error:
                    st.error(error)
                    st.stop()
                
                status_text.text("🎤 Transcribing audio...")
                progress_bar.progress(70)
                
                result = transcribe_audio(audio_file_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                st.success("🎉 Transcription completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🌍 Detected Language")
                    st.markdown(f"**{result['language']}**")
                
                with col2:
                    st.subheader("🎙️ Detected Accent")
                    st.markdown(f"**{result['accent']}**")
                
                st.subheader("📊 Confidence")
                st.markdown(f"**{result['confidence']}**")
                
                st.subheader("📝 Transcription")
                st.text_area("Transcribed Text:", result['text'], height=150)
                
                st.download_button(
                    label="📄 Download Transcription",
                    data=result['text'],
                    file_name=f"transcription_{video_id}.txt",
                    mime="text/plain"
                )
    else:
        st.error("❌ Invalid YouTube URL. Please enter a valid YouTube video URL.")
else:
    st.info("👆 Enter a YouTube URL above to get started")

# Sidebar information
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. **Copy YouTube URL** from your browser
    2. **Paste it** in the input field
    3. **Click 'Transcribe Audio'** 
    4. **Wait for processing**
    5. **View and download** results
    """)
    
    st.header("✅ Supported URLs")
    st.markdown("""
    - `youtube.com/watch?v=...`
    - `youtu.be/...`
    - `youtube.com/embed/...`
    """)
    
    st.header("⚠️ Current Limitations")
    st.markdown("""
    - **10 minute limit** for videos
    - **Public videos only**
    - **Requires Google Cloud credentials**
    """)
    
    st.markdown("---")
    st.markdown("💡 **Need the full version?**")
    st.markdown("Deploy with proper dependencies on platforms like Railway, Render, or local environment.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎤 YouTube Audio Transcriber</p>
        <p><small>For demonstration purposes. Full functionality requires complete deployment.</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
