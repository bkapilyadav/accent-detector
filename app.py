import streamlit as st
import requests
import tempfile
import os
import json
from urllib.parse import urlparse, parse_qs
import re
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.error("⚠️ Advanced audio analysis not available on Streamlit Cloud")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    st.error("⚠️ YouTube download not available")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import langdetect
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

st.set_page_config(
    page_title="YouTube Audio Transcriber - Streamlit Cloud", 
    page_icon="🎤", 
    layout="wide"
)

st.title("🎤 YouTube Audio Transcriber (Streamlit Cloud Version)")
st.write("Language and accent detection for YouTube videos")

# Display availability status
with st.expander("🔧 Feature Availability Check"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Audio Processing:**")
        st.write(f"✅ Librosa: {LIBROSA_AVAILABLE}" if LIBROSA_AVAILABLE else "❌ Librosa: Not Available")
        st.write(f"✅ Speech Recognition: {SPEECH_RECOGNITION_AVAILABLE}" if SPEECH_RECOGNITION_AVAILABLE else "❌ Speech Recognition: Not Available")
        st.write(f"✅ YouTube Download: {YT_DLP_AVAILABLE}" if YT_DLP_AVAILABLE else "❌ YouTube Download: Not Available")
    
    with col2:
        st.write("**Language Detection:**")
        st.write(f"✅ TextBlob: {TEXTBLOB_AVAILABLE}" if TEXTBLOB_AVAILABLE else "❌ TextBlob: Not Available")
        st.write(f"✅ LangDetect: {LANGDETECT_AVAILABLE}" if LANGDETECT_AVAILABLE else "❌ LangDetect: Not Available")

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

def get_video_info_fallback(video_id):
    """Get basic video information - fallback method"""
    try:
        if YT_DLP_AVAILABLE:
            url = f'https://www.youtube.com/watch?v={video_id}'
            ydl_opts = {'quiet': True, 'no_warnings': True}
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Video Title'),
                    'duration': info.get('duration', 0),
                    'available': True
                }
        else:
            return {
                'title': f'Video {video_id}',
                'duration': 180,  # Default 3 minutes
                'available': True
            }
    except Exception as e:
        st.warning(f"Could not fetch video info: {str(e)}")
        return {
            'title': f'Video {video_id}',
            'duration': 180,
            'available': True
        }

def detect_language_simple(text):
    """Simple language detection with fallbacks"""
    if not text or len(text.strip()) < 3:
        return "English", 0.5
    
    try:
        if LANGDETECT_AVAILABLE:
            detected_lang = langdetect.detect(text)
            lang_mapping = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
                'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
            }
            language = lang_mapping.get(detected_lang, detected_lang.upper())
            return language, 0.8
        
        # Fallback: simple keyword detection
        text_lower = text.lower()
        if any(word in text_lower for word in ['the', 'and', 'you', 'that', 'was', 'for', 'are']):
            return "English", 0.7
        elif any(word in text_lower for word in ['el', 'la', 'y', 'que', 'de', 'en', 'un']):
            return "Spanish", 0.6
        elif any(word in text_lower for word in ['le', 'de', 'et', 'à', 'un', 'il', 'être']):
            return "French", 0.6
        else:
            return "English", 0.5
            
    except Exception as e:
        st.warning(f"Language detection error: {str(e)}")
        return "English", 0.5

def detect_accent_simple(text, video_id):
    """Simple accent detection based on text patterns"""
    if not text:
        return "General English", 0.5
    
    text_lower = text.lower()
    accent_scores = {}
    
    # American English indicators
    american_words = ['color', 'favor', 'center', 'aluminum', 'mom', 'candy', 'gas', 'apartment']
    american_score = sum(1 for word in american_words if word in text_lower)
    if american_score > 0:
        accent_scores['American'] = min(0.8, 0.3 + american_score * 0.1)
    
    # British English indicators
    british_words = ['colour', 'favour', 'centre', 'aluminium', 'mum', 'sweets', 'petrol', 'flat']
    british_score = sum(1 for word in british_words if word in text_lower)
    if british_score > 0:
        accent_scores['British'] = min(0.8, 0.3 + british_score * 0.1)
    
    # Australian indicators
    australian_words = ['mate', 'bloke', 'fair dinkum', 'arvo', 'barbie', 'brekkie']
    australian_score = sum(1 for word in australian_words if word in text_lower)
    if australian_score > 0:
        accent_scores['Australian'] = min(0.8, 0.4 + australian_score * 0.1)
    
    # Indian English indicators
    indian_words = ['yaar', 'na', 'only', 'itself', 'good name', 'out of station']
    indian_score = sum(1 for phrase in indian_words if phrase in text_lower)
    if indian_score > 0:
        accent_scores['Indian'] = min(0.8, 0.4 + indian_score * 0.1)
    
    # Default scoring based on video ID patterns (demo purposes)
    if not accent_scores:
        vid_hash = hash(video_id) % 100
        if vid_hash < 25:
            accent_scores['American'] = 0.6
        elif vid_hash < 45:
            accent_scores['British'] = 0.55
        elif vid_hash < 60:
            accent_scores['Australian'] = 0.5
        elif vid_hash < 75:
            accent_scores['Indian'] = 0.5
        else:
            accent_scores['Canadian'] = 0.5
    
    if accent_scores:
        best_accent = max(accent_scores, key=accent_scores.get)
        confidence = accent_scores[best_accent]
        return best_accent, confidence
    else:
        return "General English", 0.5

def transcribe_demo(video_id, video_title):
    """Demo transcription based on video characteristics"""
    
    # Generate demo transcription based on video ID
    demo_transcriptions = [
        "Hello everyone, welcome to this video tutorial. Today we're going to learn about artificial intelligence and machine learning concepts.",
        "Good morning, and thank you for joining us today. In this presentation, we'll discuss the latest developments in technology.",
        "Hi there! In this video, I'll be showing you how to create amazing content using various tools and techniques.",
        "Welcome back to our channel. Today's topic is really interesting and I'm excited to share it with you.",
        "Greetings everyone! This is a comprehensive guide that will help you understand complex topics in simple terms."
    ]
    
    # Select demo transcription based on video ID
    transcript_index = hash(video_id) % len(demo_transcriptions)
    base_transcript = demo_transcriptions[transcript_index]
    
    # Add video-specific context
    if len(video_title) > 10:
        context_words = video_title.lower().split()[:3]
        context = f"This video about {' '.join(context_words)} provides valuable insights. "
        transcript = context + base_transcript
    else:
        transcript = base_transcript
    
    return transcript

# Main application logic
if youtube_url:
    video_id = extract_video_id(youtube_url)
    
    if video_id:
        st.success(f"✅ Valid YouTube URL detected (Video ID: {video_id})")
        
        # Get video info
        video_info = get_video_info_fallback(video_id)
        if video_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                title_display = video_info['title'][:30] + "..." if len(video_info['title']) > 30 else video_info['title']
                st.metric("📺 Title", title_display)
            with col2:
                duration_display = f"{video_info['duration']//60}:{video_info['duration']%60:02d}"
                st.metric("⏱️ Duration", duration_display)
            with col3:
                st.metric("📊 Status", "Demo Mode")
        
        if st.button("🎯 Analyze Audio (Demo Mode)", type="primary"):
            
            with st.spinner("Processing video..."):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Video validation
                status_text.text("🔍 Validating video...")
                progress_bar.progress(20)
                time.sleep(1)
                
                # Step 2: Transcription (demo)
                status_text.text("🎤 Generating demo transcription...")
                progress_bar.progress(50)
                time.sleep(2)
                
                transcribed_text = transcribe_demo(video_id, video_info['title'])
                
                # Step 3: Language detection
                status_text.text("🌍 Detecting language...")
                progress_bar.progress(70)
                time.sleep(1)
                
                detected_language, lang_confidence = detect_language_simple(transcribed_text)
                
                # Step 4: Accent detection  
                status_text.text("🗣️ Analyzing accent...")
                progress_bar.progress(85)
                time.sleep(1)
                
                detected_accent, accent_confidence = detect_accent_simple(transcribed_text, video_id)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                time.sleep(1)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.success("🎉 Demo Analysis Completed!")
                
                # Main results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🌍 Language Detection")
                    st.metric("Detected Language", detected_language)
                    st.metric("Confidence", f"{lang_confidence:.1%}")
                
                with col2:
                    st.subheader("🗣️ Accent Detection")
                    st.metric("Detected Accent", detected_accent)
                    st.metric("Confidence", f"{accent_confidence:.1%}")
                
                with col3:
                    st.subheader("📊 Processing Info")
                    st.metric("Mode", "Demo")
                    st.metric("Platform", "Streamlit Cloud")
                
                # Transcription
                st.subheader("📝 Demo Transcription")
                st.text_area("Generated Text:", transcribed_text, height=150)
                
                # Technical details
                with st.expander("ℹ️ Demo Information"):
                    st.write("**Note:** This is a demonstration version running on Streamlit Cloud.")
                    st.write("**Limitations:**")
                    st.write("- No actual audio processing")
                    st.write("- Simulated transcription based on video metadata")
                    st.write("- Pattern-based accent detection")
                    st.write("- Limited language detection capabilities")
                    st.write("")
                    st.write("**For full functionality:**")
                    st.write("- Deploy on platforms with audio processing support")
                    st.write("- Use local installation with all dependencies")
                    st.write("- Consider cloud services like Railway, Render, or AWS")
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📄 Download Demo Transcription",
                        data=transcribed_text,
                        file_name=f"demo_transcription_{video_id}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create demo report
                    report = f"""YouTube Audio Analysis Demo Report
=========================================

Video Information:
- Video ID: {video_id}
- Title: {video_info['title']}
- Duration: {video_info['duration']//60}:{video_info['duration']%60:02d}
- Platform: Streamlit Cloud (Demo Mode)

Analysis Results:
- Detected Language: {detected_language} ({lang_confidence:.1%} confidence)
- Detected Accent: {detected_accent} ({accent_confidence:.1%} confidence)

Demo Transcription:
{transcribed_text}

Note: This is a demonstration version. For full audio processing
capabilities, please deploy on a platform that supports audio libraries
or run locally with complete dependencies.

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                    st.download_button(
                        label="📊 Download Demo Report",
                        data=report,
                        file_name=f"demo_analysis_report_{video_id}.txt",
                        mime="text/plain"
                    )
    else:
        st.error("❌ Invalid YouTube URL. Please enter a valid YouTube video URL.")

else:
    st.info("👆 Enter a YouTube URL above to get started with the demo")
    
    # Feature explanation
    st.subheader("🎯 About This Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demo Features:**")
        st.write("- URL validation and video info extraction")
        st.write("- Simulated language detection")
        st.write("- Pattern-based accent analysis")
        st.write("- Demo transcription generation")
        st.write("- Report generation and downloads")
    
    with col2:
        st.write("**Streamlit Cloud Limitations:**")
        st.write("- No audio processing libraries")
        st.write("- Limited system packages")
        st.write("- No real YouTube audio download")
        st.write("- Simplified analysis methods")
        st.write("- Demo mode only")

# Sidebar information
with st.sidebar:
    st.header("📖 Demo Instructions")
    st.markdown("""
    1. **Enter YouTube URL** in the input field
    2. **Click 'Analyze Audio'** button
    3. **Wait for demo processing**
    4. **View simulated results**
    5. **Download demo reports**
    """)
    
    st.header("⚠️ Streamlit Cloud Limitations")
    st.markdown("""
    - **No real audio processing**
    - **Demo transcription only**
    - **Simulated accent detection**
    - **Limited package support**
    - **No YouTube downloads**
    """)
    
    st.header("🚀 Full Version Deployment")
    st.markdown("""
    **Recommended Platforms:**
    - **Railway** (Good audio support)
    - **Render** (Full Linux packages)
    - **DigitalOcean** (Custom setup)
    - **AWS/GCP** (Complete control)
    - **Local Installation** (Best option)
    """)
    
    st.header("💡 Getting Full Features")
    st.markdown("""
    1. **Clone the repository**
    2. **Deploy on Railway/Render**
    3. **Use Docker deployment**
    4. **Run locally with setup.sh**
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎤 YouTube Audio Transcriber | Streamlit Cloud Demo</p>
        <p><small>Limited demo version - Deploy elsewhere for full functionality</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Warning notice
st.warning("""
🚨 **Important Notice**: This is a limited demo version running on Streamlit Cloud. 
Many audio processing libraries are not available on this platform. 

For full functionality with real audio transcription and accent detection, 
please deploy on Railway, Render, or run locally with the complete setup.
""")
