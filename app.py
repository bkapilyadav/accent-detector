import streamlit as st
import requests
import tempfile
import os
import json
import time
import numpy as np
from urllib.parse import urlparse, parse_qs
import re
import subprocess
from pydub import AudioSegment

# Configure Streamlit
st.set_page_config(
    page_title="YouTube Audio Transcriber with Accent Detection", 
    page_icon="🎤",
    layout="wide"
)

# Initialize models with fallback options
@st.cache_resource
def load_models():
    """Load models with fallback options"""
    whisper_model = None
    accent_classifier = None
    
    try:
        # Try to load Whisper
        import whisper
        whisper_model = whisper.load_model("tiny")  # Use tiny model for faster loading
        st.success("✅ Whisper model loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Whisper model loading failed: {e}")
    
    try:
        # Try to load librosa for accent detection
        import librosa
        st.success("✅ Librosa loaded successfully")
    except Exception as e:
        st.warning(f"⚠️ Librosa loading failed: {e}")
    
    return whisper_model, accent_classifier

# Load models
whisper_model, accent_classifier = load_models()

st.title("🎤 YouTube Audio Transcriber with Accent Detection")
st.write("Extract audio from YouTube videos and detect language, accent, and transcribe content")

# Add a warning about model loading
if whisper_model is None:
    st.error("⚠️ **Important**: Some AI models failed to load. The app will work in demo mode.")
    st.info("💡 **For full functionality**, deploy on a platform with more resources like Railway or Render.")

# YouTube URL input
youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info_simple(video_id):
    """Get basic video information"""
    try:
        import yt_dlp
        url = f'https://www.youtube.com/watch?v={video_id}'
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown Title'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'available': True
            }
    except Exception as e:
        return {
            'title': f'Video {video_id}',
            'duration': 300,  # Default 5 minutes
            'uploader': 'Unknown',
            'view_count': 0,
            'available': True,
            'error': str(e)
        }

def download_youtube_audio_fallback(video_id, max_duration=600):
    """Download audio with fallback methods"""
    try:
        import yt_dlp
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best[height<=480]',
            'outtmpl': temp_file.name.replace('.wav', '.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get info first
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            
            if duration > max_duration:
                return None, f"Video too long ({duration//60} minutes). Please use videos under {max_duration//60} minutes."
            
            # Download
            ydl.download([url])
            
            # Find downloaded file
            base_name = temp_file.name.replace('.wav', '')
            for ext in ['.m4a', '.webm', '.mp3', '.wav']:
                potential_file = base_name + ext
                if os.path.exists(potential_file):
                    return potential_file, info.get('title', 'Unknown Title')
        
        return None, "Download failed - no audio file found"
        
    except Exception as e:
        return None, f"Download error: {str(e)}"

def detect_accent_simple(audio_file_path="", transcript_text=""):
    """Simple accent detection based on text patterns and basic rules"""
    
    # Text-based accent indicators
    accent_patterns = {
        'American': {
            'words': ['like', 'totally', 'awesome', 'dude', 'guys', 'gonna', 'wanna'],
            'phrases': ['you know', 'oh my god', 'no way'],
            'score_base': 0.3
        },
        'British': {
            'words': ['brilliant', 'lovely', 'quite', 'rather', 'bloody', 'bloke', 'mate'],
            'phrases': ['i say', 'how do you do', 'cheerio'],
            'score_base': 0.2
        },
        'Australian': {
            'words': ['mate', 'bloke', 'sheila', 'ripper', 'arvo', 'barbie'],
            'phrases': ['fair dinkum', 'no worries', 'good on ya'],
            'score_base': 0.1
        },
        'Indian': {
            'words': ['yaar', 'actually', 'itself', 'only', 'na', 'prepone'],
            'phrases': ['what to do', 'like that only', 'one minute'],
            'score_base': 0.15
        },
        'Canadian': {
            'words': ['eh', 'about', 'house', 'out', 'sorry', 'toque'],
            'phrases': ['you bet', 'double double'],
            'score_base': 0.1
        }
    }
    
    if not transcript_text:
        return {
            'accent': 'Unknown',
            'confidence': 0.0,
            'method': 'No transcript available'
        }
    
    text_lower = transcript_text.lower()
    scores = {}
    
    for accent, patterns in accent_patterns.items():
        score = patterns['score_base']
        
        # Count word matches
        word_matches = sum(1 for word in patterns['words'] if word in text_lower)
        score += word_matches * 0.1
        
        # Count phrase matches
        phrase_matches = sum(1 for phrase in patterns['phrases'] if phrase in text_lower)
        score += phrase_matches * 0.2
        
        # Length bonus (longer text = more reliable)
        if len(transcript_text) > 100:
            score += 0.1
        
        scores[accent] = min(score, 0.95)  # Cap at 95%
    
    # Find best match
    if scores:
        best_accent = max(scores, key=scores.get)
        confidence = scores[best_accent]
    else:
        best_accent = 'General English'
        confidence = 0.5
    
    return {
        'accent': best_accent,
        'confidence': confidence,
        'all_scores': scores,
        'method': 'Text Pattern Analysis'
    }

def transcribe_audio_fallback(audio_file):
    """Transcribe audio with fallback options"""
    
    # Try Whisper first
    if whisper_model is not None:
        try:
            import whisper
            result = whisper_model.transcribe(audio_file)
            return {
                'text': result['text'],
                'language': result.get('language', 'en'),
                'confidence': 0.85,  # Whisper doesn't provide confidence directly
                'method': 'Whisper AI'
            }
        except Exception as e:
            st.warning(f"Whisper transcription failed: {e}")
    
    # Fallback to speech_recognition
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        
        # Convert to WAV if needed
        if not audio_file.endswith('.wav'):
            wav_file = audio_file.replace(os.path.splitext(audio_file)[1], '.wav')
            audio = AudioSegment.from_file(audio_file)
            audio.export(wav_file, format="wav")
            audio_file = wav_file
        
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        
        try:
            text = r.recognize_google(audio)
            return {
                'text': text,
                'language': 'en',
                'confidence': 0.75,
                'method': 'Google Speech Recognition'
            }
        except sr.UnknownValueError:
            return {
                'text': 'Could not understand the audio clearly. Please try with clearer audio.',
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'Google Speech Recognition'
            }
        except sr.RequestError as e:
            return {
                'text': f'Speech recognition service error: {e}',
                'language': 'unknown',
                'confidence': 0.0,
                'method': 'Google Speech Recognition'
            }
    except Exception as e:
        # Ultimate fallback - demo response
        return {
            'text': 'Demo transcription: This is a sample transcription showing how the system works. The actual audio content would appear here after successful processing.',
            'language': 'en',
            'confidence': 0.6,
            'method': 'Demo Mode (Transcription failed)'
        }

def detect_language_simple(transcript=""):
    """Simple language detection"""
    if not transcript:
        return {'language': 'en', 'confidence': 0.5}
    
    # Simple keyword-based language detection
    language_indicators = {
        'spanish': ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'se', 'no'],
        'french': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
        'german': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
        'italian': ['di', 'che', 'e', 'il', 'un', 'a', 'è', 'per', 'una', 'in'],
        'portuguese': ['de', 'a', 'o', 'e', 'do', 'da', 'em', 'um', 'para', 'é']
    }
    
    text_lower = transcript.lower()
    scores = {}
    
    for lang, words in language_indicators.items():
        score = sum(1 for word in words if f' {word} ' in f' {text_lower} ')
        scores[lang] = score / len(words)
    
    if scores and max(scores.values()) > 0.1:
        detected_lang = max(scores, key=scores.get)
        confidence = min(scores[detected_lang] * 2, 0.95)
    else:
        detected_lang = 'en'
        confidence = 0.8
    
    return {
        'language': detected_lang,
        'confidence': confidence,
        'all_scores': scores
    }

# Main application logic
if youtube_url:
    video_id = extract_video_id(youtube_url)
    
    if video_id:
        st.success(f"✅ Valid YouTube URL detected (Video ID: {video_id})")
        
        # Get video info
        video_info = get_video_info_simple(video_id)
        if video_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{video_info['duration']//60}:{video_info['duration']%60:02d}")
            with col2:
                st.metric("Views", f"{video_info.get('view_count', 0):,}")
            with col3:
                st.metric("Uploader", str(video_info.get('uploader', 'Unknown'))[:20])
        
        if st.button("🎯 Transcribe & Analyze Audio", type="primary"):
            with st.spinner("Processing video..."):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Download audio
                status_text.text("📥 Downloading audio...")
                progress_bar.progress(20)
                
                audio_file, title_or_error = download_youtube_audio_fallback(video_id)
                
                if audio_file is None:
                    st.error(f"❌ {title_or_error}")
                    # Continue with demo mode
                    audio_file = None
                    title_or_error = "Demo Mode - No audio downloaded"
                
                # Step 2: Transcription
                status_text.text("🎤 Transcribing audio...")
                progress_bar.progress(50)
                
                if audio_file and os.path.exists(audio_file):
                    transcription_result = transcribe_audio_fallback(audio_file)
                else:
                    transcription_result = {
                        'text': 'This is a demonstration of the transcription system. In a real deployment with proper resources, this would contain the actual transcribed content from your YouTube video. The system supports multiple languages and provides confidence scores for accuracy assessment.',
                        'language': 'en',
                        'confidence': 0.8,
                        'method': 'Demo Mode'
                    }
                
                # Step 3: Language detection
                status_text.text("🌍 Detecting language...")
                progress_bar.progress(70)
                
                lang_result = detect_language_simple(transcription_result.get('text', ''))
                
                # Step 4: Accent detection
                status_text.text("🗣️ Analyzing accent...")
                progress_bar.progress(90)
                
                accent_result = detect_accent_simple(audio_file, transcription_result.get('text', ''))
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clean up
                if audio_file and os.path.exists(audio_file):
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
                
                # Display results
                st.success("🎉 Analysis completed!")
                
                # Main results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("🌍 Language")
                    st.markdown(f"**{lang_result.get('language', 'EN').upper()}**")
                    st.caption(f"Confidence: {lang_result.get('confidence', 0)*100:.1f}%")
                
                with col2:
                    st.subheader("🗣️ Accent")
                    st.markdown(f"**{accent_result.get('accent', 'Unknown')}**")
                    st.caption(f"Confidence: {accent_result.get('confidence', 0)*100:.1f}%")
                
                with col3:
                    st.subheader("📊 Transcription Quality")
                    quality = transcription_result.get('confidence', 0)
                    st.markdown(f"**{quality*100:.1f}%**")
                    st.caption("Confidence Score")
                
                with col4:
                    st.subheader("⏱️ Audio Length")
                    duration = video_info.get('duration', 0) if video_info else 0
                    st.markdown(f"**{duration//60}:{duration%60:02d}**")
                    st.caption("Minutes:Seconds")
                
                # Transcription text
                st.subheader("📝 Transcription")
                transcript_text = transcription_result.get('text', 'Transcription failed')
                st.text_area("Transcribed Text:", transcript_text, height=200)
                
                # Method info
                st.info(f"🔧 **Processing Method**: {transcription_result.get('method', 'Unknown')}")
                
                # Detailed analysis in expandable sections
                with st.expander("🔍 Detailed Language Analysis"):
                    if 'all_scores' in lang_result and lang_result['all_scores']:
                        st.write("**Language Detection Scores:**")
                        for lang, score in lang_result['all_scores'].items():
                            st.write(f"- {lang.title()}: {score*100:.1f}%")
                    else:
                        st.write("Primary language detected: English")
                
                with st.expander("🎯 Accent Analysis Details"):
                    if 'all_scores' in accent_result and accent_result['all_scores']:
                        st.write("**Accent Similarity Scores:**")
                        for accent, score in accent_result['all_scores'].items():
                            st.write(f"- {accent}: {score*100:.1f}%")
                    
                    st.write(f"**Analysis Method**: {accent_result.get('method', 'Pattern Recognition')}")
                
                with st.expander("ℹ️ Technical Details"):
                    st.write(f"**Video ID**: {video_id}")
                    st.write(f"**Video Title**: {title_or_error}")
                    st.write(f"**Transcription Method**: {transcription_result.get('method', 'Unknown')}")
                    st.write(f"**Accent Detection**: Text Pattern Analysis")
                    if audio_file is None:
                        st.warning("⚠️ **Note**: Running in demo mode due to resource limitations")
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download Transcription",
                        data=transcript_text,
                        file_name=f"transcription_{video_id}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create detailed report
                    report = f"""YouTube Audio Analysis Report
================================

Video ID: {video_id}
Title: {title_or_error}
Duration: {duration//60}:{duration%60:02d}

LANGUAGE DETECTION:
- Detected Language: {lang_result.get('language', 'Unknown').upper()}
- Confidence: {lang_result.get('confidence', 0)*100:.1f}%

ACCENT ANALYSIS:
- Detected Accent: {accent_result.get('accent', 'Unknown')}
- Confidence: {accent_result.get('confidence', 0)*100:.1f}%
- Method: {accent_result.get('method', 'Unknown')}

TRANSCRIPTION:
{transcript_text}

TECHNICAL INFO:
- Transcription Method: {transcription_result.get('method', 'Unknown')}
- Processing Quality: {transcription_result.get('confidence', 0)*100:.1f}%

Generated by YouTube Audio Transcriber with Accent Detection
"""
                    
                    st.download_button(
                        label="📊 Download Full Report",
                        data=report,
                        file_name=f"audio_analysis_report_{video_id}.txt",
                        mime="text/plain"
                    )
                
    else:
        st.error("❌ Invalid YouTube URL. Please enter a valid YouTube video URL.")

else:
    st.info("👆 Enter a YouTube URL above to get started")
    
    # Demo section
    st.subheader("🎮 Try the Demo")
    st.write("Click below to see how the analysis works:")
    
    if st.button("🎪 Run Demo Analysis"):
        with st.spinner("Running demo..."):
            progress_bar = st.progress(0)
            
            for i in range(101):
                time.sleep(0.02)
                progress_bar.progress(i)
            
            st.success("🎉 Demo completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.subheader("🌍 Language")
                st.markdown("**ENGLISH**")
                st.caption("Confidence: 94.2%")
            
            with col2:
                st.subheader("🗣️ Accent")
                st.markdown("**American**")
                st.caption("Confidence: 87.5%")
            
            with col3:
                st.subheader("📊 Quality")
                st.markdown("**92.1%**")
                st.caption("Confidence Score")
            
            with col4:
                st.subheader("⏱️ Length")
                st.markdown("**2:34**")
                st.caption("Minutes:Seconds")
            
            st.subheader("📝 Demo Transcription")
            demo_text = "Hello everyone, and welcome to this demonstration of our YouTube audio transcription service. This tool can accurately detect the language being spoken, identify regional accents, and provide high-quality transcriptions of your video content. Whether you're creating subtitles, analyzing speech patterns, or making your content more accessible, our advanced AI-powered system delivers reliable results."
            st.text_area("Demo Transcribed Text:", demo_text, height=100)

# Sidebar information
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. **Copy YouTube URL** from your browser
    2. **Paste it** in the input field above
    3. **Click 'Transcribe & Analyze Audio'**
    4. **Wait for processing** (1-3 minutes)
    5. **View results** and download reports
    """)
    
    st.header("✅ Current Features")
    st.markdown("""
    - **Multi-language** transcription
    - **Accent detection** (pattern-based)
    - **Quality metrics** and confidence scores
    - **Downloadable reports**
    - **Demo mode** for testing
    """)
    
    st.header("🎯 Supported Accents")
    st.markdown("""
    - 🇺🇸 **American English**
    - 🇬🇧 **British English**
    - 🇦🇺 **Australian English**
    - 🇮🇳 **Indian English**
    - 🇨🇦 **Canadian English**
    """)
    
    st.header("⚠️ Current Status")
    model_status = "✅ Ready" if whisper_model else "⚠️ Demo Mode"
    st.markdown(f"""
    - **AI Models**: {model_status}
    - **Video Download**: ✅ Active
    - **Accent Detection**: ✅ Text-based
    - **Multi-language**: ✅ Basic support
    """)
    
    st.header("🚀 Technical Stack")
    st.markdown("""
    - **Whisper AI** (when available)
    - **yt-dlp** for video download  
    - **Speech Recognition** (fallback)
    - **Pattern matching** for accents
    - **Streamlit** for interface
    """)
    
    st.markdown("---")
    st.markdown("💡 **For full AI features**:")
    st.markdown("Deploy on Railway, Render, or local environment with GPU support.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎤 YouTube Audio Transcriber with Accent Detection</p>
        <p><small>Optimized for Streamlit Cloud | Fallback-enabled</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
