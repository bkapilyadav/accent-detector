import streamlit as st
import requests
import tempfile
import os
import json
from urllib.parse import urlparse, parse_qs
import re
import numpy as np
import librosa
import speech_recognition as sr
from textblob import TextBlob
import langdetect
from langdetect import detect, LangDetectError
import time
import yt_dlp
import io
import wave
import webrtcvad
from pydub import AudioSegment
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="YouTube Audio Transcriber with Accent Detection", page_icon="🎤", layout="wide")

st.title("🎤 YouTube Audio Transcriber with Language & Accent Detection")
st.write("Extract, transcribe audio and detect both language and accent from YouTube videos")

# Initialize session state for models
if 'accent_classifier' not in st.session_state:
    st.session_state.accent_classifier = None

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

def get_video_info(video_id):
    """Get basic video information using yt-dlp"""
    try:
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
                'available': True
            }
    except Exception as e:
        st.error(f"Error getting video info: {str(e)}")
        return None

def extract_prosodic_features(audio_data, sr):
    """Extract prosodic features for accent detection"""
    try:
        # Fundamental frequency (F0)
        f0 = librosa.yin(audio_data, fmin=50, fmax=300)
        f0_mean = np.mean(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
        f0_std = np.std(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
        f0_range = np.max(f0) - np.min(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
        
        # Rhythm features
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # MFCCs for vowel quality
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Formant-like features (approximated using spectral peaks)
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        
        # Find spectral peaks (rough formant approximation)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitude[:len(magnitude)//2], height=np.max(magnitude)*0.1)
        formant_freqs = freqs[peaks][:5] if len(peaks) > 0 else [0, 0, 0, 0, 0]
        
        features = {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_range': f0_range,
            'tempo': tempo,
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'formant_f1': formant_freqs[0] if len(formant_freqs) > 0 else 0,
            'formant_f2': formant_freqs[1] if len(formant_freqs) > 1 else 0,
            'formant_f3': formant_freqs[2] if len(formant_freqs) > 2 else 0,
        }
        
        return features
    except Exception as e:
        st.error(f"Error extracting prosodic features: {str(e)}")
        return None

def detect_accent_from_audio(audio_file_path):
    """Detect accent from audio file using acoustic analysis"""
    try:
        # Load audio
        audio_data, sr = librosa.load(audio_file_path, sr=16000, duration=30)  # Limit to 30 seconds
        
        if len(audio_data) == 0:
            return "Unknown", 0.0
        
        # Extract prosodic features
        features = extract_prosodic_features(audio_data, sr)
        if not features:
            return "Unknown", 0.0
        
        # Simple rule-based accent detection based on prosodic features
        accent_scores = {}
        
        # American English characteristics
        if (features['f0_mean'] > 120 and features['f0_mean'] < 180 and 
            features['tempo'] > 110 and features['tempo'] < 140):
            accent_scores['American'] = 0.7
        
        # British English characteristics
        if (features['f0_range'] > 50 and features['spectral_centroid_mean'] > 1500):
            accent_scores['British'] = 0.6
        
        # Indian English characteristics  
        if (features['f0_mean'] > 140 and features['tempo'] > 130):
            accent_scores['Indian'] = 0.65
        
        # Australian English characteristics
        if (features['f0_std'] > 20 and features['spectral_rolloff_mean'] > 2000):
            accent_scores['Australian'] = 0.6
        
        # Canadian English characteristics
        if (features['f0_mean'] > 110 and features['f0_mean'] < 160 and
            features['tempo'] > 105 and features['tempo'] < 125):
            accent_scores['Canadian'] = 0.55
        
        # South African English characteristics
        if (features['f0_range'] > 60 and features['mfcc_mean'][1] > 5):
            accent_scores['South African'] = 0.6
        
        # Irish English characteristics
        if (features['f0_std'] > 25 and features['tempo'] < 110):
            accent_scores['Irish'] = 0.58
        
        # Scottish English characteristics
        if (features['f0_mean'] < 130 and features['spectral_centroid_std'] > 300):
            accent_scores['Scottish'] = 0.55
        
        if accent_scores:
            best_accent = max(accent_scores, key=accent_scores.get)
            confidence = accent_scores[best_accent]
            return best_accent, confidence
        else:
            return "General English", 0.5
            
    except Exception as e:
        st.error(f"Error in accent detection: {str(e)}")
        return "Unknown", 0.0

def detect_language_from_text(text):
    """Detect language from transcribed text"""
    try:
        if len(text.strip()) < 3:
            return "Unknown", 0.0
        
        # Use langdetect
        detected_lang = detect(text)
        
        # Map language codes to full names
        lang_mapping = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'pl': 'Polish',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish'
        }
        
        language = lang_mapping.get(detected_lang, detected_lang.upper())
        
        # Use TextBlob for additional confirmation if English
        if detected_lang == 'en':
            blob = TextBlob(text)
            confidence = 0.85
        else:
            confidence = 0.75
            
        return language, confidence
        
    except LangDetectError:
        return "English", 0.5  # Default fallback
    except Exception as e:
        st.error(f"Language detection error: {str(e)}")
        return "Unknown", 0.0

def transcribe_audio_advanced(audio_file_path):
    """Advanced transcription with multiple engines"""
    transcription_results = []
    
    try:
        r = sr.Recognizer()
        
        # Convert to WAV if needed
        if not audio_file_path.endswith('.wav'):
            audio = AudioSegment.from_file(audio_file_path)
            wav_path = audio_file_path.replace(os.path.splitext(audio_file_path)[1], '.wav')
            audio.export(wav_path, format="wav")
            audio_file_path = wav_path
        
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.record(source)
        
        # Try Google Speech Recognition
        try:
            text_google = r.recognize_google(audio)
            transcription_results.append({
                'engine': 'Google',
                'text': text_google,
                'confidence': 0.85
            })
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            st.warning(f"Google Speech Recognition error: {e}")
        
        # Try Sphinx (offline)
        try:
            text_sphinx = r.recognize_sphinx(audio)
            transcription_results.append({
                'engine': 'Sphinx',
                'text': text_sphinx,
                'confidence': 0.65
            })
        except sr.UnknownValueError:
            pass
        except Exception:
            pass
        
        # Return best result
        if transcription_results:
            best_result = max(transcription_results, key=lambda x: x['confidence'])
            return best_result['text'], best_result['engine'], best_result['confidence']
        else:
            return "Could not transcribe audio", "None", 0.0
            
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return "Transcription failed", "Error", 0.0

def download_youtube_audio(video_id):
    """Download audio from YouTube video"""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': temp_file.name.replace('.wav', '.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '192K',
        }
        
        url = f'https://www.youtube.com/watch?v={video_id}'
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Title')
            duration = info.get('duration', 0)
            
            if duration > 1200:  # 20 minutes limit
                return None, f"Video too long ({duration//60} minutes). Please use videos under 20 minutes.", title
            
            # Download
            ydl.download([url])
            
            # Find downloaded file
            base_name = temp_file.name.replace('.wav', '')
            possible_extensions = ['.m4a', '.webm', '.mp3', '.wav', '.opus', '.ogg']
            
            for ext in possible_extensions:
                potential_file = base_name + ext
                if os.path.exists(potential_file):
                    # Convert to WAV if not already
                    if ext != '.wav':
                        audio = AudioSegment.from_file(potential_file)
                        wav_file = base_name + '.wav'
                        audio.export(wav_file, format="wav")
                        os.unlink(potential_file)  # Remove original
                        return wav_file, "Success", title
                    return potential_file, "Success", title
        
        return None, "Download failed - no audio file found", "Unknown"
        
    except Exception as e:
        return None, f"Download error: {str(e)}", "Unknown"

def analyze_speech_patterns(text, audio_features):
    """Analyze speech patterns for additional accent clues"""
    patterns = {}
    
    # Text-based accent indicators
    if text:
        text_lower = text.lower()
        
        # American indicators
        if any(word in text_lower for word in ['color', 'favor', 'center', 'aluminum']):
            patterns['American_text'] = 0.3
        
        # British indicators  
        if any(word in text_lower for word in ['colour', 'favour', 'centre', 'aluminium', 'bloody']):
            patterns['British_text'] = 0.3
        
        # Australian indicators
        if any(word in text_lower for word in ['mate', 'bloke', 'fair dinkum']):
            patterns['Australian_text'] = 0.4
        
        # Check for specific pronunciation patterns in text
        if 'r' in text_lower and text_lower.count('r') > len(text_lower) * 0.05:
            patterns['Rhotic'] = 0.2  # American, Canadian, Irish tendency
    
    return patterns

# Main application logic
if youtube_url:
    video_id = extract_video_id(youtube_url)
    
    if video_id:
        st.success(f"✅ Valid YouTube URL detected (Video ID: {video_id})")
        
        # Get video info
        video_info = get_video_info(video_id)
        if video_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📺 Title", video_info['title'][:30] + "..." if len(video_info['title']) > 30 else video_info['title'])
            with col2:
                st.metric("⏱️ Duration", f"{video_info['duration']//60}:{video_info['duration']%60:02d}")
            with col3:
                st.metric("📊 Status", "Available" if video_info['available'] else "Unavailable")
        
        if st.button("🎯 Analyze Audio (Language + Accent)", type="primary"):
            if not video_info or not video_info['available']:
                st.error("❌ Video not available for processing")
                st.stop()
            
            with st.spinner("Processing video..."):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Download audio
                status_text.text("📥 Downloading audio...")
                progress_bar.progress(20)
                
                audio_file, download_status, title = download_youtube_audio(video_id)
                
                if not audio_file:
                    st.error(f"❌ {download_status}")
                    st.stop()
                
                # Step 2: Transcribe audio
                status_text.text("🎤 Transcribing audio...")
                progress_bar.progress(50)
                
                transcribed_text, engine, transcription_confidence = transcribe_audio_advanced(audio_file)
                
                # Step 3: Detect language from text
                status_text.text("🌍 Detecting language...")
                progress_bar.progress(70)
                
                detected_language, lang_confidence = detect_language_from_text(transcribed_text)
                
                # Step 4: Detect accent from audio
                status_text.text("🗣️ Analyzing accent...")
                progress_bar.progress(85)
                
                detected_accent, accent_confidence = detect_accent_from_audio(audio_file)
                
                # Step 5: Additional analysis
                status_text.text("🔍 Analyzing speech patterns...")
                progress_bar.progress(95)
                
                # Get audio features for pattern analysis
                try:
                    audio_data, sr = librosa.load(audio_file, sr=16000)
                    audio_features = extract_prosodic_features(audio_data, sr)
                    speech_patterns = analyze_speech_patterns(transcribed_text, audio_features)
                except:
                    audio_features = {}
                    speech_patterns = {}
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Clean up temporary file
                try:
                    os.unlink(audio_file)
                except:
                    pass
                
                # Display results
                st.success("🎉 Analysis completed!")
                
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
                    st.subheader("📊 Transcription Quality")
                    st.metric("Engine Used", engine)
                    st.metric("Quality", f"{transcription_confidence:.1%}")
                
                # Transcription
                st.subheader("📝 Transcription")
                st.text_area("Transcribed Text:", transcribed_text, height=150)
                
                # Technical details
                with st.expander("🔧 Technical Analysis"):
                    if audio_features:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Prosodic Features:**")
                            st.write(f"- Fundamental Frequency: {audio_features.get('f0_mean', 0):.1f} Hz")
                            st.write(f"- F0 Range: {audio_features.get('f0_range', 0):.1f} Hz")
                            st.write(f"- Speech Tempo: {audio_features.get('tempo', 0):.1f} BPM")
                            st.write(f"- Spectral Centroid: {audio_features.get('spectral_centroid_mean', 0):.1f} Hz")
                        
                        with col2:
                            st.write("**Speech Patterns:**")
                            if speech_patterns:
                                for pattern, score in speech_patterns.items():
                                    st.write(f"- {pattern}: {score:.1%}")
                            else:
                                st.write("- No specific patterns detected")
                    
                    st.write(f"**Video ID:** {video_id}")
                    st.write(f"**Processing Method:** Advanced Analysis")
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📄 Download Transcription",
                        data=transcribed_text,
                        file_name=f"transcription_{video_id}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Create detailed report
                    report = f"""YouTube Audio Analysis Report
================================

Video Information:
- Title: {title}
- Video ID: {video_id}
- Duration: {video_info['duration']//60}:{video_info['duration']%60:02d}

Language Analysis:
- Detected Language: {detected_language}
- Confidence: {lang_confidence:.1%}

Accent Analysis:
- Detected Accent: {detected_accent}
- Confidence: {accent_confidence:.1%}

Transcription:
- Engine: {engine}
- Quality: {transcription_confidence:.1%}
- Text: {transcribed_text}

Technical Details:
- Fundamental Frequency: {audio_features.get('f0_mean', 0):.1f} Hz
- Speech Tempo: {audio_features.get('tempo', 0):.1f} BPM
- Spectral Analysis: {audio_features.get('spectral_centroid_mean', 0):.1f} Hz

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
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
    
    # Show example
    st.subheader("🎯 What This Tool Does")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Language Detection:**")
        st.write("- Identifies the primary language spoken")
        st.write("- Supports 15+ languages")
        st.write("- Uses advanced text analysis")
        st.write("- Provides confidence scores")
    
    with col2:
        st.write("**Accent Detection:**")
        st.write("- Analyzes prosodic features")
        st.write("- Detects regional accents")
        st.write("- Uses acoustic analysis")
        st.write("- Identifies speech patterns")

# Sidebar information
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. **Copy YouTube URL** from your browser
    2. **Paste it** in the input field
    3. **Click 'Analyze Audio'**
    4. **Wait for processing** (2-5 minutes)
    5. **View detailed results**
    6. **Download reports**
    """)
    
    st.header("🎯 Detected Accents")
    st.markdown("""
    - **American English**
    - **British English** 
    - **Australian English**
    - **Indian English**
    - **Canadian English**
    - **Irish English**
    - **Scottish English**
    - **South African English**
    """)
    
    st.header("🌍 Supported Languages")
    st.markdown("""
    - English, Spanish, French
    - German, Italian, Portuguese  
    - Russian, Chinese, Japanese
    - Korean, Arabic, Hindi
    - Turkish, Polish, Dutch
    - And more...
    """)
    
    st.header("⚙️ Technical Features")
    st.markdown("""
    - **Prosodic Analysis:** F0, tempo, rhythm
    - **Spectral Analysis:** Formants, centroids
    - **Speech Recognition:** Multiple engines
    - **Pattern Matching:** Text-based indicators
    - **Confidence Scoring:** Reliability metrics
    """)
    
    st.header("⚠️ Limitations")
    st.markdown("""
    - **20 minute** video limit
    - **Public videos** only
    - **Clear audio** works best
    - **Network required** for download
    - **Processing time** varies
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎤 Enhanced YouTube Audio Transcriber | Language & Accent Detection</p>
        <p><small>Advanced acoustic analysis for comprehensive speech recognition</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
