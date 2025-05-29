import openai
import streamlit as st
import re
import json
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Accent Detection App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .feature-found {
        background: #e8f5e8;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Get API key from secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

def detect_accent_patterns(text: str, language_code: str) -> Dict:
    """Rule-based accent detection using linguistic patterns"""
    
    text_lower = text.lower()
    detected_features = []
    accent_scores = {}
    
    if language_code.startswith('en'):  # English language variants
        
        # American English indicators
        american_indicators = [
            ('color', 'favor', 'honor'),  # -or spellings
            ('ize', 'ization'),  # -ize endings
            ('center', 'theater'),  # -er endings
            ('mom', 'gotten', 'fall'),  # vocabulary
            ('elevator', 'apartment', 'candy'),
            ('truck', 'gas', 'sidewalk'),
            ('y\'all', 'gonna', 'wanna')  # contractions
        ]
        
        # British English indicators
        british_indicators = [
            ('colour', 'favour', 'honour'),  # -our spellings
            ('ise', 'isation'),  # -ise endings
            ('centre', 'theatre'),  # -re endings
            ('mum', 'autumn', 'lift'),  # vocabulary
            ('flat', 'biscuit', 'rubber'),
            ('lorry', 'petrol', 'pavement'),
            ('brilliant', 'mate', 'cheers'),
            ('whilst', 'amongst')  # formal variants
        ]
        
        # Australian English indicators
        australian_indicators = [
            ('arvo', 'barbie', 'brekkie'),  # abbreviations
            ('mate', 'sheila', 'bloke'),
            ('fair dinkum', 'no worries', 'too right'),
            ('uni', 'servo', 'bottle-o'),
            ('crikey', 'strewth')
        ]
        
        # Canadian English indicators
        canadian_indicators = [
            ('eh', 'about', 'house'),  # 'about' pronunciation markers in text
            ('toque', 'hydro', 'washroom'),
            ('double-double', 'loonie', 'toonie'),
            ('chesterfield', 'runners')
        ]
        
        # South African English indicators
        south_african_indicators = [
            ('braai', 'boet', 'howzit'),
            ('lekker', 'shame', 'eish'),
            ('robot', 'bakkie', 'kombi'),  # robot = traffic light
            ('now now', 'just now')
        ]
        
        # Score each accent
        accent_patterns = {
            'American': american_indicators,
            'British': british_indicators,
            'Australian': australian_indicators,
            'Canadian': canadian_indicators,
            'South African': south_african_indicators
        }
        
        for accent, patterns in accent_patterns.items():
            score = 0
            found_features = []
            
            for pattern_group in patterns:
                for pattern in pattern_group:
                    if pattern in text_lower:
                        score += 1
                        found_features.append(pattern)
            
            if score > 0:
                accent_scores[accent] = {
                    'score': score,
                    'features': found_features,
                    'confidence': min(score * 15, 85)  # Cap at 85%
                }
    
    # Handle other languages
    elif language_code.startswith('es'):  # Spanish
        spanish_regions = {
            'Mexican': ['órale', 'güey', 'chido', 'padre'],
            'Argentinian': ['che', 'boludo', 'bárbaro'],
            'Spanish (Spain)': ['vale', 'tío', 'guay', 'joder'],
            'Colombian': ['parce', 'chimba', 'bacano']
        }
        
        for region, indicators in spanish_regions.items():
            score = sum(1 for word in indicators if word in text_lower)
            if score > 0:
                accent_scores[region] = {
                    'score': score,
                    'features': [w for w in indicators if w in text_lower],
                    'confidence': min(score * 20, 80)
                }
    
    elif language_code.startswith('fr'):  # French
        french_regions = {
            'French (France)': ['putain', 'bordel', 'coucou'],
            'Canadian French': ['tabarnak', 'câlisse', 'osti'],
            'Belgian French': ['nonante', 'septante']
        }
        
        for region, indicators in french_regions.items():
            score = sum(1 for word in indicators if word in text_lower)
            if score > 0:
                accent_scores[region] = {
                    'score': score,
                    'features': [w for w in indicators if w in text_lower],
                    'confidence': min(score * 25, 80)
                }
    
    # Return results
    if accent_scores:
        top_accent = max(accent_scores.items(), key=lambda x: x[1]['score'])
        return {
            'detected_accent': top_accent[0],
            'confidence': top_accent[1]['confidence'],
            'features_found': top_accent[1]['features'],
            'all_scores': accent_scores,
            'method': 'rule_based'
        }
    else:
        return {
            'detected_accent': 'Unknown',
            'confidence': 0,
            'features_found': [],
            'all_scores': {},
            'method': 'rule_based'
        }

def analyze_accent_with_openai(text: str, language_code: str) -> Dict:
    """Use OpenAI to analyze accent from transcribed text"""
    
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured"}
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Language-specific prompts
    language_context = {
        'en': "English (considering American, British, Australian, Canadian, South African, Irish, etc.)",
        'es': "Spanish (considering Mexican, Argentinian, Spanish, Colombian, etc.)",
        'fr': "French (considering French, Canadian French, Belgian French, etc.)",
        'de': "German (considering German, Austrian, Swiss German, etc.)",
        'it': "Italian (considering standard Italian, regional variants)",
        'pt': "Portuguese (considering Brazilian, European Portuguese)"
    }
    
    lang_context = language_context.get(language_code[:2], f"Language code: {language_code}")
    
    prompt = f"""
    Analyze the following transcribed speech text and identify the likely accent/dialect for {lang_context}:
    
    Text: "{text}"
    
    Please provide a JSON response with:
    {{
        "primary_accent": "most likely accent/dialect",
        "confidence": confidence_percentage_as_number,
        "key_indicators": ["list", "of", "specific", "words", "or", "phrases"],
        "alternative_accents": [
            {{"accent": "alternative1", "probability": percentage}},
            {{"accent": "alternative2", "probability": percentage}}
        ],
        "regional_specificity": "more specific region if identifiable",
        "reasoning": "brief explanation of the analysis"
    }}
    
    Focus on vocabulary choices, spelling patterns, idiomatic expressions, and linguistic markers.
    Be specific about regional variants when possible.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert linguist specializing in accent and dialect identification from text. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean the response to ensure valid JSON
        if result.startswith('```json'):
            result = result[7:-3]
        elif result.startswith('```'):
            result = result[3:-3]
        
        try:
            parsed_result = json.loads(result)
            parsed_result['method'] = 'openai_gpt4'
            return parsed_result
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "primary_accent": "Analysis available",
                "confidence": 50,
                "key_indicators": [],
                "alternative_accents": [],
                "reasoning": result,
                "method": "openai_text_fallback"
            }
            
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}"}

def phonetic_accent_analysis(text: str, language_code: str) -> Dict:
    """Analyze phonetic patterns that might indicate accent"""
    
    phonetic_indicators = {}
    
    if language_code.startswith('en'):
        # Look for phonetic spelling patterns
        patterns = {
            'r_dropping': re.findall(r'\b\w*ah\b|\b\w*eh\b', text.lower()),  # Non-rhotic accents
            'r_adding': re.findall(r'\bidear\b|\bsofa[hr]\b', text.lower()),  # Intrusive R
            'th_substitution': re.findall(r'\bdis\b|\bdat\b|\bdose\b', text.lower()),
            'vowel_shifts': re.findall(r'\bcah\b|\bpahk\b|\bbahth\b', text.lower()),  # Boston/NY patterns
        }
        
        phonetic_indicators = {k: len(v) for k, v in patterns.items() if v}
    
    return {
        'phonetic_patterns': phonetic_indicators,
        'method': 'phonetic_analysis'
    }

def enhanced_accent_detection(text: str, language_code: str) -> Dict:
    """Enhanced accent detection using multiple methods"""
    
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'text_analyzed': text[:100] + "..." if len(text) > 100 else text,
        'language_code': language_code
    }
    
    # Method 1: Rule-based detection
    with st.spinner("Analyzing linguistic patterns..."):
        rule_based = detect_accent_patterns(text, language_code)
        results['rule_based'] = rule_based
    
    # Method 2: Phonetic analysis
    phonetic = phonetic_accent_analysis(text, language_code)
    results['phonetic'] = phonetic
    
    # Method 3: AI-powered analysis (if API key available)
    if OPENAI_API_KEY:
        with st.spinner("Getting AI accent analysis..."):
            ai_analysis = analyze_accent_with_openai(text, language_code)
            results['ai_analysis'] = ai_analysis
    else:
        results['ai_analysis'] = {"error": "OpenAI API key not configured"}
    
    # Combine results for final recommendation
    final_accent = "Unknown"
    final_confidence = 0
    reasoning = []
    
    # Priority: AI analysis > Rule-based > Default
    if 'ai_analysis' in results and 'primary_accent' in results['ai_analysis']:
        final_accent = results['ai_analysis']['primary_accent']
        final_confidence = results['ai_analysis'].get('confidence', 50)
        reasoning.append(f"AI analysis: {final_accent}")
    
    elif rule_based['detected_accent'] != 'Unknown':
        final_accent = rule_based['detected_accent']
        final_confidence = rule_based['confidence']
        reasoning.append(f"Pattern matching: {final_accent}")
    
    # Adjust confidence based on text length
    text_length_factor = min(len(text.split()) / 50, 1.0)  # Longer text = higher confidence
    final_confidence = int(final_confidence * text_length_factor)
    
    results['final_analysis'] = {
        'detected_accent': final_accent,
        'confidence': final_confidence,
        'reasoning': reasoning,
        'text_length_words': len(text.split()),
        'reliability': 'High' if final_confidence > 70 else 'Medium' if final_confidence > 40 else 'Low'
    }
    
    return results

def transcribe_with_openai(audio_file) -> Dict:
    """Transcribe audio using OpenAI Whisper API"""
    
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key required for Whisper transcription")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as audio_file_obj:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_obj,
                    response_format="verbose_json",  # Get more detailed response
                    language=None,  # Auto-detect language
                    temperature=0  # More deterministic results
                )
            
            # Extract language information
            detected_language = getattr(transcript, 'language', 'unknown')
            
            return {
                "text": transcript.text,
                "language": detected_language,
                "confidence": 0.9,  # Whisper doesn't provide confidence scores
                "duration": getattr(transcript, 'duration', 0),
                "segments": getattr(transcript, 'segments', []),
                "method": "openai_whisper"
            }
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
    except Exception as e:
        raise Exception(f"OpenAI Whisper transcription failed: {str(e)}")

def display_accent_results(results: Dict):
    """Display accent detection results in Streamlit"""
    
    st.subheader("🎭 Accent Analysis Results")
    
    # Final analysis
    final = results.get('final_analysis', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detected Accent", final.get('detected_accent', 'Unknown'))
    with col2:
        st.metric("Confidence", f"{final.get('confidence', 0)}%")
    with col3:
        st.metric("Reliability", final.get('reliability', 'Unknown'))
    
    # Text analysis info
    st.info(f"📝 Analyzed {final.get('text_length_words', 0)} words at {results.get('timestamp', 'unknown time')}")
    
    # Detailed breakdown
    with st.expander("📊 Detailed Analysis", expanded=False):
        
        # Rule-based results
        if 'rule_based' in results:
            st.write("**Pattern-Based Analysis:**")
            rb = results['rule_based']
            if rb['detected_accent'] != 'Unknown':
                st.write(f"- Detected: {rb['detected_accent']} ({rb['confidence']}%)")
                if rb['features_found']:
                    st.write("- Key features found:")
                    for feature in rb['features_found']:
                        st.markdown(f"  <span class='feature-found'>{feature}</span>", unsafe_allow_html=True)
            else:
                st.write("- No clear patterns detected")
        
        # AI analysis results
        if 'ai_analysis' in results and 'primary_accent' in results['ai_analysis']:
            st.write("**AI Analysis:**")
            ai = results['ai_analysis']
            st.write(f"- Primary: {ai['primary_accent']} ({ai.get('confidence', 0)}%)")
            
            if ai.get('key_indicators'):
                st.write(f"- Indicators: {', '.join(ai['key_indicators'])}")
            
            if ai.get('alternative_accents'):
                st.write("- Alternatives:")
                for alt in ai['alternative_accents'][:3]:  # Show top 3
                    st.write(f"  - {alt.get('accent', 'Unknown')} ({alt.get('probability', 0)}%)")
            
            if ai.get('reasoning'):
                st.write(f"- Reasoning: {ai['reasoning']}")
        
        # Phonetic analysis
        if 'phonetic' in results and results['phonetic']['phonetic_patterns']:
            st.write("**Phonetic Patterns:**")
            for pattern, count in results['phonetic']['phonetic_patterns'].items():
                st.write(f"- {pattern.replace('_', ' ').title()}: {count} occurrences")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Accent Detection App</h1>
        <p>Analyze accents and dialects from text or speech using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        if OPENAI_API_KEY:
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key missing")
            st.info("Add your OpenAI API key to Streamlit secrets to enable AI analysis and audio transcription.")
        
        # Language selection
        language_options = {
            "English": "en-US",
            "Spanish": "es-ES", 
            "French": "fr-FR",
            "German": "de-DE",
            "Italian": "it-IT",
            "Portuguese": "pt-PT"
        }
        
        selected_language = st.selectbox(
            "Select Language:",
            list(language_options.keys())
        )
        language_code = language_options[selected_language]
        
        st.divider()
        
        # Sample texts
        st.subheader("📝 Sample Texts")
        sample_texts = {
            "American English": "I'm gonna grab some coffee and candy from the elevator, then head to my apartment. Y'all want anything?",
            "British English": "I'll pop to the lift and get some sweets, then head back to my flat. Cheers mate, brilliant!",
            "Australian English": "G'day mate! Let's have a barbie this arvo down at the servo, it'll be bonzer! No worries!",
            "Canadian English": "It's aboot time we head to the washroom, eh? Don't forget your toque, it's cold out there!",
            "Mexican Spanish": "¡Órale güey! Vamos a echar la chela, está muy chido el ambiente aquí, ¿no?",
            "French (France)": "Salut ! On va prendre un café au bistrot, c'est chouette cette ambiance, non ?",
        }
        
        for accent, text in sample_texts.items():
            if st.button(f"Use {accent} sample", key=f"sample_{accent}"):
                st.session_state.sample_text = text
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📝 Text Analysis", "🎙️ Audio Analysis"])
    
    with tab1:
        st.header("Text-based Accent Detection")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.get('sample_text', ''),
            height=150,
            placeholder="Type or paste text here..."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button("🔍 Analyze Accent", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.sample_text = ""
            st.rerun()
        
        if analyze_button:
            if text_input.strip():
                with st.spinner("Analyzing accent..."):
                    results = enhanced_accent_detection(text_input, language_code)
                    display_accent_results(results)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.header("Audio-based Accent Detection")
        
        if not OPENAI_API_KEY:
            st.warning("⚠️ OpenAI API key required for audio transcription. Please add it to your Streamlit secrets.")
        else:
            st.info("🎤 Upload an audio file to transcribe and analyze the accent.")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
                help="Supported formats: WAV, MP3, M4A, OGG, FLAC"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("🎙️ Transcribe & Analyze", type="primary"):
                    try:
                        with st.spinner("Transcribing audio..."):
                            transcription_result = transcribe_with_openai(uploaded_file)
                        
                        st.success("✅ Transcription completed!")
                        
                        # Display transcription
                        st.subheader("📝 Transcription")
                        st.write(f"**Detected Language:** {transcription_result.get('language', 'Unknown')}")
                        st.write(f"**Text:** {transcription_result.get('text', '')}")
                        
                        # Analyze accent from transcription
                        if transcription_result.get('text'):
                            with st.spinner("Analyzing accent from transcription..."):
                                accent_results = enhanced_accent_detection(
                                    transcription_result['text'], 
                                    transcription_result.get('language', language_code)
                                )
                                display_accent_results(accent_results)
                    
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {str(e)}")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎭 Accent Detection App | Powered by OpenAI GPT-4 & Whisper</p>
        <p><small>Analyzes linguistic patterns, vocabulary, and regional markers to identify accents and dialects</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
