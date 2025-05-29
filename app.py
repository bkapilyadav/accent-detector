import openai
import streamlit as st

# Add to your configuration section
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

def analyze_accent_with_openai(text, language_code):
    """Use OpenAI to analyze accent from transcribed text"""
    
    if not OPENAI_API_KEY:
        return {"error": "OpenAI API key not configured"}
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"""
    Analyze the following transcribed speech text and identify the likely accent/dialect:
    
    Text: "{text}"
    Detected Language: {language_code}
    
    Please provide:
    1. Most likely accent/dialect (e.g., British, American, Australian, etc.)
    2. Confidence level (0-100%)
    3. Key indicators that suggest this accent
    4. Alternative possibilities if uncertain
    5. Regional specificity if possible
    
    Focus on vocabulary choices, spelling patterns, and linguistic markers.
    Respond in JSON format.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for lower cost
            messages=[
                {"role": "system", "content": "You are an expert linguist specializing in accent and dialect identification from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse the response
        result = response.choices[0].message.content
        
        # Try to parse as JSON, fallback to text analysis if needed
        try:
            import json
            parsed_result = json.loads(result)
            return parsed_result
        except:
            # Fallback: parse text response
            return {"analysis": result, "method": "openai_text"}
            
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}"}

# Updated function to replace the existing detect_accent_patterns
def enhanced_accent_detection(text, language_code):
    """Enhanced accent detection using both rule-based and AI methods"""
    
    # Original rule-based detection
    rule_based_results = detect_accent_patterns(text, language_code)
    
    # OpenAI-powered analysis
    ai_results = analyze_accent_with_openai(text, language_code)
    
    # Combine results
    combined_analysis = {
        "rule_based": rule_based_results,
        "ai_analysis": ai_results,
        "recommendation": "Use both analyses for best accuracy"
    }
    
    return combined_analysis

# Additional: Use OpenAI for transcription (alternative to AssemblyAI)
def transcribe_with_openai(audio_file_path):
    """Alternative transcription using OpenAI Whisper API"""
    
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key required for Whisper transcription")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="json",
                language=None  # Auto-detect language
            )
        
        return {
            "text": transcript.text,
            "language": transcript.language if hasattr(transcript, 'language') else 'unknown',
            "confidence": 0.9,  # Whisper doesn't provide confidence scores
            "method": "openai_whisper"
        }
        
    except Exception as e:
        raise Exception(f"OpenAI Whisper transcription failed: {str(e)}")

# Modified requirements.txt would include:
"""
streamlit
yt-dlp
requests
openai>=1.0.0
"""

# Modified secrets configuration:
"""
OPENAI_API_KEY = "your-openai-api-key-here"
ASSEMBLYAI_API_KEY = "your-assemblyai-key-here"  # Keep both for options
"""
