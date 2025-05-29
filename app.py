import streamlit as st
import tempfile
import os
import yt_dlp
import openai

# Get OpenAI API key from Streamlit Secrets or Environment
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

st.title("🎤 English Accent Detector")
st.write("Enter a **public video URL** (MP4, Loom, YouTube) to analyze the speaker's English accent.")

video_url = st.text_input("Video URL")

def download_audio(url):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "audio.mp3")
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return audio_path, None
    except Exception as e:
        return None, str(e)

def transcribe_audio_whisper(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

def classify_accent_gpt(transcript):
    prompt = f"""
You are an accent detection AI. Given this transcript of spoken English, determine:
1. The likely English accent (British, American, Australian, etc.).
2. Confidence score (0-100%).
3. A short explanation of your reasoning.

Transcript:
{transcript}

Respond in this format:
Accent: <detected accent>
Confidence: <confidence score>%
Explanation: <short explanation>
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

if st.button("Analyze"):
    if not video_url:
        st.warning("Please enter a video URL.")
    else:
        with st.spinner("Downloading audio..."):
            audio_path, error = download_audio(video_url)
            if error:
                st.error(f"Download error: {error}")
            else:
                st.audio(audio_path)
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio_whisper(audio_path)
                st.subheader("📜 Transcript")
                st.write(transcript)
                with st.spinner("Classifying accent..."):
                    result = classify_accent_gpt(transcript)
                st.subheader("🌍 Detected Accent")
                st.markdown(result)
