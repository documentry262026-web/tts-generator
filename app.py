import streamlit as st
import asyncio
import os
import soundfile as sf
import numpy as np
import edge_tts
from kokoro_onnx import Kokoro

st.set_page_config(page_title="Hindi TTS Studio", page_icon="🎙️", layout="centered")

# Download Kokoro weights on the first run if needed
@st.cache_resource
def load_kokoro():
    onnx_file = "kokoro-v1.0.onnx"
    voices_file = "voices-v1.0.bin"
    if not os.path.exists(onnx_file):
        os.system("wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    if not os.path.exists(voices_file):
        os.system("wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
    return Kokoro(onnx_file, voices_file)

st.title("🎙️ Hindi TTS Studio")
st.caption("Free, unlimited text-to-speech with zero character limits.")

# UI Controls
engine = st.radio(
    "Choose Engine",
    ["Edge-TTS (Ultra-Natural Studio Quality)", "Kokoro ONNX (Offline Neural)"],
    horizontal=True
)

if "Edge-TTS" in engine:
    voice_map = {
        "Hindi - Madhur (Deep Male)": "hi-IN-MadhurNeural",
        "Hindi - Swara (Natural Female)": "hi-IN-SwaraNeural"
    }
else:
    voice_map = {
        "Hindi - Omega (Deep Male)": "hm_omega",
        "Hindi - Psi (Male)": "hm_psi",
        "Hindi - Alpha (Female)": "hf_alpha",
        "Hindi - Beta (Female)": "hf_beta"
    }

selected_label = st.selectbox("Select Character / Voice", list(voice_map.keys()))
voice_code = voice_map[selected_label]

speed = st.slider("Speaking Pace / Speed", min_value=0.5, max_value=1.5, value=1.0, step=0.05)

script_text = st.text_area(
    "Enter Your Hindi / Hinglish Script",
    placeholder="यहाँ अपना पूरा वॉइसओवर स्क्रिप्ट पेस्ट करें (5-10 मिनट लंबा टेक्स्ट भी चलेगा)...",
    height=220
)

# Generation Logic
if st.button("Generate Voiceover", type="primary", use_container_width=True):
    if not script_text.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Generating audio, please wait..."):
            try:
                if "Edge-TTS" in engine:
                    rate_str = f"{int((speed - 1.0) * 100):+d}%"
                    communicate = edge_tts.Communicate(script_text, voice_code, rate=rate_str)
                    output_file = "voiceover.mp3"
                    asyncio.run(communicate.save(output_file))
                    
                    st.success("Generated successfully!")
                    st.audio(output_file, format="audio/mp3")
                    with open(output_file, "rb") as f:
                        st.download_button("⬇️ Download Audio (MP3)", f, file_name="hindi_voiceover.mp3", mime="audio/mp3", use_container_width=True)

                else:
                    kokoro = load_kokoro()
                    # Chunk long text by sentence terminators
                    delimiters = ["।", ".", "!", "?", "\n"]
                    sentences = [script_text]
                    for d in delimiters:
                        temp = []
                        for s in sentences:
                            temp.extend(s.split(d))
                        sentences = temp
                    sentences = [s.strip() for s in sentences if s.strip()]

                    all_audio = []
                    sr = 24000
                    for seg in sentences:
                        samples, sr = kokoro.create(seg, voice=voice_code, speed=speed, lang="hi")
                        all_audio.append(samples)

                    if all_audio:
                        final_audio = np.concatenate(all_audio)
                        output_file = "voiceover.wav"
                        sf.write(output_file, final_audio, sr)
                        
                        st.success("Generated successfully!")
                        st.audio(output_file, format="audio/wav")
                        with open(output_file, "rb") as f:
                            st.download_button("⬇️ Download Audio (WAV)", f, file_name="hindi_voiceover.wav", mime="audio/wav", use_container_width=True)
                    else:
                        st.error("Could not process sentences. Please check script formatting.")

            except Exception as e:
                st.error(f"Error during generation: {e}")
