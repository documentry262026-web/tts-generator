import streamlit as st
import os
import subprocess
import soundfile as sf
import numpy as np
from kokoro_onnx import Kokoro

st.set_page_config(page_title="Piper & Kokoro TTS Studio", page_icon="🎙️", layout="centered")

# --- ऑटो डाउनलोड मॉडल्स (Cached) ---
@st.cache_resource
def setup_models():
    # 1. Kokoro Models
    if not os.path.exists("kokoro-v1.0.onnx"):
        os.system("wget -q -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    if not os.path.exists("voices-v1.0.bin"):
        os.system("wget -q -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
    
    # 2. Piper Models (Hindi Rohan + Urdu/Hindi Fasih Deep Male)
    if not os.path.exists("rohan.onnx"):
        os.system("wget -q -nc -O rohan.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx")
        os.system("wget -q -nc -O rohan.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json")
    if not os.path.exists("fasih.onnx"):
        os.system("wget -q -nc -O fasih.onnx https://huggingface.co/IhorShevchuk/piper-voice-ur-fasih/resolve/main/ur_PK-male-medium.onnx")
        os.system("wget -q -nc -O fasih.onnx.json https://huggingface.co/IhorShevchuk/piper-voice-ur-fasih/resolve/main/ur_PK-male-medium.onnx.json")

    return Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

kokoro_engine = setup_models()

st.title("🎙️ Local Neural TTS Studio")
st.caption("100% Free & Unlimited | Powered by Piper TTS & Kokoro ONNX")

# Engine चुनना
engine = st.radio("Choose Model Engine", ["Kokoro ONNX (Natural Flow)", "Piper TTS (Fast & Deep Voice)"], horizontal=True)

# वॉइस कैरेक्टर्स
if "Kokoro" in engine:
    voice_options = {
        "Hindi - Omega (गहरी आवाज़ / Deep Male)": "hm_omega",
        "Hindi - Psi (शांत आवाज़ / Calm Male)": "hm_psi",
        "Hindi - Alpha (फीमेल नैरेटर / Female 1)": "hf_alpha",
        "Hindi - Beta (एक्सप्रेसिव फीमेल / Female 2)": "hf_beta"
    }
else:
    voice_options = {
        "Hindi - Rohan (डॉक्यूमेंट्री मेल / Documentary Male)": "rohan.onnx",
        "Hindi/Urdu - Fasih (अल्ट्रा डीप मेल / Ultra Deep Male)": "fasih.onnx"
    }

selected_voice = st.selectbox("Select Character / Voice", list(voice_options.keys()))
voice_code = voice_options[selected_voice]

speed = st.slider("Speaking Speed (Pace)", min_value=0.5, max_value=1.5, value=1.0, step=0.05)

script_text = st.text_area(
    "Enter Your Hindi / Hinglish Script",
    placeholder="यहाँ अपनी पूरी 5-10 मिनट की स्क्रिप्ट पेस्ट करें...",
    height=240
)

# जनरेशन बटन
if st.button("Generate Voiceover", type="primary", use_container_width=True):
    if not script_text.strip():
        st.warning("कृपया पहले कोई टेक्स्ट दर्ज करें!")
    else:
        with st.spinner("वॉइसओवर तैयार हो रहा है... कृपया प्रतीक्षा करें..."):
            try:
                output_file = "voiceover_output.wav"

                # 1. KOKORO ONNX PROCESSING
                if "Kokoro" in engine:
                    # लम्बी स्क्रिप्ट के लिए सेंटेंस चंकिंग
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
                        samples, sr = kokoro_engine.create(seg, voice=voice_code, speed=speed, lang="hi")
                        all_audio.append(samples)

                    if all_audio:
                        final_audio = np.concatenate(all_audio)
                        sf.write(output_file, final_audio, sr)
                    else:
                        st.error("टेक्स्ट प्रोसेस नहीं हो सका!")

                # 2. PIPER TTS PROCESSING
                else:
                    # Piper में speed को length_scale से सेट किया जाता है (1.0 = नॉर्मल, >1.0 धीमा/गहरा)
                    length_scale = round(1.0 / speed, 2)
                    cmd = [
                        "piper",
                        "--model", voice_code,
                        "--output_file", output_file,
                        "--length_scale", str(length_scale)
                    ]
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    process.communicate(input=script_text.encode("utf-8"))

                # ऑडियो प्लेयर और डाउनलोड
                st.success("वॉइसओवर सफलतापूर्वक तैयार हो गया!")
                st.audio(output_file, format="audio/wav")
                with open(output_file, "rb") as f:
                    st.download_button(
                        "⬇️ Download Audio (WAV)",
                        f,
                        file_name="voiceover.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"एरर आया: {e}")
