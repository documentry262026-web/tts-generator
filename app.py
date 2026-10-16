import os
import subprocess
import soundfile as sf
import numpy as np
import streamlit as st
from kokoro_onnx import Kokoro

# --- Page Setup ---
st.set_page_config(
    page_title="Aura TTS | Neural Speech Studio",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Glassmorphism UI Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Root Variables */
:root {
    --bg-primary: #0a0b10;
    --glass-bg: rgba(18, 20, 32, 0.65);
    --glass-surface: rgba(255, 255, 255, 0.04);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-border-focus: rgba(99, 102, 241, 0.45);
    --accent-glow: rgba(99, 102, 241, 0.18);
    --accent-cyan: #38bdf8;
    --accent-indigo: #6366f1;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

/* Base Body Background with Ambient Lighting */
.stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.12) 0%, transparent 45%),
        radial-gradient(circle at 50% 80%, rgba(139, 92, 246, 0.12) 0%, transparent 50%),
        #06080e !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* Floating Glass Cards */
div[data-testid="stVerticalBlock"] > div:has(div.glass-card-header) {
    background: var(--glass-bg);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), 
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

/* Glass Inputs (Text Area, Inputs) */
.stTextArea textarea {
    background: rgba(12, 14, 24, 0.55) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    padding: 16px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    transition: all 0.25s ease-in-out !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 2px var(--accent-glow), inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

/* Glass Dropdowns and Selectboxes */
div[data-baseweb="select"] > div {
    background: rgba(12, 14, 24, 0.6) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(255, 255, 255, 0.18) !important;
}

/* Radio Button Pills / Custom Options */
div[role="radiogroup"] {
    background: rgba(15, 18, 30, 0.5);
    border: 1px solid var(--glass-border);
    padding: 6px 10px;
    border-radius: 14px;
    backdrop-filter: blur(12px);
    gap: 15px;
}

/* Range Slider Glass Styling */
div[data-testid="stSlider"] > div {
    padding-top: 6px;
}

/* Glossy Action Buttons */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.85) 0%, rgba(139, 92, 246, 0.85) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 8px 24px -6px rgba(99, 102, 241, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -4px rgba(99, 102, 241, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0);
}

/* Download Button Styling */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.8) 0%, rgba(5, 150, 105, 0.8) 100%) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 20px -6px rgba(16, 185, 129, 0.45) !important;
    padding: 12px 20px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 24px -4px rgba(16, 185, 129, 0.6) !important;
}

/* Audio Player Skinning */
audio {
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
    filter: invert(1) hue-rotate(180deg) brightness(0.9);
}

/* Text Headers & Labels */
h1, h2, h3, h4 {
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

.stMarkdown label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Model Downloader & Cache ---
@st.cache_resource
def setup_models():
    # 1. Kokoro ONNX Weights
    if not os.path.exists("kokoro-v1.0.onnx"):
        os.system("wget -q -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    if not os.path.exists("voices-v1.0.bin"):
        os.system("wget -q -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
    
    # 2. Piper Models (Direct Official Raw Links to Fix Voice Collisions)
    # Rohan (Hindi Male - Clear Narration)
    if not os.path.exists("rohan.onnx"):
        os.system("wget -q -nc -O rohan.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx")
        os.system("wget -q -nc -O rohan.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json")
    
    # Fasih (Urdu / Hindi South Asian Male - Low Pitch Cinematic Voice)
    if not os.path.exists("fasih.onnx"):
        os.system("wget -q -nc -O fasih.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx")
        os.system("wget -q -nc -O fasih.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx.json")

    return Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

kokoro_engine = setup_models()

# --- Header Section ---
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 16px; margin-bottom: 12px; box-shadow: 0 0 20px rgba(99, 102, 241, 0.2);">
        <span style="font-size: 22px;">✨</span>
    </div>
    <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 6px; background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Aura Voice Studio</h1>
    <p style="color: #94a3b8; font-size: 14.5px; margin: 0;">Unbounded Neural Text-to-Speech Engine • 100% Offline</p>
</div>
""", unsafe_allow_html=True)

# --- Main Glass Card Container ---
with st.container():
    st.markdown('<div class="glass-card-header"></div>', unsafe_allow_html=True)
    
    # Engine Selection
    col_eng, col_voice = st.columns([1, 1], gap="medium")
    
    with col_eng:
        engine = st.radio(
            "Synthesizer Architecture",
            ["Kokoro ONNX (Natural Flow)", "Piper TTS (Fast & Deep Voice)"],
            horizontal=False
        )

    with col_voice:
        if "Kokoro" in engine:
            voice_options = {
                "Hindi - Omega (Deep Male / गहरी आवाज़)": "hm_omega",
                "Hindi - Psi (Calm Male / शांत आवाज़)": "hm_psi",
                "Hindi - Alpha (Narrator Female / मुख्य आवाज़)": "hf_alpha",
                "Hindi - Beta (Expressive Female / भावुक आवाज़)": "hf_beta"
            }
        else:
            voice_options = {
                "Hindi - Rohan (Clear Male Documentary)": "rohan.onnx",
                "Hindi/Urdu - Fasih (Ultra Deep Male Narrative)": "fasih.onnx"
            }

        selected_voice = st.selectbox("Voice Persona", list(voice_options.keys()))
        voice_code = voice_options[selected_voice]

    # Speaking Pace Slider
    speed = st.slider("Tempo & Articulation Speed", min_value=0.5, max_value=1.5, value=1.0, step=0.05)

    # Script Input Box
    script_text = st.text_area(
        "Script Content",
        placeholder="यहाँ अपनी पूरी डॉक्यूमेंट्री या वीडियो स्क्रिप्ट पेस्ट करें (5-10 मिनट लंबा टेक्स्ट भी बिना रुकावट चलेगा)...",
        height=210
    )

    # Action Trigger
    generate_btn = st.button("Generate Neural Audio", use_container_width=True)

# --- Audio Processing Logic ---
if generate_btn:
    if not script_text.strip():
        st.warning("Please provide input text to synthesize.")
    else:
        with st.spinner("Synthesizing audio buffers across local neural layers..."):
            try:
                output_file = "voiceover_output.wav"

                # 1. KOKORO PIPELINE (Chunk-based for unlimited lengths)
                if "Kokoro" in engine:
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
                        st.error("Text normalization could not produce valid speech tokens.")

                # 2. PIPER PIPELINE (Direct Model File Binding)
                else:
                    length_scale = round(1.0 / speed, 2)
                    cmd = [
                        "piper",
                        "--model", voice_code,
                        "--output_file", output_file,
                        "--length_scale", str(length_scale)
                    ]
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    process.communicate(input=script_text.encode("utf-8"))

                # Output Section Inside a Focused Glass Panel
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 16px; padding: 18px; margin-top: 20px;">
                    <div style="font-size: 14px; font-weight: 600; color: #34d399; margin-bottom: 8px;">✓ Synthesis Complete</div>
                </div>
                """, unsafe_allow_html=True)

                st.audio(output_file, format="audio/wav")
                
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download High-Definition Master (WAV)",
                        data=f,
                        file_name="neural_voiceover.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Synthesis Pipeline Exception: {e}")
