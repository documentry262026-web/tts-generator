import os
import subprocess
import soundfile as sf
import numpy as np
import streamlit as st
from kokoro_onnx import Kokoro

# --- Page Config ---
st.set_page_config(
    page_title="Aura TTS Studio | AI Voice Generator",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Glassmorphism UI Design Injection ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #07090e;
    --glass-bg: rgba(18, 22, 36, 0.7);
    --glass-surface: rgba(255, 255, 255, 0.04);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-glow: rgba(99, 102, 241, 0.25);
    --accent-cyan: #38bdf8;
    --accent-indigo: #6366f1;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
}

.stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: 
        radial-gradient(circle at 10% 15%, rgba(99, 102, 241, 0.16) 0%, transparent 40%),
        radial-gradient(circle at 90% 20%, rgba(56, 189, 248, 0.14) 0%, transparent 45%),
        radial-gradient(circle at 50% 85%, rgba(139, 92, 246, 0.14) 0%, transparent 50%),
        #06080e !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* Glass Card Containers */
div[data-testid="stVerticalBlock"] > div:has(div.glass-panel) {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 22px;
    padding: 26px;
    box-shadow: 0 24px 48px -12px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    margin-bottom: 24px;
}

/* Text Area */
.stTextArea textarea {
    background: rgba(12, 15, 26, 0.6) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px) !important;
    padding: 16px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}

.stTextArea textarea:focus {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 2px var(--accent-glow), inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
}

/* Dropdowns & Selects */
div[data-baseweb="select"] > div {
    background: rgba(14, 18, 30, 0.65) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    backdrop-filter: blur(8px) !important;
}

/* Action Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    border-radius: 14px !important;
    padding: 14px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 8px 24px -4px rgba(99, 102, 241, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -2px rgba(99, 102, 241, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
}

/* Download Button */
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.85) 0%, rgba(5, 150, 105, 0.85) 100%) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.45) !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
}

/* Audio Player */
audio {
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
    filter: invert(1) hue-rotate(180deg) brightness(0.9);
}
</style>
""", unsafe_allow_html=True)

# --- Model Downloader & Loader ---
@st.cache_resource
def setup_models():
    # 1. Kokoro ONNX
    if not os.path.exists("kokoro-v1.0.onnx"):
        os.system("wget -q -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    if not os.path.exists("voices-v1.0.bin"):
        os.system("wget -q -nc https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
    
    # 2. Piper Models (Guaranteed Authentic Male Voices)
    if not os.path.exists("rohan.onnx"):
        os.system("wget -q -nc -O rohan.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx")
        os.system("wget -q -nc -O rohan.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json")
    
    if not os.path.exists("male_deep.onnx"):
        os.system("wget -q -nc -O male_deep.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ne/ne_NP/google/medium/ne_NP-google-medium.onnx")
        os.system("wget -q -nc -O male_deep.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/ne/ne_NP/google/medium/ne_NP-google-medium.onnx.json")

    return Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

kokoro_engine = setup_models()

# --- Header ---
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div style="display: inline-flex; align-items: center; justify-content: center; width: 50px; height: 50px; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 16px; margin-bottom: 12px; box-shadow: 0 0 25px rgba(99, 102, 241, 0.25);">
        <span style="font-size: 24px;">🎙️</span>
    </div>
    <h1 style="font-size: 32px; font-weight: 700; margin-bottom: 6px; background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Aura Voice Studio</h1>
    <p style="color: #94a3b8; font-size: 14.5px; margin: 0;">Multi-Model Neural Speech Studio • Unlimited Offline Hindi Narration</p>
</div>
""", unsafe_allow_html=True)

# --- Main Glass Card ---
with st.container():
    st.markdown('<div class="glass-panel"></div>', unsafe_allow_html=True)

    # 1. Preset Style Selector (Auto sets Engine, Voice, and Speed)
    style_preset = st.selectbox(
        "🎬 Content Narration Style (Auto-Adjusts Pace & Character)",
        [
            "🌌 Documentary & Mystery (Deep, Slow, Suspenseful)",
            "📖 Storytelling & Kahani (Atmospheric, Engaging, Natural)",
            "💡 Explainer & Educational (Crisp, Clear, Balanced)",
            "⚡ Shorts & Reels (Punchy, Fast-Paced)",
            "⚙️ Custom / Manual (Configure Yourself)"
        ]
    )

    # Auto settings dictionary
    if "Documentary" in style_preset:
        default_engine_idx = 0
        default_speed = 0.88
    elif "Storytelling" in style_preset:
        default_engine_idx = 0
        default_speed = 0.95
    elif "Explainer" in style_preset:
        default_engine_idx = 1
        default_speed = 1.05
    elif "Shorts" in style_preset:
        default_engine_idx = 1
        default_speed = 1.20
    else:
        default_engine_idx = 0
        default_speed = 1.0

    col_eng, col_voice = st.columns([1, 1], gap="medium")

    with col_eng:
        engine = st.radio(
            "Synthesizer Architecture",
            ["Kokoro ONNX (Natural Flow)", "Piper TTS (Fast & Deep Voice)"],
            index=default_engine_idx,
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
                "Hindi - Rohan (Natural Male Narrator)": "rohan.onnx",
                "Hindi - Deep Heavy (100% Authentic Male)": "male_deep.onnx"
            }

        selected_voice = st.selectbox("Voice Persona", list(voice_options.keys()))
        voice_code = voice_options[selected_voice]

    # Speed Slider (Auto-updated by preset)
    speed = st.slider("Speaking Pace / Speed", min_value=0.5, max_value=1.5, value=default_speed, step=0.05)

    # Script Input Area
    script_text = st.text_area(
        "Voiceover Script (Hindi / Hinglish)",
        placeholder="यहाँ अपनी पूरी 5 से 15 मिनट तक की स्क्रिप्ट पेस्ट करें (कोई करैक्टर लिमिट नहीं है)...",
        height=220
    )

    generate_btn = st.button("Generate High-Definition Voiceover", use_container_width=True)

# --- Generation Logic ---
if generate_btn:
    if not script_text.strip():
        st.warning("कृपया पहले कोई टेक्स्ट दर्ज करें!")
    else:
        with st.spinner("वॉइसओवर तैयार हो रहा है... कृपया कुछ सेकंड प्रतीक्षा करें..."):
            try:
                output_file = "voiceover_output.wav"

                # 1. KOKORO ENGINE (Chunking for long scripts)
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
                        st.error("Text normalize नहीं हो सका!")

                # 2. PIPER ENGINE (Direct Audio Generation)
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

            # --- AUTO CONVERT WAV TO LIGHTWEIGHT MP3 ---
                from pydub import AudioSegment
                
                mp3_output = "voiceover_output.mp3"
                sound = AudioSegment.from_wav(output_file)
                sound.export(mp3_output, format="mp3", bitrate="192k")

                # --- OUTPUT PANEL (GLASSMORPHISM STYLE) ---
                st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 16px; margin-top: 20px;">
                    <div style="font-size: 14px; font-weight: 600; color: #34d399; margin-bottom: 4px;">✓ Voiceover Generated & Converted to MP3 (Lightweight)</div>
                </div>
                """, unsafe_allow_html=True)

                # MP3 Audio Player
                st.audio(mp3_output, format="audio/mp3")

                # MP3 Download Button
                with open(mp3_output, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Voiceover (Lightweight MP3)",
                        data=f,
                        file_name="voiceover.mp3",
                        mime="audio/mp3",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"एरर आया: {e}")
                st.error(f"एरर आया: {e}")
