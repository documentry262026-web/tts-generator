import os
import asyncio
import numpy as np
import soundfile as sf
import gradio as gr
import edge_tts
from kokoro_onnx import Kokoro

# --- Helper: Download Kokoro Model Weights if missing ---
def ensure_kokoro_files():
    onnx_file = "kokoro-v1.0.onnx"
    voices_file = "voices-v1.0.bin"
    if not os.path.exists(onnx_file):
        os.system("wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
    if not os.path.exists(voices_file):
        os.system("wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")

# --- Generation Function ---
async def generate_voice(text, engine, voice, speed):
    if not text.strip():
        return None, "Please enter some text."

    output_path = "output_voiceover.wav"
    
    # 1. Edge-TTS Engine
    if "Edge-TTS" in engine:
        rate_str = f"{int((speed - 1.0) * 100):+d}%"
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        mp3_path = "output_voiceover.mp3"
        await communicate.save(mp3_path)
        return mp3_path, "Generated successfully with Edge-TTS!"

    # 2. Kokoro ONNX Engine (with Sentence Chunking for long scripts)
    elif "Kokoro" in engine:
        ensure_kokoro_files()
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        
        # Split text into chunks using Hindi and English sentence terminators
        delimiters = ["।", ".", "!", "?", "\n"]
        sentences = [text]
        for d in delimiters:
            temp = []
            for s in sentences:
                temp.extend(s.split(d))
            sentences = temp
        
        sentences = [s.strip() for s in sentences if s.strip()]
        all_samples = []
        sample_rate = 24000
        
        for segment in sentences:
            try:
                samples, sr = kokoro.create(segment, voice=voice, speed=speed, lang="hi")
                all_samples.append(samples)
                sample_rate = sr
            except Exception as e:
                continue
                
        if not all_samples:
            return None, "Failed to generate audio. Check your text format."
            
        combined_audio = np.concatenate(all_samples)
        sf.write(output_path, combined_audio, sample_rate)
        return output_path, "Generated successfully with Kokoro!"

# Wrapper for Gradio sync calls
def run_tts(text, engine, voice, speed):
    return asyncio.run(generate_voice(text, engine, voice, speed))

# Dynamic Voice Selector based on chosen Engine
def update_voice_choices(engine):
    if "Edge-TTS" in engine:
        return gr.Dropdown(
            choices=[
                ("Hindi - Madhur (Deep Male)", "hi-IN-MadhurNeural"),
                ("Hindi - Swara (Female)", "hi-IN-SwaraNeural")
            ],
            value="hi-IN-MadhurNeural",
            label="Voice Character"
        )
    else:
        return gr.Dropdown(
            choices=[
                ("Hindi - Omega (Deep Male)", "hm_omega"),
                ("Hindi - Psi (Male)", "hm_psi"),
                ("Hindi - Alpha (Female)", "hf_alpha"),
                ("Hindi - Beta (Female)", "hf_beta")
            ],
            value="hm_omega",
            label="Voice Character"
        )

# --- Gradio UI Layout ---
with gr.Blocks(title="Free Hindi TTS Studio") as demo:
    gr.Markdown("# 🎙️ Free Unlimited Hindi Voiceover Studio")
    gr.Markdown("Zero limits, no character caps. Designed for long documentaries, stories, and videos.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Your Script (Hindi / Hinglish)",
                placeholder="यहाँ अपना पूरा वॉइसओवर स्क्रिप्ट पेस्ट करें (5 से 10 मिनट लंबा टेक्स्ट भी चलेगा)...",
                lines=10
            )
            engine_selector = gr.Radio(
                choices=["Edge-TTS (Ultra-Natural Studio Quality)", "Kokoro ONNX (Offline Local Neural)"],
                value="Edge-TTS (Ultra-Natural Studio Quality)",
                label="TTS Engine"
            )
            voice_selector = gr.Dropdown(
                choices=[
                    ("Hindi - Madhur (Deep Male)", "hi-IN-MadhurNeural"),
                    ("Hindi - Swara (Female)", "hi-IN-SwaraNeural")
                ],
                value="hi-IN-MadhurNeural",
                label="Voice Character"
            )
            speed_slider = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, step=0.05, label="Speaking Speed / Pace")
            generate_btn = gr.Button("Generate Full Voiceover", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Audio Player & Download", type="filepath")
            status_output = gr.Textbox(label="Status", interactive=False)
            
    engine_selector.change(fn=update_voice_choices, inputs=engine_selector, outputs=voice_selector)
    generate_btn.click(fn=run_tts, inputs=[text_input, engine_selector, voice_selector, speed_slider], outputs=[audio_output, status_output])

if __name__ == "__main__":
    demo.launch()
