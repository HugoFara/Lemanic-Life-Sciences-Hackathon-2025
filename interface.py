"""Create a nice interface to interact with the app."""
from main import speech_recognition
import gradio as gr
import os


def run_model(wav_file, csv_file, model_type):
    wav_file = os.path.abspath(wav_file)
    csv_file = os.path.abspath(csv_file)
    results = speech_recognition(wav_file, csv_file, model_type)
    return f"🎵 Audio File: {wav_file}\n✅ CSV file: {csv_file}🎯" + results

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Speech Recognition Interface 🎙️")
    gr.Markdown("Upload an audio file and the CSV where the language and asked words are discribed.")

    with gr.Row():
        audio_input = gr.Audio(sources=["upload"], type="filepath", format="wav", label="Upload or record a .wav file") #Here take of the note emoji or take it of from the
        csv_input = gr.File(label="Upload a CSV file", type="filepath", file_types=[".csv"])
        
    with gr.Row(): 
        model_type = gr.Radio(["Phoneme Deletion (french)", "Decoding (italian)"], label="🧠 Model Type", value="Phoneme Deletion (french)")

    run_button = gr.Button("🚀 Run Model", size="lg", variant="primary")
    output_box = gr.Textbox(label="📤 Output", lines=5, interactive=False)

    run_button.click(
        fn=run_model,
        inputs=[audio_input, csv_input, model_type],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()