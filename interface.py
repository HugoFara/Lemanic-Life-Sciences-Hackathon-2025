import gradio as gr

def speech_recognition(wav_file, model_type, config_id, threshold, csv_file=None):
    return f"🎵 Audio File: {wav_file}\n✅ CSV file: {csv_file}🎯 Model: {model_type}\n🛠️ Config: {config_id}\n📊 Threshold: {threshold}"

def update_interface_model(model_type):
    if model_type == "Phoneme Deletion (french)" :
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def update_interface_csv(value):
    if value == "Yes" :
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

# Config lists
config_list_pho = ['phondel_A_pseudo_0', 'phondel_A_pseudo_1', 'phondel_A_pseudo_2', 'phondel_A_pseudo_3', 'phondel_A_pseudo_4', 'phondel_A_pseudo_5', 'phondel_A_pseudo_6', 'phondel_A_pseudo_7', 'phondel_A_pseudo_8', 'phondel_A_word_0', 'phondel_A_word_1', 'phondel_A_word_2', 'phondel_A_word_3', 'phondel_A_word_4', 'phondel_A_word_5', 'phondel_A_word_6', 'phondel_A_word_7', 'phondel_A_word_8', 'phondel_B_pseudo_0', 'phondel_B_pseudo_1', 'phondel_B_pseudo_2', 'phondel_B_pseudo_3', 'phondel_B_pseudo_4', 'phondel_B_pseudo_5', 'phondel_B_pseudo_6', 'phondel_B_pseudo_7', 'phondel_B_pseudo_8', 'phondel_B_word_0', 'phondel_B_word_1', 'phondel_B_word_2', 'phondel_B_word_3', 'phondel_B_word_4', 'phondel_B_word_5', 'phondel_B_word_6', 'phondel_B_word_7', 'phondel_B_word_8', 'phondel_C_pseudo_0', 'phondel_C_pseudo_1', 'phondel_C_pseudo_2', 'phondel_C_pseudo_3', 'phondel_C_pseudo_4', 'phondel_C_pseudo_5', 'phondel_C_pseudo_6', 'phondel_C_pseudo_7', 'phondel_C_pseudo_8', 'phondel_C_word_0', 'phondel_C_word_1', 'phondel_C_word_2', 'phondel_C_word_3', 'phondel_C_word_4', 'phondel_C_word_5', 'phondel_C_word_6', 'phondel_C_word_7', 'phondel_C_word_8']
config_list_dec = ['config_A_complex_1', 'config_A_complex_2', 'config_A_easy_1', 'config_A_easy_2', 'config_A_pseudo_1', 'config_A_pseudo_2', 'config_B_complex_1', 'config_B_complex_2', 'config_B_easy_1', 'config_B_easy_2', 'config_B_pseudo_1', 'config_B_pseudo_2', 'config_C_complex_1', 'config_C_complex_2', 'config_C_easy_1', 'config_C_easy_2', 'config_C_pseudo_1', 'config_C_pseudo_2']

with gr.Blocks(theme=gr.themes.Soft()) as demo: #Other themes that are nice : gr.themes.Default() / gr.themes.Base()
    gr.Markdown("#🎙️ Speech Recognition Interface")
    gr.Markdown("Upload or record an audio file, choose a model and config, and get an assessment.")

    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", format="wav", label="Upload or record a .wav file") #Here take of the note emoji or take it of from the
            

        with gr.Column(scale=1):
            model_selector = gr.Radio(["Phoneme Deletion (french)", "Decoding (italian)"], label="🧠 Model Type", value="Phoneme Deletion (french)")
            phoneme_dropdown = gr.Dropdown(config_list_pho, label="Phoneme Config ID", visible=True)
            decoding_dropdown = gr.Dropdown(config_list_dec, label="Decoding Config ID", visible=False)

    with gr.Row():
        csv_selector = gr.Radio(["Yes","No"], label="📂 Upload CSV", value="No")
        threshold = gr.Slider(0, 100, value=80, label="📈 Threshold")
    
    with gr.Row():
        csv_input = gr.File(label="Upload a CSV file", file_types=[".csv"], visible=False)

    model_selector.change(
        fn=update_interface_model,
        inputs=[model_selector],
        outputs=[phoneme_dropdown, decoding_dropdown]
    )

    csv_selector.change(
    fn=update_interface_csv,
    inputs=[csv_selector],
    outputs=[csv_input]
)

    def run_model(audio, model_type, pho_config, dec_config, threshold, csv_file):
        config = pho_config if model_type == "Phoneme Deletion (french)" else dec_config
        return speech_recognition(audio, model_type, config, threshold, csv_file)

    run_button = gr.Button("🚀 Run Model", size="lg", variant="primary")
    output_box = gr.Textbox(label="📤 Output", lines=5, interactive=False)

    run_button.click(
        fn=run_model,
        inputs=[audio_input, model_selector, phoneme_dropdown, decoding_dropdown, threshold, csv_input],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()