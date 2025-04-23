import gradio as gr
# correct ?
# CI
# word

def speech_recognition(wav_file, model_type, config_id, threshold):
    return gr.Textbox(f"File: {wav_file}, Processed with model: {model_type}, config: {config_id}, threshold : {threshold}")

def update_interface(model_type):
    if model_type == "Phoneme Deletion (french)":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

config_list_pho = ['phondel_A_pseudo_0', 'phondel_A_pseudo_1', 'phondel_A_pseudo_2', 'phondel_A_pseudo_3', 'phondel_A_pseudo_4', 'phondel_A_pseudo_5', 'phondel_A_pseudo_6', 'phondel_A_pseudo_7', 'phondel_A_pseudo_8', 'phondel_A_word_0', 'phondel_A_word_1', 'phondel_A_word_2', 'phondel_A_word_3', 'phondel_A_word_4', 'phondel_A_word_5', 'phondel_A_word_6', 'phondel_A_word_7', 'phondel_A_word_8', 'phondel_B_pseudo_0', 'phondel_B_pseudo_1', 'phondel_B_pseudo_2', 'phondel_B_pseudo_3', 'phondel_B_pseudo_4', 'phondel_B_pseudo_5', 'phondel_B_pseudo_6', 'phondel_B_pseudo_7', 'phondel_B_pseudo_8', 'phondel_B_word_0', 'phondel_B_word_1', 'phondel_B_word_2', 'phondel_B_word_3', 'phondel_B_word_4', 'phondel_B_word_5', 'phondel_B_word_6', 'phondel_B_word_7', 'phondel_B_word_8', 'phondel_C_pseudo_0', 'phondel_C_pseudo_1', 'phondel_C_pseudo_2', 'phondel_C_pseudo_3', 'phondel_C_pseudo_4', 'phondel_C_pseudo_5', 'phondel_C_pseudo_6', 'phondel_C_pseudo_7', 'phondel_C_pseudo_8', 'phondel_C_word_0', 'phondel_C_word_1', 'phondel_C_word_2', 'phondel_C_word_3', 'phondel_C_word_4', 'phondel_C_word_5', 'phondel_C_word_6', 'phondel_C_word_7', 'phondel_C_word_8']
config_list_dec = ['config_A_complex_1', 'config_A_complex_2', 'config_A_easy_1', 'config_A_easy_2', 'config_A_pseudo_1', 'config_A_pseudo_2', 'config_B_complex_1', 'config_B_complex_2', 'config_B_easy_1', 'config_B_easy_2', 'config_B_pseudo_1', 'config_B_pseudo_2', 'config_C_complex_1', 'config_C_complex_2', 'config_C_easy_1', 'config_C_easy_2', 'config_C_pseudo_1', 'config_C_pseudo_2']

with gr.Blocks() as demo:
    audio_input = gr.Audio(
        sources=["upload", "microphone"], type="filepath", format="wav",
        label="Upload or record a .wav file"
    )

    model_selector = gr.Radio(
        ["Phoneme Deletion (french)", "Decoding (italian)"], label="Model type"
    )

    threshold = gr.Slider(0, 100, value=80, label="Threshold")

    phoneme_dropdown = gr.Dropdown(config_list_pho, label="Phoneme Config ID", visible=False)
    decoding_dropdown = gr.Dropdown(config_list_dec, label="Decoding Config ID", visible=False)
    
    
    # Update dropdown visibility based on model selection
    model_selector.change(
        fn=update_interface,
        inputs=model_selector,
        outputs=[phoneme_dropdown, decoding_dropdown]
    )

    # Logic to use the right dropdown depending on selection
    def run_model(audio, model_type, pho_config, dec_config, threshold):
        config = pho_config if model_type == "Phoneme Deletion (french)" else dec_config
        return speech_recognition(audio, model_type, config, threshold)

    run_button = gr.Button("Run model")
    run_button.click(
        fn=run_model,
        inputs=[audio_input, model_selector, phoneme_dropdown, decoding_dropdown, threshold],
        outputs=gr.Textbox(label="Output"),
    )


if __name__ == "__main__":
    demo.launch()
