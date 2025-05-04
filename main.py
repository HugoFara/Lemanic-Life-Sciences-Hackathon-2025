import ast

import pandas as pd

import word_segmenter.pipeline
from combining_block.combining_block import combine_decoding


def apply_whisper(wav_file, csv_file, language, output_csv="whisper.csv"):
    """
    Runs the Whisper model on the audio file and saves the output to a CSV file.
    """
    word_segmenter.pipeline.main_(csv_file, wav_file, language, output_csv)
    

def speech_recognition(wav_file, csv_file, model_type):
    """Run the complete pipeline."""
    if model_type == "Phoneme Deletion (french)":
        language = "fr"
    elif model_type == "Decoding (italian)":
        language = "it"

    # Run whisper
    whisper_path = "whisper.csv"
    apply_whisper(wav_file, csv_file, language, whisper_path)
    whisper_output = pd.read_csv(whisper_path)
    final_words_raw = whisper_output.loc[0, 'final_words']
    cleaned = final_words_raw.replace('np.float64(', '').replace(')', '')
    final_words = ast.literal_eval(cleaned)
    words = [pair[0] for pair in final_words]
    whisper_result = f"{len(words)} words detected: {words}"

    # Run Wav2vec2
    pho_path = "output_FR.csv"
    
    # Combine the outputs
    if model_type == 'Phoneme Deletion (french)':
        phonem_test = pd.read_csv("phonem_test.csv")
        combined_output = combine_decoding(pho_path, whisper_path, csv_file=csv_file, model=model_type, first_phonemes_csv=phonem_test)
    else: 
        combined_output = combine_decoding(pho_path, whisper_path, csv_file=csv_file, model=model_type)
    combined_output.to_csv("combined_output.csv", index=False)

    return whisper_result

if __name__ == "__main__":
    wav_file = "Hackathon_ASR/2_Audiofiles/Phoneme_Deletion_FR_T1/3101_edugame2023_1c148def3c254026adc7a7fdc3edc6f6_3eff2b8d9be54f24aaa5f0bf3ef81c50.wav"
    csv_file = "interface_data_FR.csv"
    model_type = "Phoneme Deletion (french)"
    
    speech_recognition(wav_file, csv_file, model_type)