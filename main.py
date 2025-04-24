'''import Whisper.pipeline as whisper
from Combining_block.combining_block import combine_decoding

import pandas as pd
import numpy as np

def create_initial_csv(wav_file, language, config_id):
    """
    Create an initial CSV file with the audio file name, language, and config ID.
    Returns the path to the CSV file.
    """

    file_name = wav_file.split("/")[-1]
    initial_df = pd.DataFrame({"file_name" : file_name, "language" : language, "config_id" : config_id})
    csv_file = "initial.csv"
    initial_df.to_csv(csv_file, index=False)
    
    return csv_file

def apply_whisper(wav_file, csv_file, language, output_csv="whisper.csv"):
    """
    Runs the Whisper model on the audio file and saves the output to a CSV file.
    """
 
    output_csv = "whisper.csv"
    whisper.main(csv_file, wav_file, language, output_csv)
    

def speech_recognition(wav_file, csv_file, model_type):
    
    if model_type == "Phoneme Deletion (french)":
        language = "fr"
    else :
        language = "it"

    if csv_file is None:
        csv_file = create_initial_csv(wav_file, language, config_id)
    
    #Run whisper
    whisper_path = "whisper.csv"
    apply_whisper(wav_file, csv_file, language, whisper_path)

    #Run Wav2vec2
    pho_path = "pho_output.csv"
    apply_pho() #TODO
    
    #Combine the outputs
    combine_decoding(pho_path, whisper_path, model=model_type, csv_file=csv_file)'''


import Whisper.pipeline as wis
from Combining_block.combining_block import combine_decoding

import pandas as pd
import numpy as np
import ast

def apply_whisper(wav_file, csv_file, language, output_csv="whisper.csv"):
    """
    Runs the Whisper model on the audio file and saves the output to a CSV file.
    """
 
    wis.main_(csv_file, wav_file, language, output_csv)
    

def speech_recognition(wav_file, csv_file, model_type):
    
    if model_type == "Phoneme Deletion (french)":
        language = "fr"
    elif model_type == "Decoding (italian)":
        language = "it"

    #Run whisper
    whisper_path = "whisper.csv"
    apply_whisper(wav_file, csv_file, language, whisper_path)
    whisper_output = pd.read_csv(whisper_path)
    final_words = whisper_output['final_words'].apply(ast.literal_eval)
    words = [pair[0] for pair in final_words]
    whisper_result = f"{len(words)} words detected : {words}"

    #Run Wav2vec2
    pho_path = "pho_output.csv"
    #apply_pho() #TODO
    
    #Combine the outputs
    #combine_decoding(pho_path, whisper_path, model=model_type, csv_file=csv_file)

    return whisper_result

if __name__ == "__main__":
    wav_file = "C:/EPFL/Hackathon/2_Audiofiles/Phoneme_Deletion_FR_T1/3101_edugame2023_1c148def3c254026adc7a7fdc3edc6f6_3eff2b8d9be54f24aaa5f0bf3ef81c50.wav"
    csv_file = "C:\EPFL\Hackathon\interface_data.csv"
    model_type = "Phoneme Deletion (french)"
    
    speech_recognition(wav_file, csv_file, model_type)