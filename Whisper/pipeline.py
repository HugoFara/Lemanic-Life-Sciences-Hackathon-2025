import whisper
import pandas as pd
import os
import csv
import numpy as np
import ast
import re
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher



# ========== Functions  ==========

# --- Cleaning Functions ---

def clean_punctuation(words):
    cleaned = []
    for item in words:
        if isinstance(item, list) and len(item) == 2:
            word, times = item
            word_cleaned = re.sub(r'[.,]+$', '', word)         # remove trailing punctuation
            word_cleaned = re.sub(r'\.{3,}', '', word_cleaned) # remove long dots
            cleaned.append([word_cleaned, times])
        else:
            cleaned.append(item)
    return cleaned

def check_stutter(words):
    cleaned_words = []
    for i in range(len(words) - 1):
        current = words[i][0].strip()
        nxt = words[i + 1][0].strip()
        if nxt.startswith(current):
            continue
        cleaned_words.append(words[i])
    cleaned_words.append(words[-1])
    return cleaned_words

def check_repetition(words):
    cleaned_words = []
    for i in range(len(words) - 1):
        current = words[i][0].strip()
        nxt = words[i + 1][0].strip()
        if current == nxt:
            continue
        cleaned_words.append(words[i])
    cleaned_words.append(words[-1])
    return cleaned_words

def concatenate_words(words):
    concatenated = []
    skip = False
    for i in range(len(words) - 1):
        if skip:
            skip = False
            continue
        word, time = words[i]
        next_word, next_time = words[i + 1]
        if len(word.strip()) <= 2:
            new_word = word.strip() + next_word.strip()
            new_time = [time[0], next_time[1]]
            concatenated.append([new_word, new_time])
            skip = True
        else:
            concatenated.append([word, time])
    if not skip:
        concatenated.append(words[-1])
    return concatenated

def clean_output(words):
    if not words:
        return words
    words = clean_punctuation(words)
    words = check_repetition(words)
    words = check_stutter(words)
    words = concatenate_words(words)
    return words

# --- Filter and clean segments column ---

def filter_segments(segments):
    return [
        [word, timestamps] 
        for word, timestamps in segments 
        if not (len(word) <= 5 and "..." in word)
    ]

def safe_filter_segments(row):
    try:
        original = ast.literal_eval(row['segments'])
        filtered = filter_segments(original)
        if filtered != original:
            return filtered
        return None
    except Exception as e:
        print(f"Error in row: {row['file_name']}\n{e}")
        return None


def clean_reponse(reponse):
    if isinstance(reponse, list):
        reponse = " ".join(reponse)
    reponse = re.sub(r"[{}.\[\]]", " ", reponse)  # remove special punctuation
    reponse = re.sub(r"[^a-zA-Zàèéìòùç\s]", "", reponse)  # keep only letters and spaces
    mots = reponse.lower().split()
    return mots

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def correct_segments(segments, config_list):
    if isinstance(segments, str):
        try:
            segments = ast.literal_eval(segments)
        except:
            return []

    mots_ref = clean_reponse(config_list)

    if len(segments) < 12:
        return []
    elif len(segments) == 12:
        return segments

    mots_whisper = [mot[0].lower() for mot in segments]

    scores = []
    for i, mot_w in enumerate(mots_whisper):
        score_total = sum(similarity(mot_w, mot_ref) for mot_ref in mots_ref)
        scores.append((i, score_total))

    meilleurs = sorted(scores, key=lambda x: x[1], reverse=True)[:12]
    indices_gardes = sorted([i for i, _ in meilleurs])

    segments_corriges = [segments[i] for i in indices_gardes]
    return segments_corriges


# --- Full cleaning process applied to segments column ---

def clean_segments_column(df):
    df = df.copy()

    def process_row(segment_str):
        try:
            parsed = ast.literal_eval(segment_str)
            filtered = filter_segments(parsed)
            cleaned = clean_output(filtered)
            return str(cleaned)
        except Exception as e:
            print(f"Error processing row: {segment_str}\n{e}")
            return segment_str  # fallback

    df['segments'] = df['segments'].apply(process_row)

    df['len_segments'] = df['segments'].apply(
        lambda x: len(ast.literal_eval(x)) if x and x != '[]' else 0
    )

    return df

def french_clean(df):
    """
    Only keep rows where the 'segments' column contains exactly one word.
    Assumes 'segments' is a stringified list of [word, [start, end]] elements.
    """
    def has_one_segment(segment_str):
        try:
            segments = ast.literal_eval(segment_str)
            return isinstance(segments, list) and len(segments) == 1
        except:
            return False  # Skip rows with malformed segments

    df_filtered = df[df['segments'].apply(has_one_segment)].copy()
    return df_filtered

def italian_clean(df):
    """
    Only keep rows where the 'segments' column contains exactly 12 words.
    """
    def keep(segment_str):
        try:
            segments = ast.literal_eval(segment_str)
            return isinstance(segments, list) and len(segments) == 12
        except:
            return False  # Skip rows with malformed segments

    df_filtered = df[df['segments'].apply(keep)].copy()
    return df_filtered
    

### --- Main function to clean the entire CSV file ---
def clean_data(file_name, language=None):
    """
    Cleans the CSV file by applying various cleaning functions to the 'segments' column.

    file_name (str): The path to the CSV file.
    language (str): The language of the data. Can be "FR" or "IT" (None will remove no rows)
    """
    df = pd.read_csv(file_name)
    df = clean_segments_column(df)
    if language == "FR":
        df = french_clean(df)
    if language == "IT":
        df = italian_clean(df)
    return df

# match .wav to line in the df
def match_wav_to_line(wav_file, df):
    file_name = os.path.basename(wav_file)
    matched_row = df[df['file_name'] == file_name] # get the row with the same file name
    return matched_row

def replace_whisper_words_with_reference(final_words, reference_words):
    """
    Replace the words in final_words with the corresponding words in reference_words.
    """
    for i, (word, _) in enumerate(final_words):
        if i < len(reference_words):
            final_words[i][0] = reference_words[i]
    return final_words

def transcribe_single_audio(wav_file, language):
    """
    language (str): The language of the audio file. Can be "fr" or "it".
    """
    model = whisper.load_model("small")
    result = model.transcribe(str(wav_file), word_timestamps=True, language=language)
    word_segments = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            word_segments.append([
                word["word"].strip(),
                [round(word["start"], 2), round(word["end"], 2)]
            ])
    return word_segments


def nettoyer_reponse(reponse):
    if isinstance(reponse, list):
        reponse = " ".join(reponse)  # join the list into a single string
    reponse = re.sub(r"[{}.\[\]]", " ", reponse)  # enlever ponctuation spéciale
    reponse = re.sub(r"[^a-zA-Zàèéìòùç\s]", "", reponse)  # garder que les lettres et espaces
    mots = reponse.lower().split()
    return mots

# Calculer une similarité basique
def similarite(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Appliquer la logique de correction
def corriger_segments(row):
    try:
        segments = ast.literal_eval(row["segments"])
    except:
        return []
    
    mots_ref = nettoyer_reponse(row["config_list"])

    if len(segments) < 12:
        return []  # trop court, on ignore
    elif len(segments) == 12:
        return segments  # parfait

    # Si trop de mots, garder les 12 plus proches
    mots_whisper = [mot[0].lower() for mot in segments]

    scores = []
    for i, mot_w in enumerate(mots_whisper):
        score_total = sum(similarite(mot_w, mot_ref) for mot_ref in mots_ref)
        scores.append((i, score_total))

    # Garder les 12 meilleurs mots (plus cohérents avec la phrase attendue)
    meilleurs = sorted(scores, key=lambda x: x[1], reverse=True)[:12]
    indices_gardes = sorted([i for i, _ in meilleurs])  # trier pour garder l'ordre

    segments_corriges = [segments[i] for i in indices_gardes]
    return segments_corriges


def build_config_list(df_original):
    df_ref = df_original.copy()
    df_ref['config_list'] = df_original['config'].apply(lambda x: x.split(';'))
    df_ref = df_ref[['file_name', 'config_list']] # build csv keep only file name and config_list
    return df_ref

def remplace_by_references(row):
    try:
        # 'segments' might still be a string, so keep literal_eval here
        segments = ast.literal_eval(row["segments"]) if isinstance(row["segments"], str) else row["segments"]
        
        # 'config_list' is already a list
        config_words = row["config_list"]

        if len(segments) != len(config_words):
            return None  # lengths mismatch

        return [[ref_word, seg[1]] for ref_word, seg in zip(config_words, segments)]
    
    except Exception as e:
        print(f"Error in row {row.get('file_name', 'N/A')}: {e}")
        return None



    

# ========== Main Function (calls everything)  ==========

def main_(csv_path, wav_file, language, output_csv):
    df_original = pd.read_csv(csv_path) # load the original CSV file
    df_ref = build_config_list(df_original) # build the config_list (target words)
    matched_row = match_wav_to_line(wav_file, df_ref) # get the row with the same file name
    
    print("=== Transcribing.... ===")
    
    words = transcribe_single_audio(wav_file, language)  # whisper words from the audio file
    
    print("=== Cleaning.... ===")
        
    cleaned_words = clean_output(words) # takes care of punctuation, stutter, repetition and concatenation
    
    if language == "it":
        final_words = correct_segments(cleaned_words, matched_row["config_list"].values[0]) #  adapts guessed words using targets
        replaced_words = replace_whisper_words_with_reference(final_words, matched_row["config_list"].values[0]) # replace guessed words with target words

    if language == "fr":
        if len(cleaned_words) == 1:
            replaced_words = cleaned_words # ensure only one word
        else:
            replaced_words = []

    print("=== Saving.... ===")
    
    df_final = pd.DataFrame({
    "file_name": [os.path.basename(wav_file)],
    "final_words": [replaced_words]
    })
    
    df_final.to_csv(output_csv, index=False)
    
    print("=== Done! ===")


if __name__ == "__main__":
    
    # csv_path = "/pvc/scratch/speech_recognition/Hackathon_ASR/1_Ground_truth/Decoding_ground_truth_IT.csv"
    csv_path = "/Users/melina/Desktop/Hackathon/Lemanic-Life-Sciences-Hackathon-2025/interface_data_FR.csv"
    # wav_file = "/pvc/scratch/speech_recognition/Hackathon_ASR/2_Audiofiles/Decoding_IT_T1/102_edugame2023_32c4a5e851c1431aba3aa409e3be8128_649d404f44214261b67b24f1845e1350.wav"
    wav_file = "/Users/melina/Desktop/Hackathon/Hackathon_ASR/2_Audiofiles/Phoneme_Deletion_FR_T1/3101_edugame2023_1c148def3c254026adc7a7fdc3edc6f6_3eff2b8d9be54f24aaa5f0bf3ef81c50.wav"
    language = "fr" # "fr" or "it"
    output_csv = "final_words.csv"
    
    main_(csv_path, wav_file, language, output_csv)
    
    

    
 