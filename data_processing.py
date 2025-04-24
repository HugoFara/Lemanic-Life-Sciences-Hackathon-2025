import pandas as pd
import re
import ipa_encoder
from tqdm import tqdm
import numpy as np

file_paths = {
    "it": "data/1_Ground_truth/Decoding_ground_truth_IT.csv",
    "fr": "data/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv",
}

phonoemes = []

for lang, file_path in file_paths.items():
    data = pd.read_csv(file_path)
    new_data = pd.DataFrame()
    if lang == "fr":
        coder = "coder1"
    elif lang == "it":
        coder = "coder2"

    print(f"Preparing data for {coder} in language: {lang.upper()}")

    trial_answer = f"trial_answer_{coder}"
    labeled_phonemes = data[trial_answer]
    new_data["file_name"] = data["file_name"]
    # Clean symbols
    labeled_phonemes = (
        labeled_phonemes.str.replace(".", " ", regex=False)
        .str.replace(r"\{.*?\}", " [UNK] ", regex=True)
        .str.replace(r"\<.*?\>", "", regex=True)
        .str.replace(" ", " [PAD] ", regex=False)
    )
    labeled_phonemes = labeled_phonemes.fillna("[UNK]")

    new_phonemes = []

    print(f"Processing {len(labeled_phonemes)} sentences in {lang.upper()}...")

    for sentence in tqdm(labeled_phonemes, desc=f"{lang.upper()} phonemizing"):
        words = sentence.split()

        if lang == "fr":
            ipa_output_nested = ipa_encoder.get_french_ipa(words)
            ipa_output = ["".join(sublist) for sublist in ipa_output_nested]
        elif lang == "it":
            ipa_output_nested = ipa_encoder.get_italian_ipa(words)
            ipa_output = ["".join(sublist) for sublist in ipa_output_nested]
        else:
            raise ValueError(f"Unsupported language: {lang}")

        new_sentence = "".join(ipa_output)
        new_sentence = new_sentence.replace(" ", "")
        new_phonemes.append(new_sentence)

    # Replace spaces with [PAD]

    column = f"trial_answer_{coder}_phoneme"
    print(f"Adding column: {column}")
    print(new_data["file_name"])
    new_data[column] = new_phonemes
    print(new_data)
    phonoemes_tmp = "".join(new_data[column].astype(str).unique())
    phonoemes.append(phonoemes_tmp)

    output_base = file_path.split("/")[-1].split(".")[0]
    new_data.to_csv(f"data/processed/2_{output_base}.csv", index=True)

    with open(f"data/processed/phonemes_{output_base}.txt", "w") as f:
        f.write("".join(phonoemes))
        print(f"Data prepared and saved to data/processed/{output_base}.csv")


# Extract unique phonemes
def extract_unique_phonemes(phoneme_str):
    pattern = r"\[PAD\]|\[UNK\]|."
    phonemes = re.findall(pattern, phoneme_str)
    return list(dict.fromkeys(phonemes))


phonoemes = "".join(phonoemes)
unique_phonemes = extract_unique_phonemes(phonoemes)

with open("data/processed/unique_phonemes2.txt", "w") as f:
    for phoneme in unique_phonemes:
        f.write(phoneme + "\n")

print("Unique phonemes extracted and saved to data/processed/unique_phonemes.txt")
