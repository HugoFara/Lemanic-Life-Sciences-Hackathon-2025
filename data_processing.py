import pandas as pd
import re
import ipa_encoder

file_path_it = "data/1_Ground_truth/Decoding_ground_truth_IT.csv"
file_path_fr = "data/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv"
phonoemes = []
for file_path in [file_path_it, file_path_fr]:
    data = pd.read_csv(file_path)
    new_data = data.copy()
    coder = "coder1"

    print("Preparing data for", coder)

    trial_answer = "trial_answer_" + coder
    labeled_phonemes = data[trial_answer]

    # Remove all symbols
    labeled_phonemes = (
        labeled_phonemes.str.replace(".", " ", regex=False)  # Replace dots with [PAD]
        .str.replace(r"\{.*?\}", "[UNK]", regex=True)  # Replace {...} with [UNK]
        .str.replace(r"\<.*?\>", "", regex=True)  # Remove <...>
    )

    new_phonemes = []
    for sentence in labeled_phonemes:
        words = sentence.split()
        ipa_output = [
            "".join(
                item for sublist in ipa_encoder.get_french_ipa(word) for item in sublist
            )
            for word in words
        ]
        new_sentence = " ".join(ipa_output)
        new_phonemes.append(new_sentence)

    # Replace spaces with [PAD]
    labeled_phonemes = pd.Series(new_phonemes)
    labeled_phonemes = labeled_phonemes.str.replace(" ", "[PAD]", regex=False)

    column = "trial_answer_" + coder + "_phoneme"
    new_data[column] = labeled_phonemes
    phonoemes_tmp = new_data[column].astype(str).unique()
    phonoemes_tmp = "".join(phonoemes_tmp)
    phonoemes.append(phonoemes_tmp)
    output_path = "data/Processed/" + file_path.split("/")[-1].split(".")[0]

    new_data.to_csv(output_path + ".csv", index=True)

    # save the unique phonemes to a file
    with open(
        "data/Processed/phonemes" + file_path.split("/")[-1].split(".")[0] + ".txt", "w"
    ) as f:
        f.write("".join(phonoemes))
        print("Data prepared and saved to", output_path)


def extract_unique_phonemes(phoneme_str):
    # Match [PAD] or [UNK] as units, and then single characters
    pattern = r"\[PAD\]|\[UNK\]|."
    phonemes = re.findall(pattern, phoneme_str)
    unique_phonemes = list(
        dict.fromkeys(phonemes)
    )  # preserves order, removes duplicates
    return unique_phonemes


phonoemes = "".join(phonoemes)
unique_phonemes = extract_unique_phonemes(phonoemes)
# Save unique phonemes to a file
with open("data/Processed/unique_phonemes.txt", "w") as f:
    for phoneme in unique_phonemes:
        f.write(phoneme + "\n")
print("Unique phonemes extracted and saved to data/Processed/unique_phonemes.txt")
