import pandas as pd
import ipa_encoder

language = "fr"

files_path = {
    "it": "Hackathon_ASR/1_Ground_truth/Decoding_ground_truth_IT.csv", 
    "fr": "Hackathon_ASR/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv"
}

data = pd.read_csv(files_path[language])

for coder in ["coder1", "coder2"]:
    print("Preparing data for", coder)

    labeled_phonemes = data["trial_answer_" + coder]

    # Remove all symbols
    labeled_phonemes = (
        labeled_phonemes.str
        .replace('.', ' ', regex=False)               # Replace dots with [PAD]
        .replace(r'\{.*?\}', '[UNK]', regex=True)     # Replace {...} with [UNK]
        .replace(r'\<.*?\>', '', regex=True)          # Remove <...>
    )

    new_phonemes = []
    for sentence in labeled_phonemes:
        words = sentence.split()
        ipa_words = ipa_encoder.get_ipa(words, language)
        new_phonemes.append("[PAD]".join(ipa_words))

    labeled_phonemes = pd.Series(new_phonemes)

    new_data = data.copy()
    new_data[f"trial_answer_phonemes"] = labeled_phonemes
    output_path = f"outputs/phonemized_{language}_{coder}"
    # new_data.to_csv(output_path, index=True)


"""
For Classification

for coder in ["coder1", "coder2"]:
    # Remove Nan values for classification
    data = pd.read_csv(file_path_it)
    
    accuracy = "accuracy_" + coder
    data = data.dropna(subset=[accuracy])

    labeled_scores = data[accuracy]
"""

