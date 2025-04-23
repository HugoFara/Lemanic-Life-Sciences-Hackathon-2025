import pandas as pd
import re
import ipa_encoder

file_path_it = "Data/Ground_Truth/Decoding_ground_truth_IT.csv"
file_path_fr = "Data/Ground_Truth/Phoneme_Deleletion_ground_truth_FR.csv"

for coder in ["coder1", "coder2"]:
    data = pd.read_csv(file_path_it)
    print("Preparing data for", coder)

    trial_answer = "trial_answer_" + coder
    labeled_phonemes = data[trial_answer]

    # Remove all symbols
    labeled_phonemes = (labeled_phonemes
        .str.replace('.', ' ', regex=False)                  # Replace dots with spaces
        .str.replace(r'\{.*?\}', '', regex=True)             # Remove {...}
        .str.replace(r'\<.*?\>', '', regex=True)             # Remove <...>
        .str.replace(r'\b(?:pause|only_environment)\b', '', regex=True)  # Remove English words
    )

    for sentence in labeled_phonemes:
        words = sentence.split()
        print(words)
        ipa_words = [ipa_encoder.get_french_ipa(word) for word in words]
        print(ipa_words)
        ipa_output = " ".join(ipa_words)
        print(ipa_output)

    new_data = data.copy()
    new_data["trial_answer_coder1"] = labeled_phonemes
    output_path = "Data/Processed" + coder
    new_data.to_csv(output_path, index=True)




"""
For Classification

for coder in ["coder1", "coder2"]:
    # Remove Nan values for classification
    data = pd.read_csv(file_path_it)
    
    accuracy = "accuracy_" + coder
    data = data.dropna(subset=[accuracy])

    labeled_scores = data[accuracy]
"""

