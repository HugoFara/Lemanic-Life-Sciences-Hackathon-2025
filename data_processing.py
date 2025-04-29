import pandas as pd
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
        .str.replace('.', ' ', regex=False)               # Replace dots with [PAD]
        .str.replace(r'\{.*?\}', '[UNK]', regex=True)     # Replace {...} with [UNK]
        .str.replace(r'\<.*?\>', '', regex=True)          # Remove <...>
    )

    new_phonemes = []
    for sentence in labeled_phonemes:
        words = sentence.split()
        ipa_words = [ipa_encoder.get_french_ipa(word) for word in words]
        ipa_output = ["".join(item for sublist in ipa_encoder.get_french_ipa(word)for item in sublist) for word in words]
        new_sentence = " ".join(ipa_output)
        new_phonemes.append(new_sentence)
    
    # Replace spaces with [PAD]
    labeled_phonemes = pd.Series(new_phonemes)
    labeled_phonemes = labeled_phonemes.str.replace(' ', '[PAD]', regex=False)     

    new_data = data.copy()
    new_data["trial_answer_coder1"] = labeled_phonemes
    output_path = "Processed_" + coder
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

