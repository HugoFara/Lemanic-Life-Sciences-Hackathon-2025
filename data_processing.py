import pandas as pd
import ipa_encoder


def get_phonemes(dataframe):
    new_dataframe = dataframe.copy()
    new_dataframe = new_dataframe.drop([
        "participant_id", "phase", "language", "form", "config", "config_id", "API_target",
        "accuracy_coder1", "notes_coder1", "accuracy_coder2", "notes_coder2"
    ], axis=1)

    for coder in ["coder1", "coder2"]:
        print("Preparing data for", coder)

        labeled_phonemes = dataframe["trial_answer_" + coder]

        # Remove all symbols
        labeled_phonemes = (
            labeled_phonemes.str
            # Replace space with a dot
            .replace(r'[ \s ]+', '.', regex=True)
            # Replace <...>, {...} with [UNK]
            .replace(r'(\{.*?\}|\<.*?\>)', '?', regex=True)
        )

        new_phonemes = ipa_encoder.get_ipa(
            [str(sentence) for sentence in labeled_phonemes],
            language
        )
        new_phonemes = [
            sentence.strip().replace(".", "[PAD]").replace("?", "[UNK]")
            for sentence in new_phonemes
        ]

        new_dataframe[f"phonemes_{coder}"] = pd.Series(new_phonemes)
        new_dataframe = new_dataframe.drop([f"trial_answer_{coder}"], axis=1)
    return new_dataframe


"""
For Classification

for coder in ["coder1", "coder2"]:
    # Remove Nan values for classification
    data = pd.read_csv(file_path_it)
    
    accuracy = "accuracy_" + coder
    data = data.dropna(subset=[accuracy])

    labeled_scores = data[accuracy]
"""


if __name__ == "__main__":
    language = ("fr", "it")[0]
    files_path = {
        "it": "Hackathon_ASR/1_Ground_truth/Decoding_ground_truth_IT.csv", 
        "fr": "Hackathon_ASR/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv"
    }
    dataframe = pd.read_csv(files_path[language])

    new_dataframe = get_phonemes(dataframe)
    output_path = f"datasets/phonemized_{language}.csv"
    new_dataframe.to_csv(output_path, index=False)

