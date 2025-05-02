import os

import pandas as pd
import ipa_encoder
import tqdm


def phonemize_text(
    csv_path, col, language, undefined_token="[UNK]", padding_token="[PAD]"
):
    """
    Phonemize the text in the specified column of a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        col (str): Column name to phonemize.
        language (str): Language code for phonemization. 
        Check https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?gid=557940309#gid=557940309
        undefined_token (str): Token for undefined words.
        padding_token (str): Token for padding.
    """
    df = pd.read_csv(csv_path)
    print(f"Preparing to phonemize {col} from {csv_path}.")
    words_to_phonemize = df[["file_name", col]].head(20)
    new_col = col + "_phonemized"
    undefined_token_sep = undefined_token
    padding_token_sep = padding_token
    words_to_phonemize[new_col] = words_to_phonemize[col].str.replace(
        " ", padding_token_sep, regex=True
    )

    words_to_phonemize[new_col] = (
        words_to_phonemize[col]
        .str.replace(".", padding_token_sep, regex=False)
        # Modified regex to handle [UNK] properly:
        .str.replace(
            r"(?<!\s)(\[UNK\])(?!\s)", r" \1 ", regex=True
        )  # Add spaces around [UNK] if not already there
        .str.replace(r"\{.*?\}", undefined_token_sep, regex=True)
        .str.replace(r"\<.*?\>", undefined_token_sep, regex=True)
        .str.replace(r"\[.*?\]", undefined_token_sep, regex=True)
        .str.replace(r"\(.*?\)", undefined_token_sep, regex=True)
        # Handle string boundaries:
        .str.replace(
            rf"^{padding_token}", padding_token + " ", regex=True
        )
        # Start of string
        .str.replace(
            rf"{padding_token}$", " " + padding_token, regex=True
        )
        # End of string
        .str.replace(rf"^{undefined_token}", undefined_token + " ", regex=True)
        .str.replace(rf"{undefined_token}$", " " + undefined_token, regex=True)
    )
    words_to_phonemize[new_col] = words_to_phonemize[new_col].fillna(undefined_token)
    text_to_phoneme = ipa_encoder.Text2PhonemeConverter(
        language=language, words_to_exclude=[undefined_token]
    )
    # Enable pandas integration
    tqdm.tqdm.pandas()

    # Add progress bar to your apply()
    words_to_phonemize[new_col] = words_to_phonemize[new_col].progress_apply(
        lambda x: text_to_phoneme.phonemize(x, padding_token=padding_token)
    )
    phonemized_df = pd.merge(
        df, words_to_phonemize[["file_name", new_col]], on="file_name", how="left"
    )
    return phonemized_df


if __name__ == "__main__":
    data_folder = "Hackathon_ASR"
    saved_folder = "outputs/phonemizer/"
    os.makedirs(saved_folder, exist_ok=True)
    phonemized_df_ita = phonemize_text(
        f"{data_folder}/1_Ground_truth/Decoding_ground_truth_IT.csv",
        "trial_answer_coder2",
        "ita",
    )
    phonemized_df_ita.to_csv("outputs/phonemizer/phonemized_IT.csv", index=False)
    phonemized_df_fr = phonemize_text(
        f"{data_folder}/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv",
        "trial_answer_coder1",
        "fra",
    )
    phonemized_df_fr.to_csv("outputs/phonemizer/phonemized_FR.csv", index=False)
    all_phonemes = (
        phonemized_df_ita["trial_answer_coder2_phonemized"].dropna().tolist()
        + phonemized_df_fr["trial_answer_coder1_phonemized"].dropna().tolist()
    )
    ipa_encoder.get_vocab_json(all_phonemes, "custom_tokenizer")
