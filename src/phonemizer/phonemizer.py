import json
import os

import pandas as pd
import tqdm

import ipa_encoder


def phonemize_text(
    csv_path, which_coder, language, undefined_token="[UNK]", padding_token="[PAD]"
):
    """
    Phonemize the text in the specified column of a CSV file.

    :param str csv_path: Path to the CSV file.
    :param list[int] which_coder: Coder to phonemize.
    :apram str language: Language code for phonemization. Two letters.
    Check https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?gid=557940309#gid=557940309
    :param str undefined_token: Token for undefined words.
    :apram str padding_token: Token for padding.
    """
    df = pd.read_csv(csv_path)
    print(f"Phonemizing {which_coder} from {csv_path}.")
    columns = [f"trial_answer_coder{i}" for i in which_coder]
    words_to_phonemize = df[["file_name"] + columns].head(10)
    new_cols = [f"phonemized{i}" for i in which_coder]
    undefined_token_sep = undefined_token
    padding_token_sep = padding_token
    for in_col, out_col in zip(columns, new_cols):
        words_to_phonemize[out_col] = words_to_phonemize[in_col].str.replace(
            " ", padding_token_sep, regex=True
        )

        words_to_phonemize[out_col] = (
            words_to_phonemize[in_col]
            .str.replace(".", padding_token_sep, regex=False)
            # Modified regex to handle [UNK] properly:
            .str.replace(r"\{.*?\}|\<.*?\>|\[.*?\]|\(.*?\)", undefined_token_sep, regex=True)
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
        words_to_phonemize[out_col] = words_to_phonemize[out_col].fillna(undefined_token)

    text_to_phoneme = ipa_encoder.Text2PhonemeConverter(
        language=language, words_to_exclude=[undefined_token]
    )
    # Enable pandas integration
    tqdm.tqdm.pandas()

    for out_col in new_cols:
        # Add progress bar to your apply()
        words_to_phonemize[out_col] = words_to_phonemize[out_col].progress_apply(
            lambda x: text_to_phoneme.phonemize(
                x.split(" "),
                padding_token=" " + padding_token + " "
            )
        )

    phonemized_df = pd.merge(
        df,
        words_to_phonemize[["file_name"] + new_cols],
        on="file_name",
        how="left",
        validate="one_to_one"
    )
    return phonemized_df


def get_vocabulary_json(all_phonemes):
    """Save the vocab to a JSON file."""
    phonemes = " ".join(all_phonemes)
    unique_phonemes = {"[PAD]", "[UNK]"}
    unique_phonemes.update(phonemes.split(" "))
    return {ph: i for i, ph in enumerate(unique_phonemes)}


def save_vocabulary(vocabulary, output_file):
    # Save the unique phonemes to a JSON file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved as {output_file}")


if __name__ == "__main__":
    data_folder = "Hackathon_ASR"
    saved_folder = "outputs/phonemizer"
    phonemes_folder = "custom_tokenizer"
    os.makedirs(saved_folder, exist_ok=True)
    os.makedirs(phonemes_folder, exist_ok=True)
    paths = {
        "fr": "Phoneme_Deleletion_ground_truth_FR",
        "it": "Decoding_ground_truth_IT"
    }
    dataseries = []
    for language in ("fr", "it"):
        coders = [1, 2]
        phonemized_dataframe = phonemize_text(
            f"{data_folder}/1_Ground_truth/{paths[language]}.csv",
            coders,
            language
        )
        phonemized_dataframe.to_csv(
            f"{saved_folder}/phonemized_{language}.csv",
            index=False
        )
        for i in coders:
            dataseries.extend(phonemized_dataframe[f"phonemized{i}"].dropna())

    save_vocabulary(
        get_vocabulary_json(dataseries),
        f"{phonemes_folder}/vocab{language}.json"
    )
