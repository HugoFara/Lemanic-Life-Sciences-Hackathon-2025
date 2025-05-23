import json
import os

import datasets
import pandas as pd
import torch

from . import text_to_phoneme_converter

VOCAB_FOLDER = "custom_tokenizer"
VOCAB_FILE = f"{VOCAB_FOLDER}/vocab.json"

def phonemized_dataset(dataset, language, in_features, out_features=None):
    """
    Take a dataset and convert the requested columns to phonemes.
    
    :param datasets.Dataset dataset: Dataset to work on.
    :param str language: Lanugage to process phonemes to.
    :param list[str] in_features: Features to as an input string from the dataset.
    :param list[str] out_features: The output will be mapped on this feature.
    """
    if out_features is None:
        out_features = in_features.copy()

    text_to_phoneme = text_to_phoneme_converter.Text2PhonemeConverter(
        language=language,
        cuda=torch.cuda.is_available()
    )
    return dataset.map(
        lambda batch: {
            out_col: text_to_phoneme.phonemize(batch[in_col])
            for in_col, out_col in zip(in_features, out_features)
        },
        desc=f"Phonemizing {language}",
        batched=True,
        batch_size=300
    )


def phonemize_text(
    csv_path, 
    which_coder, 
    language,
    undefined_token="[UNK]",
    padding_token="[PAD]",
    max_rows=-1
):
    """
    Phonemize the text in the specified column of a CSV file.

    :param str csv_path: Path to the CSV file.
    :param list[int] which_coder: Coder to phonemize.
    :apram str language: Language code for phonemization. Two letters.
    Check https://docs.google.com/spreadsheets/d/1y7kisk-UZT9LxpQB0xMIF4CkxJt0iYJlWAnyj6azSBE/edit?gid=557940309#gid=557940309
    :param str undefined_token: Token for undefined words.
    :param str padding_token: Token for padding.
    :param int max_rows: Limit the number of processed rows to go faster.
    """
    dataframe = pd.read_csv(csv_path)
    columns = [f"trial_answer_coder{i}" for i in which_coder]
    words_to_phonemize = dataframe[["file_name"] + columns]
    if max_rows != -1:
        words_to_phonemize = words_to_phonemize.head(max_rows)
    new_cols = [f"phonemized{i}" for i in which_coder]
    undefined_token_sep = undefined_token
    padding_token_sep = padding_token
    for in_col, out_col in zip(columns, new_cols):
        words_to_phonemize[out_col] = (
            words_to_phonemize[in_col].str
            .replace(r".| ", padding_token_sep, regex=False)
            # Modified regex to handle [UNK] properly:
            .replace(r"\{.*?\}|\<.*?\>|\[.*?\]|\(.*?\)", undefined_token_sep, regex=True)
            # Handle string boundaries:
            .replace(
                rf"^{padding_token}", padding_token + " ", regex=True
            )
            # Start of string
            .replace(
                rf"{padding_token}$", " " + padding_token, regex=True
            )
            # End of string
            .replace(rf"^{undefined_token}", undefined_token + " ", regex=True)
            .replace(rf"{undefined_token}$", " " + undefined_token, regex=True)
            .fillna(undefined_token)
        )

    dataset = datasets.Dataset.from_pandas(words_to_phonemize[new_cols])
    
    processed = phonemized_dataset(dataset, language, new_cols)
    
    for out_col in new_cols:
        words_to_phonemize[out_col] = processed[out_col] 

    phonemized_df = pd.merge(
        dataframe,
        words_to_phonemize[["file_name"] + new_cols],
        on="file_name",
        how="left",
        validate="one_to_one"
    )
    return phonemized_df


def get_vocabulary_dict(all_phonemes):
    """Take the unique phonemes and return them as a dictionary."""
    phonemes = " ".join(all_phonemes)
    special_tokens = {"[PAD]": 0, "[UNK]": 1}
    unique_phonemes = set(phonemes.split(" "))
    unique_phonemes.difference_update(special_tokens)
    output_dict = special_tokens.copy()
    ordered_phonemes = list(unique_phonemes)
    ordered_phonemes.sort()
    output_dict.update({
        phoneme: i 
        for i, phoneme in enumerate(ordered_phonemes, start=len(special_tokens))
    })
    return output_dict


def phonemize_texts(file_paths, max_rows=-1):
    """
    Recreate the vocubalary files and create the phonemized texts.

    :param dict[str, str] file_paths: (language_name, file_path) for each file to process.
    :param int max_rows: Maximum number of rows to process.
    """
    saved_folder = "outputs/phonemizer"
    os.makedirs(saved_folder, exist_ok=True)
    output_paths = {}
    for language, data_file in file_paths.items():
        coders = [1, 2]
        phonemized_dataframe = phonemize_text(
            data_file,
            coders,
            language,
            max_rows=max_rows
        )
        output_paths[language] = f"{saved_folder}/phonemized_{language}.csv"
        phonemized_dataframe.to_csv(output_paths[language], index=False)
    return output_paths


def get_all_phonemes(phonemized_files):
    """
    Get all phonemes from a phonemized file.

    :param phonemized_files: Path to the phonemized files where to get phonemes from.
    It is in format (language, file_path)
    :type phonemized_files: dict[str, str]

    """
    dataseries = []
    for phonemized_file in phonemized_files.values():
        coders = [1, 2]
        phonemized_dataframe = pd.read_csv(phonemized_file)
        for i in coders:
            dataseries.extend(phonemized_dataframe[f"phonemized{i}"].dropna())
    return dataseries


def regenerate_vocabulary(dataseries, output_file=VOCAB_FILE):
    """
    Recreate the vocabulary files and create the phonemized texts.

    :param list dataseries: List of all phonemes we have.

    :return str: Output file path.
    """
    # Save the unique phonemes to a JSON file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(get_vocabulary_dict(dataseries), file, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved as {output_file}")

    return output_file


def check_regenerate_vocabulary(all_phonemes):
    """Regenerate the vocabulary if at least one phoneme is missing."""
    os.makedirs(VOCAB_FOLDER, exist_ok=True)
    if not os.path.exists(VOCAB_FILE):
        regenerate_vocabulary(all_phonemes, VOCAB_FILE)
    else:
        # File exists, check phonemes
        with open(VOCAB_FILE, "r") as file:
            data = json.load(file)
            registered_phonemes = set(data.keys())
        
        proposed = set(all_phonemes)
        if proposed.difference(registered_phonemes):
            print("Updating vocabulary")
            regenerate_vocabulary(all_phonemes, VOCAB_FILE)

    with open(VOCAB_FILE, "r") as file:
        phonemes_dict = json.load(file)
        
    return phonemes_dict


if __name__ == "__main__":
    file_paths = {
        "fr": "Hackathon_ASR/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv",
        "it": "Hackathon_ASR/1_Ground_truth/Decoding_ground_truth_IT.csv"
    }
    phonemized_files = phonemize_texts(file_paths, 10)
    dataseries = get_all_phonemes(phonemized_files)
    check_regenerate_vocabulary(dataseries)
