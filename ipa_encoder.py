"""
This is a text-to-IPA module.

You have two ways of usign it:

- Using espeak, works for all languages.
- Harnessing the power of AI through GPT2, post-processing and riding on modernity.
Work for French only. It is based on https://huggingface.co/Marxav/frpron
"""

import re

import pandas as pd
import torch
import transformers

import phonemizer

SPECIAL_TOKENS = {"pad": "[PAD]", "unk": "[UNK]"}

def preprocessor(sentences):
    """
    Pre-process input sentences.

    :param list[str] sentences: Input sentences to process
    :return list[str]: Processed element
    """
    out_sentences = [""] * len(sentences)
    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        if " " in sentence:
            splitted = sentence.split(" ")
            for j, split in enumerate(splitted):
                splitted[j] = re.sub(r"\W", "", split) + ":"
            out_sentences[i] = splitted
        elif not sentence.endswith(":"):
            sentence = re.sub(r"\W", "", sentence)
            out_sentences[i] = sentence + ":"
        else:
            out_sentences[i] = re.sub(r"\W", "", sentence)

    return out_sentences


def process_from_model(sentences):
    """
    Process sentences using a model.

    :param list[str] sentences: Input sentences to process
    :return list[str]: Processed element
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("Marxav/frpron")
    model = transformers.AutoModelForCausalLM.from_pretrained("Marxav/frpron")

    clean_sentences = preprocessor(sentences)
    output = [[""] * len(clean_sentence) for clean_sentence in clean_sentences]
    for i, sentence in enumerate(clean_sentences):
        for j, word in enumerate(sentence):
            inputs = tokenizer(word, return_tensors="pt")

            with torch.no_grad():
                logits = model(**inputs).logits

                output[i][j] = tokenizer.decode(logits.argmax(dim=-1)[0])
        print(output[i])

    return output


def get_generator():
    return transformers.pipeline("text-generation", model="Marxav/frpron")


def post_process(predictions, remove_special_characters=True):
    """
    Remove the supplementary text in the predictions:

    :param list[dict] predictions: All the predicted answers from the model.
    :param bool remove_special_characters: Remove non-word characters.
    :return list[str]: Cleaned-up output.
    """
    output = [""] * len(predictions)
    for i, prediction in enumerate(predictions):
        generated_text = prediction["generated_text"]
        if remove_special_characters:
            generated_text = re.sub(r"[^\w:]", "", generated_text)
        output[i] = generated_text[generated_text.index(":") + 1 :]
    return output


def get_gpt_french_ipa(input_text, generator=None):
    """
    Return the IPA of a text, using a GPT2 model.

    :param list[str] input_text: Text to use, non-word characters will be discarded.
    :param generator: IPA generator to use.
    :return list[str]: List where each word is converted to IPA.
    """

    sentences_list = preprocessor(input_text)
    if generator is None:
        generator = get_generator()
    predictions = generator.predict(sentences_list)

    return [post_process(prediction) for prediction in predictions]


def get_ipa(sentences, language):
    """
    Convert a list of words to IPA using phonemizer.

    Check the phonemizer module description for a list of supported languages.

    :param list[str] words: List of words
    :return list[list[str]]: IPA-transcribed words or original token if special
    """
    # Phonemize only non-special tokens
    if language == "fr":
        language += "-fr"
        generator = get_generator()
    phonemized = phonemizer.phonemize(
        sentences,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=True
    )
    # Robusting by post-processing English insertions
    if language == "fr-fr":
        failed = []
        for i, sentence in enumerate(sentences):
            if "(en)" in phonemized[i] or "(fr)" in phonemized[i]:
                failed.append((i, sentence))
        reprocessed = [get_gpt_french_ipa(fail[1].split("."), generator) for fail in failed]
        for fail, new_phonemes in zip(failed, reprocessed):
            phonemized[fail[0]] = ".".join([x[0] for x in new_phonemes])
    return phonemized


def get_french_ipa(words):
    """
    Convert a list of words to IPA for French using phonemizer,
    skipping special tokens like [PAD], [UNK], etc.

    :param list[str] words: List of words
    :return list[list[str]]: IPA-transcribed words or original token if special
    """
    return get_ipa(words, "fr")



def get_italian_ipa(words):
    """
    Convert a list of words to IPA for Italian using phonemizer,
    skipping special tokens like [PAD], [UNK], etc.

    :param list[str] words: List of words
    :return list[list[str]]: IPA-transcribed words or original token if special
    """
    return get_ipa(words, "it")



def get_phonemes(dataframe, language):
    """Get the phonemes corresponding to the language."""
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

        new_phonemes = get_ipa(
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


def regenerate_ipa_csv(language):
    """From a language, recreate the IPA CSV file."""
    files_path = {
        "it": "Hackathon_ASR/1_Ground_truth/Decoding_ground_truth_IT.csv", 
        "fr": "Hackathon_ASR/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv"
    }
    dataframe = pd.read_csv(files_path[language])

    new_dataframe = get_phonemes(dataframe, language)
    output_path = f"datasets/phonemized_{language}.csv"
    new_dataframe.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return new_dataframe


if __name__ == "__main__":
    text = "ceci est un texte français et accentué.2000 years. Davoit, buccurelle.Plus de mots pour embrouiller le modèle".replace(" ", ".")
    # text = "bonjour"
    # text = "Bongiorno a tutti le ragazzie"

    print(get_ipa([text], "fr"))
    # print(process_from_model([text]))