"""
This is a text-to-IPA module.

You have two ways of usign it:

- Using espeak, works for all languages.
- Harnessing the power of AI through GPT2, post-processing and riding on modernity.
Work for French only. It is based on https://huggingface.co/Marxav/frpron
"""

import re
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


def get_ipa(words, language):
    """
    Convert a list of words to IPA using phonemizer.

    It skips special tokens like [PAD], [UNK], etc.

    Check the phonemizer module description for a list of supported languages.

    :param list[str] words: List of words
    :return list[list[str]]: IPA-transcribed words or original token if special
    """
    ipa_output = []
    normal_words = []
    word_indices = []

    for i, word in enumerate(words):
        if word == SPECIAL_TOKENS["pad"]:
            word_indices.append(0)
        elif word == SPECIAL_TOKENS["unk"]:
            word_indices.append(1)
        else:
            normal_words.append(word)
            word_indices.append(2)

    # Phonemize only non-special tokens
    if language == "fr":
        language += "-fr"
    phonemized = phonemizer.phonemize(normal_words, language=language, backend="espeak")
    for i, word in enumerate(words):
        if word_indices[i] == 0:
            ipa_output.append("[PAD]")
        elif word_indices[i] == 1:
            ipa_output.append("[UNK]")
        else:
            ipa_output.append(phonemized.pop(0))

    return ipa_output


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


if __name__ == "__main__":
    text = "Ceci est un texte français et accentué. Davoit, buccurelle. Plus de mots pour embrouiller le modèle"
    # text = "bonjour"
    text = "Bongiorno a tutti le ragazzie"

    print(get_ipa([text], "it"))
    # print(process_from_model([text]))