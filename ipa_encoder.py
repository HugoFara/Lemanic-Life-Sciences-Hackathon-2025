"""
This is a text-to-IPA module.

For now, it works on French and is based on https://huggingface.co/Marxav/frpron

It uses a GPT2 model, that may be quite old.
"""
import re

import torch
import transformers


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
        output[i] = generated_text[generated_text.index(":") + 1:]
    return output


def get_french_ipa(input_text, generator=None):
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


if __name__ == "__main__":
    # text = "Ceci est un texte français et accentué. Davoit, buccurelle. Plus de mots pour embrouiller le modèle"
    text = "bonjour"

    print(get_french_ipa([text]))
    #print(process_from_model([text]))
