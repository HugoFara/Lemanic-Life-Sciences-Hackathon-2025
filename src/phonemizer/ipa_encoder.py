"""
Customized phonemization wrapper.
Original wrapper by Lingjuan Zhu (@lingjzhu):
https://github.com/lingjzhu/text2phonemesequence

"""

import json
import os
import re
from pathlib import Path

from segments import Tokenizer
from transformers import AutoTokenizer, T5ForConditionalGeneration


class Text2PhonemeConverter:
    def __init__(
        self,
        words_to_exclude=["[UNK]"],
        tokenizer="google/byt5-small",
        language="fra",
        is_cuda=True,
        folder_language="lang_dict",
    ):
        """
        Args:
            words_to_exclude (list): List of words to exclude from phonemization.
            tokenizer (str): Pre-trained tokenizer model name.
            language (str): Language code for phonemization.
            is_cuda (bool): Flag to use CUDA for GPU acceleration.
            folder_language (str): Folder path to save language dictionaries.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained(
            "charsiu/g2p_multilingual_byT5_small_100"
        )
        self.is_cuda = is_cuda
        if self.is_cuda:
            self.model = self.model.cuda()
        self.exclude_token = words_to_exclude
        self.segment_tool = Tokenizer()
        self.language = language
        self.phone_dict = {}
        ## Download language dictionary if not exists
        os.makedirs(folder_language, exist_ok=True)
        language_path = os.path.join(folder_language, language + ".tsv")
        print(language_path)
        if not os.path.exists(language_path):
            os.system(
                f"wget -O {language_path} "  # -O specifies output path
                f"https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/{language}.tsv"
            )
        # SETTING PHONEME LENGTH
        config_path = Path("phoneme_lengths.json")
        with open(config_path, "r", encoding="utf-8") as f:
            self.phoneme_lengths = json.load(f)
        if os.path.exists(language_path):
            f = open(language_path, "r", encoding="utf-8")
            list_words = f.read().strip().split("\n")
            f.close()
            for word_phone in list_words:
                w_p = word_phone.split("\t")
                assert len(w_p) == 2
                if "," not in w_p[1]:
                    self.phone_dict[w_p[0]] = [w_p[1]]
                else:
                    self.phone_dict[w_p[0]] = [w_p[1].split(",")[0]]

    def phonemize(self, words="", padding_token="[PAD]"):
        """
        Convert text to phonemes using the T5 model.
        Args:
            words (str): Input text to be converted.
            padding_token (str): Token used for padding."""
        # First normalize the spacing around special tokens
        words = re.sub(r"(?<!\s)(\[UNK\]|\[PAD\])(?!\s)", r" \1 ", words)
        words = re.sub(r" +", " ", words).strip()  # Collapse multiple spaces

        list_words = words.split(" ")
        list_phones = []
        self.exclude_token.append(padding_token)
        for i in range(len(list_words)):
            if list_words[i] in self.phone_dict:
                list_phones.append(self.phone_dict[list_words[i]][0])
            elif list_words[i] in self.exclude_token:
                list_phones.append(list_words[i])
            else:
                out = self.tokenizer(
                    "<" + self.language + ">: " + list_words[i],
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                if self.is_cuda:
                    out["input_ids"] = out["input_ids"].cuda()
                    out["attention_mask"] = out["attention_mask"].cuda()
                if self.language + ".tsv" not in self.phoneme_lengths.keys():
                    self.phoneme_lengths[self.language + ".tsv"] = 50
                preds = self.model.generate(
                    **out,
                    num_beams=1,
                    max_length=self.phoneme_lengths[self.language + ".tsv"],
                )
                phones = self.tokenizer.batch_decode(
                    preds.tolist(), skip_special_tokens=True
                )
                list_phones.append(phones[0])
        for i in range(len(list_phones)):
            if list_phones[i] in self.exclude_token:
                continue  # skip excluded tokens of segmentation
            else:
                try:
                    segmented_phone = self.segment_tool(list_phones[i], ipa=True)
                except:
                    segmented_phone = self.segment_tool(list_phones[i])
                list_phones[i] = segmented_phone
        # fill gaps with padding token
        return padding_token.join(list_phones)


def extract_unique_phonemes(phoneme_str):
    pattern = r"\[PAD\]|\[UNK\]|."
    phonemes = re.findall(pattern, phoneme_str)
    # remove ' ' should not be in the list
    phonemes = [ph for ph in phonemes if ph != " "]
    return list(dict.fromkeys(phonemes))


def get_vocab_json(all_phonemes, output_path):
    phonoemes = "".join(all_phonemes)
    unique_phonemes_fr = extract_unique_phonemes(phonoemes)
    unique_phonemes_dict_fr = {ph: i for i, ph in enumerate(unique_phonemes_fr)}
    # Save the unique phonemes to a JSON file
    os.makedirs(output_path, exist_ok=True)
    final_path = os.path.join(output_path, "vocab.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(unique_phonemes_dict_fr, f, ensure_ascii=False, indent=2)
