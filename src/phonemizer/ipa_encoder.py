"""
Customized phonemization wrapper.
Original wrapper by Lingjuan Zhu (@lingjzhu):
https://github.com/lingjzhu/text2phonemesequence

"""
import csv
import json
import os
import subprocess
import re
import warnings

import torch

import segments
import transformers


class Text2PhonemeConverter:
    def __init__(
        self,
        words_to_exclude=None,
        tokenizer="google/byt5-small",
        language="fra",
        cuda=True,
        folder_language="lang_dict",
    ):
        """
        Load the rules to convert from a language to the corresponding phonemes.

        :param list words_to_exclude: List of words to exclude from phonemization.
        :param str tokenizer: Pre-trained tokenizer model name.
        :param str language: Language code for phonemization. Two letters only (e.g.: "fr")
        :param bool cuda: Flag to use CUDA for GPU acceleration.
        :param str folder_language: Folder path to save language dictionaries.
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            "charsiu/g2p_multilingual_byT5_small_100"
        )
        device = "cpu"
        if cuda:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                warnings.warn("CUDA is not available but was requested")
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        if words_to_exclude is None:
            words_to_exclude = ["[UNK]"]
        self.exclude_token = words_to_exclude
        self.segment_tool = segments.Tokenizer()
        self.language = {
            "fr": "fra",
            "it": "ita"
        }[language]
        self.phone_dict = {}
        ## Download language dictionary if not exists
        os.makedirs(folder_language, exist_ok=True)
        language_path = os.path.join(folder_language, self.language + ".tsv")
        if not os.path.exists(language_path):
            subprocess.run(
                [
                    "wget",
                    "-O", 
                    language_path,
                    f"https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/{self.language}.tsv"
                ],
                check=True
            )
        # SETTING PHONEME LENGTH
        config_path = "configs/phoneme_lengths.json"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.phoneme_lengths = json.load(f)
        else:
            warnings.warn("Loading dummy values for phonemes lengths!")
            self.phoneme_lengths = {
                "fra.tsv": 50,
                "ita.tsv": 50
            }
        if os.path.exists(language_path):
            with open(language_path, "r", encoding="utf-8") as file:
                tsv_file = csv.reader(file, delimiter="\t")
                # printing data line by line
                for word_phonemes in tsv_file:
                    assert len(word_phonemes) == 2
                    self.phone_dict[word_phonemes[0]] = [word_phonemes[1].split(",")[0]]
                    

    def phonemize(self, words, padding_token="[PAD]"):
        """
        Convert text to phonemes using the T5 model.
        
        :param list[str] words: Input text to be converted.
        :param str padding_token: Token used for padding.
        """
        phonemes_list = [""] * len(words)
        new_words = []
        self.exclude_token.append(padding_token)
        for i, word in enumerate(words):
            # First normalize the spacing around special tokens
            word = re.sub(r"(?<!\s)\[(UNK|PAD)\](?!\s)", r" [\1] ", word)
            # Then collapse multiple spaces
            word = re.sub(r" +", " ", word).strip()
            if word in self.phone_dict:
                phonemes_list[i] = self.phone_dict[word][0]
            elif word in self.exclude_token:
                phonemes_list[i] = word
            else:
                new_words.append((i, word))

        # Then batch the unknown words
        if new_words:
            out = self.tokenizer(
                [f"<{self.language}>: {word[1]}" for word in new_words],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            out["input_ids"] = out["input_ids"].to(self.device)
            out["attention_mask"] = out["attention_mask"].to(self.device)
            predictions = self.model.generate(
                **out,
                num_beams=1,
                max_length=self.phoneme_lengths[self.language + ".tsv"],
            )
            phonemes = self.tokenizer.batch_decode(
                predictions.tolist(), skip_special_tokens=True
            )
            for i, phonemized in enumerate(phonemes):
                phonemes_list[new_words[i][0]] = phonemized

        for i, phoneme in enumerate(phonemes_list):
            # skip excluded tokens of segmentation
            if phoneme not in self.exclude_token:
                phonemes_list[i] = self.segment_tool(phoneme, ipa=True)

        # fill gaps with padding token
        return padding_token.join(phonemes_list)


def extract_unique_phonemes(phoneme_str):
    pattern = r"\[PAD\]|\[UNK\]|."
    phonemes = re.findall(pattern, phoneme_str)
    # remove ' ' should not be in the list
    phonemes = [ph for ph in phonemes if ph != " "]
    return list(dict.fromkeys(phonemes))


def get_vocab_json(all_phonemes, output_path):
    phonemes = "".join(all_phonemes)
    unique_phonemes = extract_unique_phonemes(phonemes)
    unique_phonemes_dict = {ph: i for i, ph in enumerate(unique_phonemes)}
    # Save the unique phonemes to a JSON file
    os.makedirs(output_path, exist_ok=True)
    final_path = os.path.join(output_path, "vocab.json")
    with open(final_path, "w", encoding="utf-8") as file:
        json.dump(unique_phonemes_dict, file, ensure_ascii=False, indent=2)
