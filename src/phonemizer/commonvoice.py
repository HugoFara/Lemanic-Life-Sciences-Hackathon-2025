import json

import datasets
import huggingface_hub

from . import text_phonemizer

LIMIT_ITEMS = 3000
LANGUAGES = ("fr", "it")


def get_phonemized_datasets(languages, limit_items=LIMIT_ITEMS):
    """Get a fully phonemized dataset from common voice."""
    with open("config.json", "r") as file:
        config = json.load(file)
        huggingface_hub.login(config["hf_token"])

    ds = {}
    partial = ""
    if limit_items != -1:
        partial = f"[:{limit_items}]"

    for language in languages:
        common_voice_ds = datasets.load_dataset(
            "mozilla-foundation/common_voice_17_0",
            language,
            split=f"train{partial}"
        ).remove_columns(
            ["client_id", "path", "up_votes", "down_votes", "gender", "locale", "segment", "variant"]
        )
        # audio and sentence
        ds[language] = text_phonemizer.phonemized_dataset(
            common_voice_ds,
            language,
            ["sentence"],
            ["target_phonemes1"]
        )
    return ds


if __name__ == "__main__":
    ds = get_phonemized_datasets(LANGUAGES)
    print(ds[LANGUAGES[0]]["sentence"])
