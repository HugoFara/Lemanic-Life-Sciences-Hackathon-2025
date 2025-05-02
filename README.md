# Lemanic-Life-Sciences-Hackathon-2025

Repository for code on the Lemanic Life Sciences Hackathon 2025 submission!

## Phonemizer Module

This folder `src/phonemizer` contains tools for phonemizing text data into phoneme sequences using a multilingual grapheme-to-phoneme (G2P) model and IPA segmentation.

### Files

- `ipa_encoder.py`:

  - Defines the `Text2PhonemeConverter` class, which converts text into phonemes using the `charsiu/g2p_multilingual_byT5_small_100` model.
  - Also provides utilities to extract and save the set of unique phonemes (`get_vocab_json`).

- `phonemizer.py`:
  - Provides the `phonemize_text` function to phonemize text columns in CSV files.
  - Prepares the text by handling undefined tokens, padding, and formatting.
  - Generates phonemized CSV outputs and a phoneme vocabulary JSON file.

### How to Use

1. **Phonemize a CSV File**

   Example:

   ```bash
   python phonemizer.py
   ```

   By default, this will:

   - Phonemize specific columns from CSV files located in `data/1_Ground_truth/`.
   - Save the phonemized versions in `outputs/phonemizer/`.
   - Generate a `vocab.json` file containing all unique phonemes found.

2. **Use the Code Programmatically**
   Example:

   ```python
   from phonemizer import phonemize_text

   df_phonemized = phonemize_text(
       csv_path="your_data.csv",
       col="your_column_name",
       language="fra",  # or "ita"
       undefined_token="[UNK]",
       padding_token="[PAD]"
   )
   df_phonemized.to_csv("path_to_save_phonemized_data.csv", index=False)
   ```

   Check the `phonemizer.py` file for more details on the function parameters.

3. **Customize Parameters**

   - `language`: 3-letter ISO code (e.g., "fra" for French, "ita" for Italian).
   - `undefined_token`: Token for unknown or special words (default: `[UNK]`).
   - `padding_token`: Token for spacing between phonemes (default: `[PAD]`).

4. **Phoneme Vocabulary Generation**
   - After phonemizing, all unique phonemes are extracted and saved to `custom_tokenizer/vocab.json`.

## Requirements

- `transformers`
- `segments`
- `pandas`
- `numpy`
- `tqdm`
The goal is to classify kid speech as correct or incorect from reading exercises.

## Intuition behind code

The proposed solution for this hackathon was:

1. Split the speech into words with Whisper, and get time ranges for each word.
2. Classify the speech into phonemes, and get the timestamp or time range for each phoneme.
3. Combine both information using times. Retrieve phonemes for each word.
4. Classify a word as correct or incorrect based on phonemes.

## Important files

- The `Whisper` folder has information for Whisper, see [Whisper/README](./Whisper/README.md).
- For the phoneme classification, everything is in `microsoft_wavlmbaseplus.ipynb`.
  - As a bonus step, you may want to get phonemes from raw texts. This is where `ipa_encoder.py` comes in.
- The combining block is in `Combining_block`.
- The final classification comes from `SpeechClassifier.ipynb`.

If you want to run the complete pipeline, good luck! Most of the code is scattered around places and needs consequent clean-up.
