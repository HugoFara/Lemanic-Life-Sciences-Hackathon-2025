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
