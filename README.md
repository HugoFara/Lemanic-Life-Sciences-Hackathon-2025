# Lemanic-Life-Sciences-Hackathon-2025

Repository for code on the Lemanic Life Sciences Hackathon 2025 submission!
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
