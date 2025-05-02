# 📁 Step 1: Whisper

This subdirectory contains a pipeline for transcribing and cleaning speech data using OpenAI's Whisper model.

- It transcribes each audio file listed in a CSV into a sequence of words with precise start and end timestamps.
- The transcription output is then cleaned by removing trailing punctuation, stutters, repeated words, and short noisy fragments.
- Language-specific filtering is applied when needed: keeping only transcriptions with exactly 12 words for Italian, or exactly 1 word for French.
- The final cleaned dataset is saved as a CSV, and a histogram is generated to visualize the distribution of word counts before and after cleaning.
