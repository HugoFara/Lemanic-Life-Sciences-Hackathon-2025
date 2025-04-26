# Required imports
import os
import json
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Union
from collections import Counter

from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
)
from evaluate import load as load_metric

# --- Paths ---
audio_paths = {
    "FR": "/pvc/scratch/speech_recognition/Hackathon_ASR/2_Audiofiles/Phoneme_Deletion_FR_T1",
    "IT": "/pvc/scratch/speech_recognition/Hackathon_ASR/2_Audiofiles/Decoding_IT_T1",
}
csv_paths = {
    "FR": "2_Phoneme_Deleletion_ground_truth_FR.csv",
    "IT": "2_Decoding_ground_truth_IT.csv",
}
vocab_json_path = "unique_phonemes_FR_IT.json"


# --- Utility: check .wav validity ---
def is_valid_wav(path):
    try:
        sf.info(path)
        return True
    except:
        return False


# --- Load and combine datasets ---
combined_filepaths, combined_phonemes, combined_langs, combined_filenames = (
    [],
    [],
    [],
    [],
)
skipped_files = []

for lang in ["FR", "IT"]:
    print(f"\n🔍 Loading {lang} data...")
    df = pd.read_csv(csv_paths[lang])
    df["full_path"] = df["file_name"].apply(
        lambda x: os.path.join(audio_paths[lang], x)
    )

    valid_mask = df["full_path"].apply(lambda x: os.path.exists(x) and is_valid_wav(x))
    skipped = df.loc[~valid_mask]

    if not skipped.empty:
        print(f"Skipped {len(skipped)} invalid or missing audio files for {lang}:")
        for fname in skipped["file_name"].tolist()[:10]:  # show first 10
            print(f"   - {fname}")
        skipped_files.extend(skipped["file_name"].tolist())

    df = df[valid_mask]
    combined_filepaths.extend(df["full_path"])
    combined_phonemes.extend(df["phoneme"])
    combined_filenames.extend(df["file_name"])
    combined_langs.extend([lang] * len(df))

print(f"Total skipped audio files: {len(skipped_files)}")
with open("skipped_audio_files.txt", "w") as f:
    f.writelines(f"{name}\n" for name in skipped_files)

dataset_dict = {
    "audio_file": combined_filepaths,
    "phonetic": combined_phonemes,
    "file_name": combined_filenames,
    "language": combined_langs,
}
dataset = Dataset.from_dict(dataset_dict)

# --- Load vocab ---
with open(vocab_json_path, "r") as f:
    vocab = json.load(f)

if all(k.isdigit() for k in vocab.keys()):
    vocab = {v: int(k) for k, v in vocab.items()}
else:
    vocab = {k: v for k, v in vocab.items()}

vocab_ids = {int(v): k for k, v in vocab.items()}
vocab_list = [vocab_ids[i] for i in sorted(vocab_ids)]

# --- Save tokenizer ---
custom_tokenizer_path = "./custom_tokenizer"
os.makedirs(custom_tokenizer_path, exist_ok=True)
with open(
    os.path.join(custom_tokenizer_path, "vocab.json"), "w", encoding="utf-8"
) as f:
    json.dump({p: i for i, p in enumerate(vocab_list)}, f, ensure_ascii=False, indent=2)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
    custom_tokenizer_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# --- Cast to waveform ---
print("Casting audio column to waveform...")
dataset = dataset.cast_column("audio_file", Audio(sampling_rate=16000)).rename_column(
    "audio_file", "audio"
)


# --- Prepare samples safely ---
def prepare_sample(batch):
    try:
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["phonetic"]).input_ids
    except Exception as e:
        print(f"Error processing {batch.get('file_name', 'unknown')}: {e}")
        batch["input_values"] = None
        batch["input_length"] = None
        batch["labels"] = None
    return batch


print("Preprocessing samples...")
dataset = dataset.map(prepare_sample, num_proc=1)
dataset = dataset.filter(lambda x: x["input_values"] is not None)
dataset = dataset.shuffle(seed=42)

print("\n Final language distribution:", Counter(dataset["language"]))


# --- Data collator ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- Evaluation ---
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")


def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }


# --- Load model ---
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
)
model.freeze_feature_encoder()

# --- Training configuration ---
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=10,
    learning_rate=3e-5,
    save_total_limit=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=20,
    report_to="none",
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# --- Training ---
print("Starting training...")
trainer.train()
trainer.save_model("./fine_tuned_model")


# --- Export predictions ---
def export_predictions_split(model, dataset, output_dir="outputs"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    results_by_lang = {"FR": [], "IT": []}

    for i, row in enumerate(tqdm(dataset)):
        audio_data = row.get("audio")
        phonetic = row.get("phonetic")
        file_name = row.get("file_name")
        language = row.get("language")

        if (
            not isinstance(audio_data, dict)
            or "array" not in audio_data
            or phonetic is None
            or language not in results_by_lang
        ):
            print(f"Skipping sample {i}: bad data")
            continue

        try:
            inputs = processor(
                audio_data["array"], return_tensors="pt", sampling_rate=16000
            ).to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

            results_by_lang[language].append(
                {
                    "file_name": file_name,
                    "phoneme": phonetic,
                    "probabilities": json.dumps(probs.tolist()),
                    "timestamps": json.dumps(
                        [
                            [
                                j / model.config.vocab_size,
                                (j + 1) / model.config.vocab_size,
                            ]
                            for j in range(probs.shape[0])
                        ]
                    ),
                }
            )

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    for lang, rows in results_by_lang.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, f"output_{lang}.csv"), index=False)
        print(f" Saved {lang} predictions to {output_dir}/output_{lang}.csv")


print("Exporting predictions...")
export_predictions_split(model, dataset)
