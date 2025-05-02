import os
import json
import torch
import soundfile as sf
import numpy as np
import pandas as pd
import yaml
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


def is_valid_wav(path):
    try:
        sf.info(path)
        return True
    except:
        return False


def load_and_prepare_dataset(audio_paths, csv_paths):
    combined = {"audio_file": [], "phonetic": [], "file_name": [], "language": []}
    skipped_files = []

    for lang in ["FR", "IT"]:
        print(f"Loading {lang} data...")
        df = pd.read_csv(csv_paths[lang])
        df["full_path"] = df["file_name"].apply(
            lambda x: os.path.join(audio_paths[lang], x)
        )
        valid_mask = df["full_path"].apply(
            lambda x: os.path.exists(x) and is_valid_wav(x)
        )
        skipped = df.loc[~valid_mask]

        if not skipped.empty:
            print(f"Skipped {len(skipped)} invalid or missing audio files for {lang}:")
            for fname in skipped["file_name"].tolist()[:10]:
                print(f"   - {fname}")
            skipped_files.extend(skipped["file_name"].tolist())

        df = df[valid_mask]
        combined["audio_file"].extend(df["full_path"])
        combined["phonetic"].extend(df["phoneme"])
        combined["file_name"].extend(df["file_name"])
        combined["language"].extend([lang] * len(df))

    print(f"Total skipped audio files: {len(skipped_files)}")
    with open("skipped_audio_files.txt", "w") as f:
        f.writelines(f"{name}\n" for name in skipped_files)

    dataset = Dataset.from_dict(combined)
    dataset = dataset.cast_column(
        "audio_file", Audio(sampling_rate=16000)
    ).rename_column("audio_file", "audio")
    return dataset


def prepare_sample_fn(processor):
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

    return prepare_sample


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


def compute_metrics_fn(tokenizer):
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

    return compute_metrics


def export_predictions_split(model, dataset, processor, output_dir="outputs"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    results_by_lang = {"FR": [], "IT": []}
    for i, row in enumerate(tqdm(dataset)):
        try:
            audio_data, phonetic, file_name, language = (
                row["audio"],
                row["phonetic"],
                row["file_name"],
                row["language"],
            )
            if not isinstance(audio_data, dict) or "array" not in audio_data:
                continue
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
    for lang, rows in results_by_lang.items():
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, f"output_{lang}.csv"), index=False
        )
        print(f"Saved {lang} predictions to {output_dir}/output_{lang}.csv")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    audio_paths = {"FR": config["audio_fr"], "IT": config["audio_it"]}
    csv_paths = {"FR": config["csv_fr"], "IT": config["csv_it"]}

    dataset = load_and_prepare_dataset(audio_paths, csv_paths)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        config["tokenizer_path"],
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
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    print("Preprocessing samples...")
    dataset = dataset.map(prepare_sample_fn(processor), num_proc=1)
    dataset = dataset.filter(lambda x: x["input_values"] is not None)
    dataset = dataset.shuffle(seed=42)
    print("\nFinal language distribution:", Counter(dataset["language"]))

    model = Wav2Vec2ForCTC.from_pretrained(
        config["pretrained_model"],
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
    )
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=config["save_steps"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        save_total_limit=3,
        fp16=True,
        logging_dir=config["log_dir"],
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor),
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics_fn(tokenizer),
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(config["output_dir"])

    print("Exporting predictions...")
    export_predictions_split(model, dataset, processor)


if __name__ == "__main__":
    main()
