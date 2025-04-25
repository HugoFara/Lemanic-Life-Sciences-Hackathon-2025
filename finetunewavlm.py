import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, AutoFeatureExtractor
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# ————————————————————————————————————————————————————————————————————————
# PhonemeRecognizer: WavLM + CTC for phoneme speech recognition
# ————————————————————————————————————————————————————————————————————————

# Define a list of English phonemes (ARPABET format) + blank token for CTC
PHONEMES = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
           'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P',
           'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SP']
# Add blank token for CTC
PHONEME_DICT = {p: i for i, p in enumerate(['<blank>'] + PHONEMES)}
NUM_PHONEMES = len(PHONEME_DICT)

class PhonemeRecognizer(nn.Module):
    def __init__(self, wavlm_model, num_phonemes=NUM_PHONEMES):
        super().__init__()
        self.wavlm = wavlm_model

        # Get the hidden size from the WavLM model
        hidden_size = self.wavlm.config.hidden_size

        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(0.1)

        # Linear layer to map from WavLM hidden states to phoneme classes (including blank)
        self.phoneme_classifier = nn.Linear(hidden_size, num_phonemes)

    def forward(self, inputs):
        # Get WavLM embeddings
        outputs = self.wavlm(**inputs)
        hidden_states = outputs.last_hidden_state

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Apply the linear layer to get logits for each time step
        logits = self.phoneme_classifier(hidden_states)

        # Apply log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def recognize(self, inputs, beam_width=100):
        """Perform phoneme recognition with beam search decoding"""
        self.eval()
        with torch.no_grad():
            # Forward pass to get log probabilities
            log_probs = self(inputs)

            # Convert to CPU for decoding
            log_probs_cpu = log_probs.cpu().detach().numpy()

            # Simple greedy decoding (for demonstration)
            # In a real system, you would use beam search with ctcdecode
            predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()

            # Convert to phoneme sequences with CTC decoding rules (merge repeats, remove blanks)
            phoneme_sequences = []
            for pred_seq in predictions:
                seq = []
                prev = -1
                for p in pred_seq:
                    # Skip blanks (index 0) and repeated phonemes (CTC rules)
                    if p != 0 and p != prev:
                        # Convert index back to phoneme
                        phoneme = list(PHONEME_DICT.keys())[list(PHONEME_DICT.values()).index(p)]
                        seq.append(phoneme)
                    prev = p
                phoneme_sequences.append(seq)

            return phoneme_sequences

def train_phoneme_classifier(model, train_dataset, feature_extractor, device='cuda' if torch.cuda.is_available() else 'cpu',
                             num_epochs=5, batch_size=8, learning_rate=3e-5):
    """Train the phoneme classifier layer of the model using CTC loss"""
    # Move model to device
    model.to(device)

    # Prepare optimizer - only train the classifier layer, freeze WavLM
    for param in model.wavlm.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.phoneme_classifier.parameters(), lr=learning_rate)

    # CTC loss
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    # Setup data loader
    def collate_fn(batch):
        # Process audio
        audio_inputs = feature_extractor(
            [item["audio"]["array"] for item in batch],
            sampling_rate=batch[0]["audio"]["sampling_rate"],
            return_tensors="pt",
            padding=True,
        )

        # Process phoneme targets
        # This is a placeholder - in a real scenario, you would have phoneme targets
        # For this example, we'll just use random targets

        # Calculate proper input length after WavLM processing
        # WavLM downsamples by a factor of 320, so divide by 320
        # Using a safe calculation to avoid errors
        with torch.no_grad():
            # Forward pass through WavLM to get actual sequence length
            temp_outputs = model.wavlm(**{k: v.to(device) for k, v in audio_inputs.items()})
            sequence_lengths = temp_outputs.last_hidden_state.shape[1]
            # Create input_lengths tensor with the correct sequence length
            input_lengths = torch.IntTensor([sequence_lengths] * len(batch))

        # Create dummy target sequences (these would be your actual phoneme sequences)
        # In a real implementation, you would convert text/phonemes to indices using PHONEME_DICT
        target_lengths = torch.IntTensor([10] * len(batch))  # Example length
        targets = torch.randint(1, NUM_PHONEMES, (len(batch), 10))  # Random phoneme indices

        return audio_inputs, targets, input_lengths, target_lengths

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (audio_inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            # Move inputs to device
            for k, v in audio_inputs.items():
                audio_inputs[k] = v.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            log_probs = model(audio_inputs)

            # Permute log_probs to match CTCLoss input requirements: (T, N, C)
            log_probs = log_probs.permute(1, 0, 2)

            # Calculate loss
            try:
                loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                total_loss += loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Log probs shape: {log_probs.shape}, Targets shape: {targets.shape}")
                print(f"Input lengths: {input_lengths}, Target lengths: {target_lengths}")
                continue

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    print("Training complete!")
    return model

# ————————————————————————————————————————————————————————————————————————
# Method A: Using the PhonemeRecognizer for speech-to-phoneme ASR
# ————————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    # 1. Load the feature extractor and model
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

    # Create the phoneme recognizer with the WavLM model
    phoneme_recognizer = PhonemeRecognizer(wavlm_model)

    # 2. Load dataset for training and testing
    print("Loading dataset...")
    ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

    # Split dataset for training and testing
    train_size = int(0.8 * len(ds))
    train_ds = ds.select(range(train_size))
    test_ds = ds.select(range(train_size, len(ds)))

    # 3. Train the phoneme classifier
    print("\nStarting training phase...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set smaller values for faster training in this example
    try:
        phoneme_recognizer = train_phoneme_classifier(
            model=phoneme_recognizer,
            train_dataset=train_ds,
            feature_extractor=feature_extractor,
            device=device,
            num_epochs=2,  # Reduced for demonstration
            batch_size=2,   # Smaller batch size to avoid memory issues
            learning_rate=5e-4
        )
        print("Phoneme classifier training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        # Continue with the untrained model if training fails
        print("Proceeding with evaluation using untrained model...")

    # 4. Prepare test sample for inference
    print("\nStarting inference phase...")
    test_sample = test_ds[0]
    audio_sample = test_sample["audio"]["array"]
    sr = test_sample["audio"]["sampling_rate"]

    # 5. Preprocess test sample
    inputs = feature_extractor(
        audio_sample,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 6. Switch to evaluation mode
    phoneme_recognizer.eval()

    # 7. Inference for phoneme recognition
    with torch.no_grad():
        try:
            # Get phoneme log probabilities
            log_probs = phoneme_recognizer(inputs)

            # Recognize phoneme sequence
            phoneme_sequences = phoneme_recognizer.recognize(inputs)

            # 8. Print output
            print("Log probabilities shape:", log_probs.shape)  # (batch_size, seq_len, num_phonemes)
            print("Recognized phoneme sequence:", phoneme_sequences[0])
            print("Transcript for reference:", test_sample["text"])
        except Exception as e:
            print(f"Error during inference: {e}")
