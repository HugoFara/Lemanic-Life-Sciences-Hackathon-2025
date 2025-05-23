import torch
import torch.nn.functional as F

class PhonemeMapper(torch.nn.Module):
    def __init__(self, features_size, phonemes_dict):
        super().__init__()
        self.phonemes_dict = phonemes_dict
        num_phonemes = len(phonemes_dict)

        # Add a dropout layer for regularization
        self.dropout = torch.nn.Dropout(0.1)

        self.batch_normalize = torch.nn.BatchNorm1d(features_size)

        # Linear layer to map from WavLM hidden states to phoneme classes (including blank)
        self.phoneme_classifier = torch.nn.Linear(features_size, num_phonemes)

    def language_classifer(self, language):
        """
        Return a float identifying each known language.

        "fr" has value of 0, "it" a value of one.
        Other languages will have a value increasing in lexicographic order.  
        
        :param str language: Language to identify, should be two letters.
        :return float: Unique identifier, between 0 and 1.
        """
        if language == "fr":
            return 0
        if language == "it":
            return 1
        
        # Some random code to encode a two-letter language between 0 and 1
        # "aa" should be 0+1=1 and "zz" should be 1+2=3
        codes = (
            (ord(letter) - ord("a")) / (ord("z") - ord("a")) + i
            for i, letter in enumerate(language)
        )
        # Transform to [0, 1]
        return (sum(codes) - 1) / 2

    def forward(self, input_values, language):
        """
        Forward pass, classify input features to a chain of phonemes.
        
        :param torch.Tensor input_values: Extracted features batch to map.
        :param str | Iterable language: Language for the batch, or the language for each element in the batch.
        :return torch.Tensor: Log of the probabilities for each phoneme.
        """
        input_batch = torch.empty(
            (input_values.shape[0], input_values.shape[1], input_values.shape[2] + 1),
            dtype=input_values.dtype,
            device=input_values.device
        )

        
        if isinstance(language, str):
            lang_val = torch.full((input_values.shape[1]), self.language_classifer(language))
        else:
            lang_val = (
                torch
                .tensor([[self.language_classifer(lang)] for lang in language])
                .repeat((1, input_batch.shape[1]))
            )

        input_batch[:, :, 0] = lang_val
        input_batch[:, :, 1:] = input_values

        # Apply dropout
        hidden_states = self.dropout(input_batch)

        # Normalize
        hidden_states = self.batch_normalize(hidden_states)

        # Apply the linear layer to get logits for each time step
        logits = self.phoneme_classifier(hidden_states)

        return logits
    
    def tokenize(self, char_list, lenient=False):
        """
        Go from a list of characters to a list of indices.
        
        :param list[str] char_list: Characters top be mapped.
        :param bool lenient: If True, characters not in vocab are mapped to [UNK] 
        """
        if not lenient:
            return torch.tensor([self.phonemes_dict[x] for x in char_list])
        
        return torch.tensor([
            self.phonemes_dict[x] if x in self.phonemes_dict else self.phonemes_dict["[UNK]"]
            for x in char_list
        ])

    def classify_to_phonemes_raw(self, log_probs):
        """Phoneme classification without CTC."""
        # Simple greedy decoding (for demonstration)
        predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()

        # Convert to phoneme sequences with CTC decoding rules (merge repeats, remove blanks)
        phoneme_sequences = []
        vocab = [""] * len(self.phonemes_dict)
        for phoneme, key in self.phonemes_dict.items():
            vocab[key] = phoneme

        for pred_seq in predictions:
            phoneme_sequences.append([vocab[p] for p in pred_seq])

        return phoneme_sequences
    
    def apply_ctc_collapse(self, phonemes_sequences):
        predictions = []

        for pred_seq in phonemes_sequences:
            seq = []
            prev = -1
            for phoneme in pred_seq:
                # Skip blanks (index 0) and repeated phonemes (CTC rules)
                if phoneme != "[PAD]" and phoneme != prev:
                    # Convert index back to phoneme
                    seq.append(phoneme)
                prev = phoneme
            predictions.append(seq)
        return predictions
    
    def classify_to_phonemes(self, log_probs):
        # Convert to phoneme sequences with CTC decoding rules (merge repeats, remove blanks)
        phoneme_sequences = self.classify_to_phonemes_raw(log_probs)

        return self.apply_ctc_collapse(phoneme_sequences)
    