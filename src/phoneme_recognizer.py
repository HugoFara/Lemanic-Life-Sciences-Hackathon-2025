import torch

class PhonemeRecognizer(torch.nn.Module):
    def __init__(self, wavlm_model, linear_mapper_model):
        """
        Create the new model out of a combination of both models.
        
        :param wavlm_model: Features extraction performing speech-to-features.
        :param linear_mapper_model: Second model to perform features-to-phonemes.
        """
        super().__init__()
        self.wavlm = wavlm_model

        # A dropout layer is already in the linear_mapper, otherwise add it here

        # Linear layer to map from WavLM hidden states to phoneme classes
        self.phoneme_classifier = linear_mapper_model

    def forward(self, input_values, attention_mask, language):
        """
        Classify audio to a chain of phonemes of the same length.
        """
        # Get WavLM embeddings
        features = self.wavlm(input_values=input_values, attention_mask=attention_mask)

        # Apply the linear layer to get logits for each time step
        logits = self.phoneme_classifier(features.last_hidden_state, language)

        return logits
    
    def classify_to_phonemes(self, log_probs):

        # Simple greedy decoding (for demonstration)
        # In a real system, you would use beam search with ctcdecode
        return self.phoneme_classifier.classify_to_phonemes(log_probs)


    def recognize(self, inputs):
        """Perform phoneme recognition."""
        self.eval()
        with torch.no_grad():
            # Forward pass to get log probabilities
            log_probs = self(inputs)

            return self.classify_to_phonemes(log_probs)

    def tokenize(self, char_list, lenient=False):
        """
        Go from a list of characters to a list of indices.
        
        :param list[str] char_list: Characters top be mapped.
        :param bool lenient: If True, characters not in vocab are mapped to [UNK] 
        """
        return self.phoneme_classifier.tokenize(char_list, lenient)
    
    def get_embedding(self, char_list_batch):
        max_tokens = max(map(len, char_list_batch))
        out_tensor = torch.zeros((len(char_list_batch), max_tokens, len(self.phoneme_classifier.phonemes_dict)))
        for i, char_list in enumerate(char_list_batch):
            tokens = self.tokenize(char_list)
            for j, token_id in enumerate(tokens):
                out_tensor[i, j, token_id] = 1
        return out_tensor