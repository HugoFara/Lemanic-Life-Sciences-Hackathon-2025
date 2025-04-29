import torch
import torch.nn as nn
import torchaudio
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
original_model = torch.load("WavLM-Base+.pt")

cfg = WavLMConfig(original_model['cfg'])
model = WavLM(cfg)
model.load_state_dict(original_model['model'])
model.eval()


# Define a new model class
class WavLMWithLinear(nn.Module):
    def __init__(self, base_model, output_dim):
        super(WavLMWithLinear, self).__init__()
        self.base_model = base_model
        # Replace 768 with the actual output feature size
        self.linear = nn.Linear(base_model.embed, output_dim)

    def forward(self, x):
        features = self.base_model(x)
        # If features are a tensor, pass directly
        output = self.linear(features)
        return output


model_with_linear = WavLMWithLinear(model, output_dim=10)


def get_last_layer(wav_input_16khz=None):
    # extract the representation of last layer
    if wav_input_16khz is None:
        wav_input_16khz = torch.randn(1, 10000)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(
            wav_input_16khz, wav_input_16khz.shape
        )
    rep = model.extract_features(wav_input_16khz)[0]
    return rep


def get_each_layer(wav_input_16khz=None):
    # extract the representation of each layer
    if wav_input_16khz is None:
        wav_input_16khz = torch.randn(1, 10000)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(
            wav_input_16khz, wav_input_16khz.shape
        )
    _rep, layer_results = model.extract_features(
        wav_input_16khz,
        output_layer=model.cfg.encoder_layers,
        ret_layer_results=True
    )[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
    return layer_reps


model_with_linear.eval()
waveform, sampling_rate = torchaudio.load("test.wav")
# print(get_last_layer(waveform).shape)
print(model_with_linear(waveform))
