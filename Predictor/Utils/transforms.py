import torchaudio as ta


transform = ta.transforms.Compose(
    ta.transforms.MFCC(sr=16000, n_mfcc=80, log_mels=True, melkwargs={'ws':25})
)
