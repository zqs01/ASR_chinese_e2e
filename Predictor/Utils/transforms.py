import torchaudio as ta


transform = ta.transforms.Compose(
    ta.transforms.MFCC
)