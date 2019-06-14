import fire
from Predictor.data_handler import AudioParser, Vocab
from Predictor import Models
from Predictor.Bases import BaseConfig


class InferenceConfig(BaseConfig):
    model_name = None


def load_model(path):
    model = None


def main(**kwargs):
    inference_config = InferenceConfig()
    inference_config.fn_build(kwargs)
    model = load_model(inference_config.model_name)
    model_config = model.config
    model_vocab = model.vocab
    parser = AudioParser(
        sample_rate=model_config.sample_rate, n_mels=model_config.n_mels, window_size=model_config.window_size
    )
    audio_inputer = None









if __name__ == '__main__':
    fire.Fire()