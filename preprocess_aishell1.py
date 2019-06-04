from data.data_collector.ai_shell_1 import CollectorAiShell1
from data.extract_aishell1 import extract_aishell1
from Predictor.data_handler.data_config import DataConfigAiShell1
import fire
import shutil


VOCAB = DataConfigAiShell1.vocab_path
COLLECTOR = DataConfigAiShell1.collector_path


def extract():
    shutil.rmtree('data/data_aishell/')
    extract_aishell1('data/data_aishell.tgz', 'data/')


def build():
    collector = CollectorAiShell1()
    collector.build_vocab(VOCAB)
    collector.save(COLLECTOR)


def pipeline():
    extract()
    build()


if __name__ == '__main__':
    fire.Fire(pipeline)

