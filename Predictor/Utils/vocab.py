import torch as t
from collections import Counter
from tqdm import tqdm


def tokenize_fn(s):
    s = [i for i in s]
    return s


class Vocab:
    def __init__(self, tokenize_fn=tokenize_fn, load_from=None):
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.token2id = {token: index for index, token in enumerate(self.special_tokens)}
        self.id2token = [i for i in self.special_tokens]
        self.vocab_size = len(self.id2token)
        self.tokenize_fn = tokenize_fn
        self.counter = Counter()
        if load_from is not None:
            self.load_vocab(load_from)

    def add_sentance(self, sentance):
        self.counter.update(self.tokenize_fn(sentance))

    def build(self, texts, max_num=30000, min_freq=1, build_fields=None):
        for text in tqdm(texts):
            if isinstance(text, dict):
                for key in text.keys():
                    if build_fields is not None:
                        if key in build_fields:
                            self.add_sentance(text[key])
            elif isinstance(text, str):
                self.add_sentance(text)
        print(f'totally {len(self.counter)} words')
        print(f'{"*"*20}')
        self.filtered_counter = self.counter.most_common(max_num)
        for token, freq in self.filtered_counter:
            if freq >= min_freq:
                ind = len(self.token2id)
                self.token2id[token] = ind
                self.id2token.append(token)
        print(f'{len(self.id2token) - 4} word used. {(len(self.id2token) - 4) *100 / len(self.counter)}% covered')
        print(f'{"*"*20}')
        self.vocab_size = len(self.id2token)

    def convert(self, texts, bos_eos_key=None):
        all = []
        for text in tqdm(texts):
            if isinstance(text, dict):
                sample = {}
                for key in text.keys():
                    if key == bos_eos_key:
                        sample[key] = self.numericalize(text[key], use_bos=True, use_eos=True)
                    else:
                        sample[key] = self.numericalize(text[key])
                all.append(sample)
            elif isinstance(text, str):
                all.append(self.numericalize(text))
        return all

    def dump_vocab(self, save_path):
        vocab = {'token2id': self.token2id,
                 'id2token': self.id2token,
                 'counter': self.counter,
                 'filtered_counter': self.filtered_counter}
        t.save(vocab, save_path)
        print(f'vocab saved to {save_path}')
        print(f'{"*"*20}')

    def load_vocab(self, save_path):
        vocab = t.load(save_path)
        self.token2id = vocab['token2id']
        self.id2token = vocab['id2token']
        self.counter = vocab['counter']
        self.filtered_counter = vocab['filtered_counter']
        self.vocab_size = len(self.id2token)
        print(f'vocab loaded from {save_path}')
        print(f'vocab size {len(self.id2token)}')
        print(f'{"*"*20}')

    def str2num(self, string, use_bos=False, use_eos=False):
        """
        str2num
        """
        tokens = []
        unk_idx = self.token2id[self.unk_token]

        if use_bos:
            tokens.append(self.bos_token)

        tokens += self.tokenize_fn(string)

        if use_eos:
            tokens.append(self.eos_token)
        indices = [self.token2id.get(tok, unk_idx) for tok in tokens]
        return indices

    def num2str(self, number, oov_vocab=None):
        """
        num2str
        """

        tokens = [self.id2token[x] if x < len(self.id2token) else oov_vocab[str(x)] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text)
        return text


    def numericalize(self, strings, use_bos=False, use_eos=False):
        """
        numericalize
        """
        if isinstance(strings, str):
            return self.str2num(strings, use_bos, use_eos)
        else:
            return [self.numericalize(s, use_bos, use_eos) for s in strings]

    def denumericalize(self, numbers, oov_vocab=None):
        """
        denumericalize
        """
        if isinstance(numbers, t.Tensor):
            with t.cuda.device_of(numbers):
                numbers = numbers.tolist()
        # if self.sequential:
        if not isinstance(numbers[0], list):
            return self.num2str(numbers, oov_vocab)
        else:
            return [self.denumericalize(x, oov_vocab) for x in numbers]
        # else:
        #     if not isinstance(numbers, list):
        #         return self.num2str(numbers)
        #     else:
        #         return [self.denumericalize(x) for x in numbers]

    @property
    def pad_index(self):
        return self.token2id[self.pad_token]

    @property
    def unk_index(self):
        return self.token2id[self.unk_token]

    @property
    def bos_index(self):
        return self.token2id[self.bos_token]

    @property
    def eos_index(self):
        return self.token2id[self.eos_token]
