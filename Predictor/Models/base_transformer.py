import math
import torch as t
from dataclasses import dataclass

from Predictor.Utils import Pack
from Predictor.Bases import BaseModel
from Predictor.Bases import BaseConfig
from Predictor.data_handler import Masker
from Predictor.Utils.loss import calculate_loss
from Predictor.Utils.score import calculate_cer



"""
        pack = Pack()
        wave = [i[0] for i in batch]
        tgt = [i[1] for i in batch]
        wave, wave_len = Padder.pad_tri(wave, 0)
        tgt, tgt_len = Padder.pad_two(tgt, 0)
        pack.add(wave=wave, tgt=tgt.long(), wave_len=wave_len, tgt_len=tgt_len)
        return pack
"""

class Transformer(BaseModel):
    def __init__(self, config, vocab):
        super(Transformer, self).__init__()
        self.config = config
        self.vocab = vocab
        self.input_linear = t.nn.Linear(config.n_mels, config.d_model)
        self.position_encoder = PositionalEncoding(config.d_model)
        self.word_embeder = t.nn.Embedding(vocab.vocab_size, config.d_model)
        self.encoder = Encoder(input_size=config.d_model, hidden_size=config.hidden_size, ff_size=config.ff_size,
                               num_head=config.num_head, dropout=config.dropout, layer_num=config.layer_num)
        self.decoder = Decoder(input_size=config.d_model, hidden_size=config.hidden_size, ff_size=config.ff_size,
                               num_head=config.num_head, dropout=config.dropout, layer_num=config.layer_num)
        self.output_linear = t.nn.Linear(config.d_model, vocab.vocab_size)
        self.output_linear.weight = self.word_embeder.weight
        self.x_logit_scale = (config.d_model ** -0.5)

    def forward(self, input):
        wave, wave_len, text, text_len = input.wave, input.wave_len, input.tgt, input.tgt_len
        # build masks
        wave_pad_mask = Masker.get_pad_mask(wave.sum(-1), wave_len)
        wave_self_attention_mask = Masker.get_attn_pad_mask(wave_pad_mask, wave_pad_mask.size(1))
        text_pad_mask = Masker.get_pad_mask(text, text_len)
        text_self_attention_mask = Masker.get_attn_pad_mask(text_pad_mask, text_pad_mask.size(1))
        text_subsquence_mask = Masker.get_subsequent_mask(text)
        text_self_attention_mask = text_self_attention_mask.byte() * text_subsquence_mask
        dot_attention_mask = Masker.get_attn_key_pad_mask(wave_pad_mask, input.tgt)


        wave_feature = self.input_linear(wave)
        wave_feature = wave_feature + self.position_encoder(wave_feature)
        wave_feature = self.encoder(wave_feature, wave_pad_mask, wave_self_attention_mask)

        text_feature = self.word_embeder(text) * self.x_logit_scale
        text_feature = text_feature + self.position_encoder(text_feature)
        #feature, encoder_output, pad_mask, self_attention_mask, dot_attention_mask
        output = self.decoder(text_feature, wave_feature, text_pad_mask, text_self_attention_mask, dot_attention_mask)
        output = self.output_linear(output)
        return output

    def cal_metrics(self, output, input):
        output_id = output.topk(1)[1].squeeze(-1)
        pack = Pack()
        loss = calculate_loss(output, input.tgt)
        assert not t.isinf(loss)
        score = calculate_cer(output_id, input.tgt)
        pack.add(loss=loss, score=score)
        return pack

    def iterate(self, input, optimizer=None, is_train=True):
        output = self.forward(input)
        metrics = self.cal_metrics(output, input)

        return metrics

    def greedy_search(self):
        pass

    def beam_search(self):
        pass

    @classmethod
    def get_default_config(cls):
        @dataclass
        class ModelConfig(BaseConfig):
            d_model = 256
            hidden_size = 128
            ff_size = 512
            num_head = 8
            dropout = 0.1
            layer_num = 6

        return ModelConfig

class PositionalEncoding(t.nn.Module):
    """
    Positional Encoding class
    """

    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()
        pe = t.zeros(max_length, dim_model, requires_grad=False)
        position = t.arange(0, max_length).unsqueeze(1).float()
        exp_term = t.exp(t.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = t.sin(position * exp_term)  # take the odd (jump by 2)
        pe[:, 1::2] = t.cos(position * exp_term)  # take the even (jump by 2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        args:
            input: B x T x D
        output:
            tensor: B x T
        """
        return self.pe[:, :input.size(1)]


class Encoder(t.nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_head, dropout, layer_num):
        super(Encoder, self).__init__()
        self.layer_list = t.nn.ModuleList(
            EncoderLayer(input_size, hidden_size, ff_size, num_head, dropout) for _ in range(layer_num)
        )

    def forward(self, feature, pad_mask, self_attention_mask):
        for layer in self.layer_list:
            feature = layer(feature, pad_mask, self_attention_mask)
        return feature


class EncoderLayer(t.nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_head, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.dropout1 = t.nn.Dropout(dropout)
        self.layer_norm1 = t.nn.LayerNorm(input_size)
        self.feed_forward = FeedForward(input_size, ff_size, dropout)
        self.dropout2 = t.nn.Dropout(dropout)
        self.layer_norm2 = t.nn.LayerNorm(input_size)

    def forward(self, feature, pad_mask, self_attention_mask):
        pad_mask = pad_mask.unsqueeze(-1)
        feature_ = self.self_attention(feature, feature, feature, self_attention_mask)
        feature = self.layer_norm1(self.dropout1(feature_) + feature)
        feature = feature * pad_mask
        feature_ = self.feed_forward(feature)
        feature = self.layer_norm2(self.dropout2(feature_) + feature)
        feature = feature * pad_mask
        return feature


class Decoder(t.nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_head, dropout, layer_num):
        super(Decoder, self).__init__()
        self.layer_list = t.nn.ModuleList([
            DecoderLayer(input_size, hidden_size, ff_size, num_head, dropout) for _ in range(layer_num)
        ])

    def forward(self, feature, encoder_output, pad_mask, self_attention_mask, dot_attention_mask):
        for layer in self.layer_list:
            feature = layer(feature, encoder_output, pad_mask, self_attention_mask, dot_attention_mask)
        return feature


class DecoderLayer(t.nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_head, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attnention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.dropout1 = t.nn.Dropout(dropout)
        self.layer_norm1 = t.nn.LayerNorm(input_size)
        self.dot_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.dropout2 = t.nn.Dropout(dropout)
        self.layer_norm2 = t.nn.LayerNorm(input_size)
        self.feed_forward = FeedForward(input_size, ff_size, dropout)
        self.dropout3 = t.nn.Dropout(dropout)
        self.layer_norm3 = t.nn.LayerNorm(input_size)

    def forward(self, feature, encoder_output, pad_mask, self_attention_mask, dot_attention_mask):
        pad_mask = pad_mask.unsqueeze(-1)
        feature_ = self.self_attnention(feature, feature, feature, self_attention_mask)
        feature = self.layer_norm1(self.dropout1(feature_) + feature)
        feature = feature * pad_mask
        feature_ = self.dot_attention(feature, encoder_output, encoder_output, dot_attention_mask)
        feature = self.layer_norm2(self.dropout2(feature_) + feature)
        feature = feature * pad_mask
        feature_ = self.feed_forward(feature)
        feature = self.layer_norm3(self.dropout3(feature_) + feature)
        feature = feature * pad_mask
        return feature


class FeedForward(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear = t.nn.Sequential(
            t.nn.Conv1d(input_size, hidden_size, 1),
            t.nn.ReLU(True),
            t.nn.Dropout(dropout),
            t.nn.Conv1d(hidden_size, input_size, 1),
        )
        t.nn.init.xavier_normal_(self.linear[0].weight)
        t.nn.init.xavier_normal_(self.linear[-1].weight)

    def forward(self, feature):
        net = feature.transpose(-1, -2)
        net = self.linear(net)
        net = net.transpose(-1, -2)
        return net


class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.query_linear = t.nn.Linear(input_size, num_head * hidden_size)
        self.key_linear = t.nn.Linear(input_size, num_head * hidden_size)
        self.value_linear = t.nn.Linear(input_size, num_head * hidden_size)
        self.dot_attention = DotAttention(hidden_size, dropout)
        self.output_linear = t.nn.Linear(num_head * hidden_size, input_size)
        t.nn.init.xavier_normal_(self.query_linear.weight)
        t.nn.init.xavier_normal_(self.key_linear.weight)
        t.nn.init.xavier_normal_(self.value_linear.weight)
        t.nn.init.xavier_normal_(self.output_linear.weight)

    def reshape(self, tensor, batch_size, seq_length, num_head, hidden_size):
        # B, L, N*H
        tensor = tensor.view(batch_size, seq_length, num_head, hidden_size)
        tensor = tensor.permute(2, 0, 1, 3).contiguous()
        tensor = tensor.view(num_head * batch_size, seq_length, hidden_size)
        return tensor

    def forward(self, query, key, value, attention_mask=None):
        batch_size, query_length, _ = query.size()
        _, key_length, _ = key.size()
        query_ = self.query_linear(query)
        key_ = self.key_linear(key)
        value_ = self.value_linear(value)

        query_ = self.reshape(query_, batch_size=batch_size, seq_length=query_length, num_head=self.num_head,
                              hidden_size=self.hidden_size)
        key_ = self.reshape(key_, batch_size=batch_size, seq_length=key_length, num_head=self.num_head,
                            hidden_size=self.hidden_size)
        value_ = self.reshape(value_, batch_size=batch_size, seq_length=key_length, num_head=self.num_head,
                              hidden_size=self.hidden_size)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(self.num_head, 1, 1)
        output, _ = self.dot_attention(query_, key_, value_, attention_mask)
        output = output.view(self.num_head, batch_size, query_length, -1)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, query_length, -1)
        output = self.output_linear(output)
        return output


class DotAttention(t.nn.Module):
    def __init__(self, hidden_size, dropout):
        super(DotAttention, self).__init__()
        self.C = math.sqrt(hidden_size)
        self.dropout = t.nn.Dropout(dropout)
        self.softmax = t.nn.Softmax(2)

    def forward(self, query, key, value, attention_mask=None):
        attention = t.bmm(query, key.transpose(1, 2)) / self.C
        attention = self.softmax(attention)
        if attention_mask is not None:
            attention = attention.masked_fill(attention_mask == 0, -math.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        output = t.bmm(attention, value)
        return output, attention
#
# if __name__ == '__main__':
#     d_model = 5
#     hidden_size = 512
#     num_head = 8
#     dropout = 0.1
#     layer_num = 6
#     ff_size = 512
#
#     embedding = t.nn.Embedding(300, 5)
#
#
#     #
#     inputs = t.Tensor([[2, 5, 0]]).long()
#     inputs2 = t.Tensor([[2,4,5,7,0]]).long()
#
#
#     enc_feature = embedding(inputs)
#     dec_feature = embedding(inputs2)
#     print(inputs.shape)
#     from Predictor.data_handler import Masker
#     print(Masker.get_pad_mask(inputs, pad_idx=0))
#     print(Masker.get_attn_key_pad_mask(inputs, inputs, 0))
#     print(Masker.get_subsequent_mask(inputs))
#     print(Masker.get_attn_pad_mask(Masker.get_pad_mask(inputs, pad_idx=0), expand_length=3))
#
#
#     query = inputs
#     key = inputs2
#     pad_mask = Masker.get_pad_mask(query)
#     self_attention_mask = Masker.get_attn_pad_mask(pad_mask, pad_mask.size(1))
#     print(pad_mask.shape)
#     print(self_attention_mask.shape)
#
#     encoder = Encoder(
#         num_head=num_head, dropout=dropout, layer_num=layer_num, hidden_size=hidden_size, input_size=d_model,
#         ff_size=ff_size)
#
#     print(encoder(enc_feature, pad_mask, self_attention_mask))
#     o_feature = encoder(enc_feature, pad_mask, self_attention_mask)
#     decoder = Decoder(num_head=num_head, dropout=dropout, layer_num=layer_num, hidden_size=hidden_size, input_size=d_model,
#         ff_size=ff_size)
#
#     dec_pad_mask = Masker.get_pad_mask(inputs2)
#     dot_attention_mask = Masker.get_attn_key_pad_mask(inputs, inputs2)
#     dec_self_attention_mask = Masker.get_attn_pad_mask(dec_pad_mask, dec_pad_mask.size(1))
#     # feature, encoder_output, pad_mask, self_attention_mask, dot_attention_mask
#     print(decoder(dec_feature, o_feature, dec_pad_mask, dec_self_attention_mask, dot_attention_mask))
#     print(inputs)
#     print(inputs2)
#     print(decoder(dec_feature, o_feature, dec_pad_mask, dec_self_attention_mask, dot_attention_mask).shape)
