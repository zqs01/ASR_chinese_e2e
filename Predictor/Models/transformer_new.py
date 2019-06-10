import math
import torch as t
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F

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

class TransformerNew(BaseModel):
    def __init__(self, config, vocab):
        super(TransformerNew, self).__init__()
        self.config = config
        self.vocab = vocab
        self.input_linear = Linear(config.n_mels, config.d_model, config.dropout)

        self.position_encoder = PositionalEncoding(config.d_model)
        self.word_embeder = t.nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=0)
        self.word_embeder.weight.data.normal_(0, 0.1)

        self.encoder = Encoder(input_size=config.d_model, hidden_size=config.hidden_size, ff_size=config.ff_size,
                               num_head=config.num_head, dropout=config.dropout, num_layer=config.layer_num)
        self.decoder = Decoder(input_size=config.d_model, hidden_size=config.hidden_size, ff_size=config.ff_size,
                               num_head=config.num_head, dropout=config.dropout, num_layer=config.layer_num)
        self.output_linear = t.nn.Linear(config.d_model, vocab.vocab_size, bias=False)
        t.nn.init.xavier_normal_(self.output_linear.weight)
        self.projection_scale = config.d_model ** -0.5
        self.output_linear.weight = self.word_embeder.weight

    def forward(self, input):
        wave, wave_len, text_for_input, text_len = input.wave, input.wave_len, input.tgt_for_input, input.tgt_len
        batch_size = wave.size(0)

        # build masks
        wave_pad_mask = Masker.get_pad_mask(wave[:, :, 0], wave_len)
        wave_self_attention_mask = Masker.get_dot_attention_mask(wave_pad_mask, wave_pad_mask)
        text_pad_mask = Masker.get_pad_mask(text_for_input, text_len)
        text_self_attention_mask = Masker.get_dot_attention_mask(text_pad_mask, text_pad_mask)
        text_subsquence_mask = Masker.get_subsequent_mask(text_for_input)
        text_self_attention_mask = text_self_attention_mask * text_subsquence_mask
        dot_attention_mask = Masker.get_dot_attention_mask(text_pad_mask, wave_pad_mask)

        wave_feature = self.input_linear(wave)
        wave_position_feature = self.position_encoder(wave_feature).repeat(batch_size, 1, 1) * wave_pad_mask.unsqueeze(-1)
        encoder_output = self.encoder(wave_feature, wave_position_feature, wave_pad_mask, wave_self_attention_mask)

        text_feature = self.word_embeder(text_for_input)
        text_position_feature = self.position_encoder(text_feature).repeat(batch_size, 1, 1) * text_pad_mask.unsqueeze(-1)
        decoder_output = self.decoder(
            text_feature, encoder_output, text_pad_mask, text_position_feature, text_self_attention_mask, dot_attention_mask)
        output = self.output_linear(decoder_output) * self.projection_scale
        return output

    def cal_metrics(self, output, input):
        output_id = output.topk(1)[1].squeeze(-1)
        pack = Pack()
        loss = calculate_loss(output, input.tgt_for_metric)
        assert not t.isinf(loss)
        output_str = [self.vocab.convert_id2str(i) for i in output_id]
        tgt_str = [self.vocab.convert_id2str(i) for i in input.tgt_for_metric]
        cer = sum([calculate_cer(i[0], i[1]) for i in zip(output_str, tgt_str)]) * 100 / len(output_str)
        pack.add(loss=loss, cer=t.Tensor([cer]))
        return pack

    def iterate(self, input, optimizer=None, is_train=True):
        output = self.forward(input)
        metrics = self.cal_metrics(output, input)
        if optimizer is not None and is_train:
            optimizer.zero_grad()
            metrics.loss.backward()
            t.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
        return metrics, None

    def greedy_search(self):
        pass

    def beam_search(self):
        pass

    @classmethod
    def get_default_config(cls):
        @dataclass
        class ModelConfig(BaseConfig):
            d_model = 512
            hidden_size = 64
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



def Linear(in_features, out_features, dropout=0.):
    m = t.nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return t.nn.utils.weight_norm(m)


class Encoder(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head, num_layer, ff_size):
        super(Encoder, self).__init__()
        self.encoder_list = t.nn.ModuleList(
            [EncoderLayer(input_size, hidden_size, dropout, num_head, ff_size) for i in range(num_layer)]
        )
        self.layer_norm = t.nn.LayerNorm(input_size)

    def forward(self, feature, position_feature, pad_mask, self_attention_mask):

        for step, encoder_layer in enumerate(self.encoder_list):
            if step == 0:
                embedding = self.layer_norm(feature + position_feature)
            else:
                embedding = self.layer_norm(embedding + position_feature)
            embedding = encoder_layer(embedding, pad_mask, self_attention_mask)
        return embedding


class EncoderLayer(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head, ff_size):
        super(EncoderLayer, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.dropout1 = t.nn.Dropout(dropout)
        self.layer_norm1 = t.nn.LayerNorm(input_size)
        self.transition = FeedForward(input_size, ff_size, dropout)
        self.dropout2 = t.nn.Dropout(dropout)
        self.layer_norm2 = t.nn.LayerNorm(input_size)

    def forward(self, embedding, input_mask, self_attention_mask=None):
        input_mask = input_mask.float().unsqueeze(-1)
        res1 = embedding
        net = self.multi_head_self_attention(embedding, embedding, embedding, self_attention_mask)
        net += res1
        net = self.dropout1(net)
        net = self.layer_norm1(net)
        net *= input_mask
        res2 = net
        net = self.transition(net)
        net += res2
        net = self.dropout2(net)
        net = self.layer_norm2(net)
        net *= input_mask
        return net


class Decoder(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head, num_layer, ff_size):
        super(Decoder, self).__init__()
        self.decoder_list = t.nn.ModuleList(
            [DecoderLayer(input_size, hidden_size, dropout, num_head, ff_size) for i in range(num_layer)]
        )
        self.layer_norm = t.nn.LayerNorm(input_size)

    def forward(self, feature, encoder_output, pad_mask, position_feature, self_attention_mask, dot_attention_mask):

        for step, decoder_layer in enumerate(self.decoder_list):
            if step == 0:
                embedding = self.layer_norm(feature + position_feature)
            else:
                embedding = self.layer_norm(embedding + position_feature)
            embedding = decoder_layer(embedding, encoder_output, pad_mask, self_attention_mask, dot_attention_mask)
        return embedding


class DecoderLayer(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head, ff_size):
        super(DecoderLayer, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.dropout1 = t.nn.Dropout(dropout)
        self.layer_norm1 = t.nn.LayerNorm(input_size)
        self.multi_head_dot_attention = MultiHeadAttention(input_size, hidden_size, dropout, num_head)
        self.dropout2 = t.nn.Dropout(dropout)
        self.layer_norm2 = t.nn.LayerNorm(input_size)
        self.transition = FeedForward(input_size, ff_size, dropout)
        self.dropout3 = t.nn.Dropout(dropout)
        self.layer_norm3 = t.nn.LayerNorm(input_size)

    def forward(self, embedding, encoder_output, input_mask, self_attention_mask=None, dot_attention_matrix=None):
        input_mask = input_mask.float().unsqueeze(-1)
        res1 = embedding
        net = self.multi_head_self_attention(embedding, embedding, embedding, self_attention_mask)
        net += res1
        net = self.dropout1(net)
        net = self.layer_norm1(net)
        net *= input_mask
        res2 = net
        net = self.multi_head_dot_attention(net, encoder_output, encoder_output, dot_attention_matrix)
        net += res2
        net = self.dropout2(net)
        net = self.layer_norm2(net)
        net *= input_mask
        res3 = net
        net = self.transition(net)
        net += res3
        net = self.dropout3(net)
        net = self.layer_norm3(net)
        net *= input_mask
        return net


class FeedForward(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = t.nn.Conv1d(input_size, hidden_size, 1)
        self.linear2 = t.nn.Conv1d(hidden_size, input_size, 1)
        self.relu = t.nn.ReLU(True)
        t.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()
        t.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, inputs):
        net = self.linear1(inputs.transpose(1, 2))
        net = self.relu(net)
        net = self.linear2(net)
        net = net.transpose(1, 2)
        return net


class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.output_dim = input_size
        self.key_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size, bias=False)
        self.query_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size, bias=False)
        self.value_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size, bias=False)
        self.scale = np.sqrt(self.hidden_size)
        self.linear = t.nn.Linear(self.num_head * self.hidden_size, input_size, bias=False)
        t.nn.init.xavier_normal_(self.key_projection.weight)
        t.nn.init.xavier_normal_(self.query_projection.weight)
        t.nn.init.xavier_normal_(self.value_projection.weight)
        t.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, query, key, value, attention_mask=None):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
        # key = value
        batch_size, query_lenth, query_dim = query.size()
        key_lenth = key.size(1)
        query_projection = self.query_projection(query).view(batch_size, query_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, QL, H

        key_projection = self.key_projection(key).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, H, KL

        value_projection = self.value_projection(value).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, KL, H

        attention_matrix = (query_projection @ key_projection) / self.scale
        # B, N, QL, KL

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask == 0, -float('inf'))

        attention_matrix = F.softmax(attention_matrix, -1)
        attention_matrix = attention_matrix.masked_fill(t.isnan(attention_matrix), 0)
        attention_matrix = self.dropout(attention_matrix)
        weighted = attention_matrix @ value_projection
        # B, N, QL, KL * B, N, KL, H -> B, N，QL, H
        output = weighted.permute(0, 2, 1, 3).contiguous().view(batch_size, query_lenth, self.num_head * self.hidden_size)
        output = self.linear(output)
        return output


if __name__ == '__main__':
    d_model = 5
    hidden_size = 512
    num_head = 8
    dropout = 0.1
    layer_num = 6
    ff_size = 512

    embedding = t.nn.Embedding(300, 5)


    #
    inputs = t.Tensor([[2, 5, 0]]).long()
    inputs2 = t.Tensor([[2,4,5,7,0]]).long()


    enc_feature = embedding(inputs)
    dec_feature = embedding(inputs2)
    print(inputs.shape)
    from Predictor.data_handler import Masker
    print(Masker.get_pad_mask(inputs, pad_idx=0))
    print(Masker.get_attn_key_pad_mask(inputs, inputs, 0))
    print(Masker.get_subsequent_mask(inputs))
    print(Masker.get_attn_pad_mask(Masker.get_pad_mask(inputs, pad_idx=0), expand_length=3))


    query = inputs
    key = inputs2
    pad_mask = Masker.get_pad_mask(query)
    self_attention_mask = Masker.get_attn_pad_mask(pad_mask, pad_mask.size(1))
    print(pad_mask.shape)
    print(self_attention_mask.shape)

    encoder = Encoder(
        num_head=num_head, dropout=dropout, layer_num=layer_num, hidden_size=hidden_size, input_size=d_model,
        ff_size=ff_size)

    print(encoder(enc_feature, pad_mask, self_attention_mask))
    o_feature = encoder(enc_feature, pad_mask, self_attention_mask)
    decoder = Decoder(num_head=num_head, dropout=dropout, layer_num=layer_num, hidden_size=hidden_size, input_size=d_model,
        ff_size=ff_size)

    dec_pad_mask = Masker.get_pad_mask(inputs2)
    dot_attention_mask = Masker.get_attn_key_pad_mask(inputs, inputs2)
    dec_self_attention_mask = Masker.get_attn_pad_mask(dec_pad_mask, dec_pad_mask.size(1))
    # feature, encoder_output, pad_mask, self_attention_mask, dot_attention_mask
    print(decoder(dec_feature, o_feature, dec_pad_mask, dec_self_attention_mask, dot_attention_mask))
    print(inputs)
    print(inputs2)
    print(decoder(dec_feature, o_feature, dec_pad_mask, dec_self_attention_mask, dot_attention_mask).shape)
