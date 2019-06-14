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



class TransformerOffical(BaseModel):
    def __init__(self, config, vocab):
        super(TransformerOffical, self).__init__()
        self.config = config
        self.vocab = vocab

        self.input_linear = t.nn.Linear(config.n_mels, config.d_model)
        t.nn.init.xavier_normal_(self.input_linear.weight)

        self.word_embedding = t.nn.Embedding(vocab.vocab_size, config.d_model, padding_idx=0)
        self.position_embedding = PositionEmbedding(config.d_model, 2000)
        self.encoder = Encoder(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            ff_size=config.ff_size,
            num_head=config.num_head,
            dropout=config.dropout,
            layer_num=config.layer_num
        )
        self.decoder = Decoder(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            ff_size=config.ff_size,
            num_head=config.num_head,
            dropout=config.dropout,
            layer_num=config.layer_num
        )
        self.output = t.nn.Linear(config.d_model, vocab.vocab_size, bias=False)
        self.output.weight = self.word_embedding.weight
        self.project_scale = config.d_model ** -0.5

    def forward(self, input):
        wave, wave_len, text_for_input, text_len = input.wave, input.wave_len, input.tgt_for_input, input.tgt_len

        # build masks
        wave_pad_mask = Masker.get_pad_mask(wave[:, :, 0], wave_len)
        wave_self_attention_mask = Masker.get_dot_attention_mask(wave_pad_mask, wave_pad_mask)
        wave_self_attention_mask = t.triu(t.tril(wave_self_attention_mask, 50), -50)
        text_pad_mask = Masker.get_pad_mask(text_for_input, text_len)
        text_self_attention_mask = Masker.get_dot_attention_mask(text_pad_mask, text_pad_mask)
        text_subsquence_mask = Masker.get_subsequent_mask(text_for_input)
        text_self_attention_mask = text_self_attention_mask * text_subsquence_mask
        dot_attention_mask = Masker.get_dot_attention_mask(text_pad_mask, wave_pad_mask)
        wave_pad_mask.unsqueeze_(-1)
        text_pad_mask.unsqueeze_(-1)
        # build inputs
        wave = self.input_linear(wave)
        wave += self.position_embedding(wave_len)
        text = self.word_embedding(text_for_input)
        text += self.position_embedding(text_len)

        encoded = self.encoder(wave, wave_pad_mask, wave_self_attention_mask)
        decoded = self.decoder(text, encoded, text_pad_mask, text_self_attention_mask, dot_attention_mask)

        output = self.output(decoded) * self.project_scale
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
            d_model = 256
            hidden_size = 64
            ff_size = 256
            num_head = 4
            dropout = 0.1
            layer_num = 6

        return ModelConfig
import torch.nn as nn

from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from Predictor.Models.utils import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns:
            enc_output: N x T x H
        """
        enc_slf_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Decoder(t.nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_head, dropout, layer_num):
        super(Decoder, self).__init__()
        self.layers = t.nn.ModuleList(
            DecoderLayer(
                input_size=input_size, hidden_size=hidden_size, ff_size=ff_size, num_head=num_head, dropout=dropout
            ) for i in range(layer_num)
        )

    def forward(self, embedding, encoder_output, pad_mask, self_attention_mask, dot_attention_mask):
        for layer in self.layers:
            embedding = layer(embedding, encoder_output, pad_mask, self_attention_mask, dot_attention_mask)
        return embedding

class DecoderLayer(t.nn.Module):
    def __init__(self, input_size, hidden_size, ff_size, num_head, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            input_size=input_size, hidden_size=hidden_size, num_head=num_head, dropout=dropout
        )
        self.attention_dropout = t.nn.Dropout(dropout)
        self.attention_layer_norm = t.nn.LayerNorm(input_size)

        self.dot_attention = MultiHeadAttention(
            input_size=input_size, hidden_size=hidden_size, num_head=num_head, dropout=dropout
        )
        self.dot_attention_dropout = t.nn.Dropout(dropout)
        self.dot_attention_layer_norm = t.nn.LayerNorm(input_size)

        self.feed_forward = FeedForward(input_size=input_size, hidden_size=ff_size, dropout=dropout)
        self.feed_forward_dropout = t.nn.Dropout(dropout)
        self.feed_forward_norm = t.nn.LayerNorm(input_size)

    def forward(self, embedding, encoder_output, pad_mask, self_attention_mask, dot_attention_mask):
        residual = embedding
        net = self.self_attention(embedding, embedding, embedding, self_attention_mask)
        net = self.attention_dropout(net)
        net = self.attention_layer_norm(net + residual)
        net *= pad_mask

        residual = net
        net = self.dot_attention(net, encoder_output, encoder_output, dot_attention_mask)
        net = self.dot_attention_dropout(net)
        net = self.dot_attention_layer_norm(net + residual)
        net *= pad_mask

        residual = net
        net = self.feed_forward(net)
        net = self.feed_forward_dropout(net)
        net = self.feed_forward_norm(net + residual)
        net *= pad_mask
        return net

class PositionEmbedding(t.nn.Module):
    def __init__(self, input_size, max_length):
        super(PositionEmbedding, self).__init__()
        self.position_embedding = t.nn.Embedding(max_length, input_size).from_pretrained(
            self.get_sinusoid_encoding_table(max_length, input_size, padding_idx=0),
            freeze=True
        )
        self.register_buffer('position', t.arange(1, max_length))

    def build_position_id(self, lengths: t.LongTensor):
        batch_size = lengths.size(0)
        max_length = lengths.max()
        device = lengths.device
        position_id = t.zeros(batch_size, max_length, device=device, dtype=t.long)
        for index, value in enumerate(lengths):
            position_id[index][:value] = self.position[:value]
        return position_id

    def forward(self, lengths):
        position_id = self.build_position_id(lengths)
        position_feature = self.position_embedding(position_id)
        return position_feature

    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        ''' Sinusoid position encoding table '''

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return t.FloatTensor(sinusoid_table)


class FeedForward(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = t.nn.Conv1d(input_size, hidden_size, 1)
        self.linear2 = t.nn.Conv1d(hidden_size, input_size, 1)
        self.relu = t.nn.ReLU(True)
        self.dropout = t.nn.Dropout(dropout)
        t.nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()
        t.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, inputs):
        net = self.linear1(inputs.transpose(-1, -2))
        net = self.relu(net)
        net = self.dropout(net)
        net = self.linear2(net)
        net = net.transpose(-1, -2)
        return net



class MultiHeadAttention(t.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(MultiHeadAttention, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.output_dim = input_size
        self.key_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        self.query_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        self.value_projection = t.nn.Linear(input_size, self.num_head * self.hidden_size)
        t.nn.init.normal_(self.key_projection.weight, mean=0, std=np.sqrt(2.0 / (input_size + hidden_size)))
        t.nn.init.normal_(self.query_projection.weight, mean=0, std=np.sqrt(2.0 / (input_size + hidden_size)))
        t.nn.init.normal_(self.value_projection.weight, mean=0, std=np.sqrt(2.0 / (input_size + hidden_size)))
        self.scale = np.sqrt(self.hidden_size)
        self.linear = t.nn.Linear(self.num_head * self.hidden_size, input_size, bias=False)
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

