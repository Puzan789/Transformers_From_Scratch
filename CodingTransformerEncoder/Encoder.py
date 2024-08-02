import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scaled += mask

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.size()

        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        values = self.linear_layer(values)

        return values


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, unbiased=False, keepdim=True)
        std = (variance + self.eps).sqrt()
        normalized = (inputs - mean) / std
        output = self.gamma * normalized + self.beta
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])

    def forward(self, x, mask=None):
        residual_x = x
        x = self.attention(x, mask)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        x = self.norm2(x + residual_x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.layers(x)
        return x
    
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5

    encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    x = torch.randn((batch_size, max_sequence_length, d_model))
    out = encoder(x)

    print("Output shape:", out.shape)
