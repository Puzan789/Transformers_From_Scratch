# Encoder Architecture

The encoder architecture is composed of multiple layers that include multi-head attention mechanisms, layer normalization, and position-wise feed-forward networks. This explanation covers the main components and their mathematical operations.

## Components

### 1. Scaled Dot-Product Attention

The scaled dot-product attention is computed as follows:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
where $Q$, $K$, and $V$ are the queries, keys, and values respectively, and $d_k$ is the dimension of the keys.

### 2. Multi-Head Attention

The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces. The process is as follows:

1. Linear projections of $Q$, $K$, and $V$:
$$
Q, K, V = W_QX, W_KX, W_VX
$$
2. Apply scaled dot-product attention to each projected version of $Q$, $K$, and $V$:
$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i)
$$
3. Concatenate the heads and apply a final linear layer:
$$
\text{MultiHead}(Q, K, V) = W_O(\text{head}_1 \| \text{head}_2 \| \dots \| \text{head}_h)
$$

### 3. Layer Normalization

Layer normalization is applied to stabilize and accelerate the training. The normalized output is given by:
$$
\text{LN}(x) = \gamma \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta
$$
where $\mu$ and $\sigma^2$ are the mean and variance of the input $x$, and $\gamma$ and $\beta$ are learnable parameters.

### 4. Position-Wise Feed-Forward Network

The position-wise feed-forward network consists of two linear transformations with a ReLU activation in between:
$$
\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
$$

### 5. Encoder Layer

Each encoder layer contains multi-head attention and feed-forward networks, each followed by layer normalization and residual connections:
$$
\text{EncoderLayer}(x) = \text{LN}(x + \text{MultiHead}(x, x, x))
$$
$$
\text{EncoderLayer}(x) = \text{LN}(x + \text{FFN}(x))
$$

### 6. Complete Encoder

The complete encoder is a stack of $N$ encoder layers:
$$
\text{Encoder}(x) = \text{EncoderLayer}_N(\dots \text{EncoderLayer}_2(\text{EncoderLayer}_1(x)) \dots)
$$

## Code Explanation

### Initialization

The encoder is initialized with the following parameters:
- `d_model = 512`: Dimensionality of the model.
- `num_heads = 8`: Number of attention heads.
- `drop_prob = 0.1`: Dropout probability.
- `batch_size = 30`: Number of samples in a batch.
- `max_sequence_length = 200`: Maximum length of the input sequence.
- `ffn_hidden = 2048`: Dimensionality of the feed-forward network's hidden layer.
- `num_layers = 5`: Number of encoder layers.

### Encoder Class

The `Encoder` class is defined as:
```python
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.layers(x)
        return x
