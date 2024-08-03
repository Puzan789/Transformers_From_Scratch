
# Decoder Architecture

The decoder is responsible for generating the target sequence from the encoded input sequence in a sequence-to-sequence model. It consists of multiple layers, each with self-attention, encoder-decoder attention, and feed-forward networks, incorporating residual connections and layer normalization.

## Components

1. **Multi-Head Attention**: This mechanism allows the decoder to focus on different parts of the input sequence. It is applied twice in each decoder layer:
   - **Masked Self-Attention**: Prevents attending to future positions.
   - **Encoder-Decoder Attention (Multi-Head Cross Attention)**: Allows attending to the relevant parts of the input sequence.

2. **Layer Normalization**: Normalizes the input to the next sub-layer to improve convergence and stability.

3. **Position-Wise Feed-Forward Network**: A fully connected feed-forward network applied independently to each position.

4. **Residual Connections**: Skip connections that add the input of each sub-layer to its output.

5. **Dropout**: Regularization technique to prevent overfitting.

## Mathematical Formulations

### Multi-Head Attention

Multi-Head Attention consists of multiple attention heads, each computing scaled dot-product attention.

#### Scaled Dot-Product Attention

For a query \(Q\), key \(K\), and value \(V\), the scaled dot-product attention is computed as:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

where \(d_k\) is the dimension of the key vectors.

#### Multi-Head Attention

With multiple heads, the attention mechanism is:

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) W^O
```

where each head is:

```math
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
```

\(W_i^Q, W_i^K, W_i^V\) are projection matrices for the \(i\)-th head, and \(W^O\) is the output projection matrix.

### Multi-Head Cross Attention

The Multi-Head Cross Attention mechanism is similar to the standard multi-head attention but involves two different sequences: the target sequence and the encoded input sequence. This mechanism allows the decoder to attend to relevant parts of the encoded input sequence when generating the target sequence.

Given the encoded input sequence \(X\) and the target sequence \(Y\), the cross attention mechanism is defined as:

```math
\text{CrossAttention}(Y, X) = \text{MultiHead}(Y W^Q, X W^K, X W^V)
```

where \(W^Q, W^K, W^V\) are the projection matrices for the queries, keys, and values, respectively. The queries \(Q\) come from the target sequence \(Y\), while the keys \(K\) and values \(V\) come from the encoded input sequence \(X\).

### Layer Normalization

Layer normalization is applied to the input to the sub-layers:

```math
\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

where \(\mu\) and \(\sigma^2\) are the mean and variance of the input \(x\), and \(\gamma\) and \(\beta\) are learned parameters.

### Position-Wise Feed-Forward Network

A two-layer feed-forward network applied to each position separately:

```math
\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
```

where \(W_1, W_2, b_1, b_2\) are learned parameters.

### Residual Connections

Residual connections are used to add the input of each sub-layer to its output:

```math
\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
```

### Decoder Layer

Each decoder layer consists of the following components applied in sequence:

1. **Masked Self-Attention**:
    ```math
    \text{SelfAttn}(Q, K, V, \text{mask})
    ```
   
2. **Encoder-Decoder Attention (Multi-Head Cross Attention)**:
    ```math
    \text{CrossAttention}(Q, K, V)
    ```

3. **Feed-Forward Network**:
    ```math
    \text{FFN}(x)
    ```

### Overall Decoder

The overall decoder stacks multiple decoder layers. Given the input sequence \(X\) and the target sequence \(Y\), with a mask for the target sequence, the decoder generates the output sequence.

## Usage Example

To create and use the decoder:

```python
import torch

d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

x = torch.randn((batch_size, max_sequence_length, d_model)) # Encoded input sequence
y = torch.randn((batch_size, max_sequence_length, d_model)) # Target sequence
mask = torch.full([max_sequence_length, max_sequence_length], float('-inf'))
mask = torch.triu(mask, diagonal=1)

decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
output = decoder(x, y, mask)
```
