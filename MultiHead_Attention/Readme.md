

### Example Sentence Processing with Multi-Head Attention

#### Sentence: "She saw the man with the telescope"

### Step 1: Tokenization
The sentence is split into individual tokens (words). Each token is assigned an index based on a predefined vocabulary.

- **Tokens**: ["She", "saw", "the", "man", "with", "the", "telescope"]
- **Token IDs** (Example): [101, 102, 103, 104, 105, 103, 106]

### Step 2: Embedding
Each token ID is mapped to a corresponding embedding vector, which captures semantic meaning. Suppose the model's embedding dimension (\(d_{\text{model}}\)) is 6 for simplicity.

- **Embedding Vectors** (Example):
  - "She": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  - "saw": [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]
  - "the": [0.5, 0.4, 0.3, 0.2, 0.1, 0.6]
  - "man": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  - "with": [0.3, 0.2, 0.1, 0.6, 0.5, 0.4]
  - "the": [0.5, 0.4, 0.3, 0.2, 0.1, 0.6]
  - "telescope": [0.4, 0.3, 0.2, 0.1, 0.6, 0.5]

### Step 3: Positional Encoding
Since transformers do not inherently understand the order of tokens, positional encodings are added to the embeddings to provide this information. For simplicity, we'll use sine and cosine functions for this encoding.

- **Positional Encoding** (Example for each position):
  - Position 0: [0.0, 0.8415, 0.9093, 0.1411, 0.1411, 0.9093]
  - Position 1: [0.8415, 0.9093, 0.1411, -0.7568, -0.7568, 0.1411]
  - And so on...

- **Final Input Representation**:
  - Add the positional encoding to the embeddings.

### Step 4: Applying Multi-Head Attention

#### Configuration
- **Number of Heads (h)**: 2 (for simplicity)
- **Dimensionality per Head** (\(d_k\), \(d_v\)): \( \frac{d_{\text{model}}}{h} = 3 \)

#### Linear Projections
For each head, compute the Query (Q), Key (K), and Value (V) matrices using learned linear transformations.

- **Head 1**:
  - \( Q_1 = X W_1^Q \)
  - \( K_1 = X W_1^K \)
  - \( V_1 = X W_1^V \)

- **Head 2**:
  - \( Q_2 = X W_2^Q \)
  - \( K_2 = X W_2^K \)
  - \( V_2 = X W_2^V \)

#### Attention Calculation for Each Head
- **Head 1**:
  - Compute attention scores: \( \text{Scores}_1 = \frac{Q_1 K_1^T}{\sqrt{d_k}} \)
  - Apply softmax to obtain attention weights: \( \text{Weights}_1 = \text{softmax}(\text{Scores}_1) \)
  - Compute output: \( \text{Output}_1 = \text{Weights}_1 V_1 \)

- **Head 2**:
  - Compute attention scores: \( \text{Scores}_2 = \frac{Q_2 K_2^T}{\sqrt{d_k}} \)
  - Apply softmax to obtain attention weights: \( \text{Weights}_2 = \text{softmax}(\text{Scores}_2) \)
  - Compute output: \( \text{Output}_2 = \text{Weights}_2 V_2 \)

#### Concatenation and Final Linear Transformation
- **Concatenate Outputs**: \( \text{Concat}(\text{Output}_1, \text{Output}_2) \)
- **Final Linear Transformation**: Apply a final linear transformation to combine the outputs from all heads.

### Example Output
Suppose the final transformed output for each token is:

- "She": [0.5, 0.4, 0.7, 0.6, 0.8, 0.9]
- "saw": [0.6, 0.5, 0.7, 0.7, 0.8, 1.0]
- "the": [0.7, 0.6, 0.5, 0.8, 0.7, 0.6]
- "man": [0.8, 0.7, 0.6, 0.9, 0.8, 0.7]
- "with": [0.9, 0.8, 0.7, 1.0, 0.9, 0.8]
- "the": [0.7, 0.6, 0.5, 0.8, 0.7, 0.6]
- "telescope": [0.8, 0.7, 0.6, 0.9, 0.8, 0.7]

These vectors represent the final encoded state of each token in the sentence after applying multi-head attention and are used for subsequent layers in the transformer model, such as the feed-forward layers, or for making predictions.

### Conclusion
The multi-head attention mechanism allows the model to focus on different aspects of the input sentence simultaneously, such as syntax, semantics, and context. This leads to a richer and more nuanced understanding of the data, which is crucial for tasks like translation, summarization, and more.
