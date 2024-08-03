
### 1. Data Preparation

#### Loading and Cleaning the Data
- We load English and Kannada sentences from text files.
- Only the first 10,000 sentences are considered to limit the data size.
- Sentences are cleaned by removing any trailing newline characters or extra spaces.

#### Vocabulary Creation
- We define vocabularies for both English and Kannada, which include all possible characters that can appear in the sentences.
- Special tokens are added to the vocabularies: `<START>`, `<PADDING>`, and `<END>`.
- Mappings from characters to indices and vice versa are created for both languages, facilitating the conversion of sentences into numerical format.

#### Sentence Validation
- Sentences are checked for invalid characters and excessive length.
- Valid sentences are those that:
  - Contain only characters from the respective vocabulary.
  - Have a length less than the maximum sequence length (200 characters).

### 2. Dataset and DataLoader

#### Custom Dataset Class
- We define a custom dataset class `TextDatasets` to manage pairs of English and Kannada sentences.
- This class implements the `__len__` and `__getitem__` methods to fetch the number of sentences and specific sentence pairs, respectively.

#### DataLoader
- A DataLoader is used to batch the data, which helps in efficient training.
- Batches of sentences are fetched for each iteration during training.

### 3. Tokenization

#### Tokenizing Sentences
- Sentences are converted into sequences of numerical indices using the vocabulary mappings.
- Special tokens are added:
  - `<START>` at the beginning of Kannada sentences.
  - `<END>` at the end of both English and Kannada sentences.
  - `<PADDING>` to pad sentences to the maximum sequence length.

### 4. Masking

#### Creating Masks
- Masks are created to:
  - Prevent the model from considering padding tokens.
  - Handle the attention mechanism effectively.
- Three types of masks are used:
  - **Look-ahead mask:** Prevents the decoder from attending to future tokens during training.
  - **Encoder padding mask:** Ensures the encoder ignores padding tokens.
  - **Decoder padding mask:** Ensures the decoder ignores padding tokens during self-attention and cross-attention.

### 5. Model Architecture

#### Sentence Embedding
- We define a `SentenceEmbedding` class to create embeddings for sentences.
- The embedding layer converts token indices to dense vectors.
- Positional encoding is added to embeddings to give the model information about the position of each token in the sentence.

#### Positional Encoding
- Positional encoding helps the model understand the order of tokens in a sentence.
- This encoding is added to the token embeddings before feeding them into the model.


### SentenceEmbedding Class



```python
import torch
import torch.nn as nn

# Define a simple positional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self):
        return self.encoding

# SentenceEmbedding class definition
class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)  # The size of the vocabulary
        self.max_sequence_length = max_sequence_length  # Maximum sequence length
        self.embedding = nn.Embedding(self.vocab_size, d_model)  # Embedding layer
        self.language_to_index = language_to_index  # Mapping from tokens to indices
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)  # Positional encoding
        self.dropout = nn.Dropout(p=0.1)  # Dropout layer to prevent overfitting
        self.START_TOKEN = START_TOKEN  # Start token
        self.END_TOKEN = END_TOKEN  # End token
        self.PADDING_TOKEN = PADDING_TOKEN  # Padding token

    def batch_tokenize(self, batch, start_token=True, end_token=True):
        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indices = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indices)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, end_token=True):  # sentence
        x = self.batch_tokenize(x, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage
max_sequence_length = 20
d_model = 16

# Example vocabularies
english_vocabulary = [START_TOKEN, 'a', 'b', 'c', 'd', PADDING_TOKEN, END_TOKEN]
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

# Create an instance of SentenceEmbedding
sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
sentence_embedding.to(get_device())

# Example sentences
sentences = ["abc", "abcd"]

# Get the embeddings for the example sentences
embeddings = sentence_embedding(sentences)
print("Embeddings shape:", embeddings.shape)
print("Embeddings:", embeddings)
```

### Line-by-Line Explanation

1. **Importing Libraries**:
   ```python
   import torch
   import torch.nn as nn
   ```

2. **PositionalEncoding Class**:
   - This class adds positional information to the embeddings.
   - It creates a matrix where each position has a unique encoding.
   ```python
   class PositionalEncoding(nn.Module):
       def __init__(self, d_model, max_len=200):
           super(PositionalEncoding, self).__init__()
           self.encoding = torch.zeros(max_len, d_model)
           position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
           div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
           self.encoding[:, 0::2] = torch.sin(position * div_term)
           self.encoding[:, 1::2] = torch.cos(position * div_term)
           self.encoding = self.encoding.unsqueeze(0)

       def forward(self):
           return self.encoding
   ```

3. **SentenceEmbedding Class**:
   - This class converts sentences into embeddings that the model can process.
   - **Initialization**:
     - `vocab_size`: Length of the vocabulary.
     - `max_sequence_length`: Maximum length of the sentences.
     - `embedding`: Embedding layer converting tokens to vectors.
     - `position_encoder`: Positional encoding for order information.
     - `dropout`: Dropout layer to prevent overfitting.
     - `START_TOKEN`, `END_TOKEN`, `PADDING_TOKEN`: Special tokens for marking sentence boundaries and padding.
   ```python
   class SentenceEmbedding(nn.Module):
       "For a given sentence, create an embedding"
       def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
           super().__init__()
           self.vocab_size = len(language_to_index)
           self.max_sequence_length = max_sequence_length
           self.embedding = nn.Embedding(self.vocab_size, d_model)
           self.language_to_index = language_to_index
           self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
           self.dropout = nn.Dropout(p=0.1)
           self.START_TOKEN = START_TOKEN
           self.END_TOKEN = END_TOKEN
           self.PADDING_TOKEN = PADDING_TOKEN
   ```

4. **batch_tokenize Method**:
   - Converts a batch of sentences into sequences of indices.
   - Adds special tokens and pads the sentences to the maximum sequence length.
   ```python
   def batch_tokenize(self, batch, start_token=True, end_token=True):
       def tokenize(sentence, start_token=True, end_token=True):
           sentence_word_indices = [self.language_to_index[token] for token in list(sentence)]
           if start_token:
               sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
           if end_token:
               sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
           for _ in range(len(sentence_word_indices), self.max_sequence_length):
               sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
           return torch.tensor(sentence_word_indices)

       tokenized = []
       for sentence_num in range(len(batch)):
           tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
       tokenized = torch.stack(tokenized)
       return tokenized.to(get_device())
   ```

5. **forward Method**:
   - Converts sentences to embeddings and adds positional encoding.
   - Applies dropout to the combined embeddings and positional encodings.
   ```python
   def forward(self, x, end_token=True):  # sentence
       x = self.batch_tokenize(x, end_token)
       x = self.embedding(x)
       pos = self.position_encoder().to(get_device())
       x = self.dropout(x + pos)
       return x
   ```

6. **get_device Function**:
   - Determines if a GPU is available and returns the appropriate device.
   ```python
   def get_device():
       return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

7. **Example Usage**:
   - Defines the maximum sequence length and embedding dimension.
   - Creates an instance of the `SentenceEmbedding` class.
   - Converts example sentences into embeddings and prints the results.
   ```python
   max_sequence_length = 20
   d_model = 16

   english_vocabulary = [START_TOKEN, 'a', 'b', 'c', 'd', PADDING_TOKEN, END_TOKEN]
   english_to_index = {v: k for k, v in enumerate(english_vocabulary)}

   sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
   sentence_embedding.to(get_device())

   sentences = ["abc", "abcd"]
   embeddings = sentence_embedding(sentences)
   print("Embeddings shape:", embeddings.shape)
   print("Embeddings:", embeddings)
   ```
