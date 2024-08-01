

### Add (Residual Connection)

In the Add & Norm mechanism, the **Add** part represents the use of residual connections, which help in training deep networks by allowing gradients to flow more effectively. Specifically, the output of a sub-layer (such as Multi-Head Attention or Feed Forward Network) is added to the original input of that sub-layer.

Mathematically, this can be expressed as:

$$ \text{Output} = \text{Sub-layer Output} + \text{Input} $$

This formula indicates that the input to a sub-layer is added to its output, forming a residual connection.

### Norm (Layer Normalization)

The **Norm** part refers to layer normalization, which normalizes the summed output from the residual connection. This normalization ensures that the inputs to each layer maintain a stable distribution, which is crucial for efficient training.

The layer normalization formula is:

$$ \text{Norm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta $$

Where:
- \( x \) is the input vector to be normalized,
- \( \mu \) is the mean of the input values,
- \( \sigma \) is the standard deviation of the input values,
- \( \gamma \) and \( \beta \) are learnable parameters used for scaling and shifting the normalized output.

## Why Use Add & Norm?

The Add & Norm operation enhances the model's ability to learn by stabilizing the training process. Residual connections help mitigate the vanishing gradient problem, and layer normalization helps accelerate convergence by maintaining a consistent input distribution to each layer.


