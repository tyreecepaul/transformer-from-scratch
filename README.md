# Transformer From Scratch  
**By Tyreece Paul**

This project is a from-scratch implementation of the Transformer architecture using **PyTorch**. It is based on the seminal paper *["Attention is All You Need"](https://arxiv.org/abs/1706.03762)* by Vaswani et al.

The goal of this project is to replicate the full architecture described in the paper without relying on high-level libraries or prebuilt components.

### Resources Used

The implementation was guided by the following resources by **Umar Jamil**:

- [Transformer From Scratch — Code Walkthrough (YouTube)](https://www.youtube.com/watch?v=ISNdQcPhsts&t=8384s)
- [Transformer Theory — Mathematical Foundations (YouTube)](https://www.youtube.com/watch?v=bCz4OMemCcA&t=2704s)
- [GitHub Repository: hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)

---

This README provides a detailed explanation of:
- The mathematical foundations of the Transformer model
- How each component is implemented in code

---

## Before Transformers
- RNNs

## Attention is All You Need

The Transformer model is composed of two main components: the encoder and the decoder.

### Input Embedding
A sentence $f$ is first tokenized into a sequence of tokens $(t_1, t_2, \dots, t_n)$
where \(n\) may be greater than the number of words due to subword tokenization (e.g., using Byte-Pair Encoding or WordPiece). 
Each token $t_i$ is mapped to an integer index

$$
x_i \in \{ z \in \mathbb{Z} \mid z \geq 0 \}
$$

known as its input ID, which corresponds to its position in a fixed vocabulary $\mathcal{V}$

These input IDs form the sequence $\mathbf{x} = (x_1, x_2, \dots, x_n)$

This sequence is then embedded via a learnable embedding matrix

$$
E \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}
$$

resulting in an input embedding matrix

$$
\mathbf{X} \in \mathbb{R}^{n \times d_{\text{model}}}
$$

where $\mathbf{X}_i = E[x_i]$

While the same token maps to the same row in $E$ the model captures contextual meaning through attention mechanisms and positional encodings, rather than using fixed static embeddings. For models with a maximum sequence length (e.g. 512), $\mathbf{X}$ is typically padded or truncated to have shape $(512, d_{\text{model}})$ 

```python
self.embedding = nn.Embedding(vocab_size, d_model)   # vocab_size = 512
```

### Positional Encoding

To enable the model to capture the order of tokens in a sequence, a positional encoding function 

$$
PE: \mathbb{N} \times \mathbb{N} \to \mathbb{R}
$$ 

is defined, where $PE(pos, i)$ encodes the position $pos$ (from 0 to the maximum sequence length) and the embedding dimension index $i$ (from 0 to $d_{\text{model}} - 1$).

The positional encoding vectors 

$$
\mathbf{P} \in \mathbb{R}^{n \times d_{\text{model}}}
$$ 

are fixed, deterministic vectors added to the input embeddings to provide information about token order. For a token at position $pos$, the positional encoding components are given by the two equations:

$$
PE(pos, 2i) = \sin \left( \frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}} \right)
$$

$$
PE(pos, 2i+1) = \cos \left( \frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}} \right)
$$

for $i = 0, 1, \dots, \frac{d_{\text{model}}}{2} - 1$.

The input to the encoder at position $pos$ is then the sum of the token embedding `X_pos ∈ ℝ^{d_model}` and the positional encoding

`P_pos ∈ ℝ^{d_model}`

`Z_pos = X_pos + P_pos`

where $\mathbf{Z} \in \mathbb{R}^{n \times d_{\text{model}}}$ is the final input embedding matrix with positional information, typically with $n = 512$ for fixed maximum sequence length.

#### Why Use Trigonometric Functions?

The sine and cosine functions naturally encode periodic, continuous patterns that enable the model to easily learn and generalize relative positions of tokens. Because these functions vary smoothly and have different frequencies across embedding dimensions, the model can infer both absolute and relative positions by attending to these patterns. This design ensures that for any fixed offset $k$, the positional encoding satisfies a linear relationship:

$$
PE(pos + k) = f(PE(pos), k)
$$

making relative positioning easier to capture. Empirically, plotting the positional encoding components shows regular oscillatory patterns that the model can leverage to understand token order.

### Mulit-Head Attention
#### Self Attention


