# Word2Vec from First Principles

A complete, mathematical derivation and implementation of Word2Vec with **zero black-box abstractions**.

> **"If you can't implement it from scratch, you don't understand it."**

This project treats Word2Vec as what it actually is: a **parametric conditional language model optimized via maximum likelihood estimation**, not neural magic.

---

## 🎯 Project Goals

- **No high-level libraries** (no gensim, no TensorFlow, no PyTorch)
- **Every equation maps to code** - direct, traceable, explicit
- **Complete mathematical derivation** - from probability theory to gradient descent
- **Manual gradient computation** - no autograd, no backprop frameworks
- **Statistical rigor** - maximum likelihood, not hand-waving

## What This Is

Word2Vec is:
- A **statistical model** of word co-occurrence patterns
- A **low-rank approximation** of a shifted PMI (Pointwise Mutual Information) matrix  
- A **maximum likelihood estimator** for conditional probabilities: P(context | center)
- A **logistic regression** model with negative sampling

Word2Vec is **not**:
- A deep neural network
- "AI learning meaning."
- Magic embeddings

---

## Mathematical Foundation

### The Core Objective

Given corpus tokens w₁, w₂, ..., wₜ, we maximize:

```
L(θ) = ∏ P(wₜ₊ⱼ | wₜ; θ)
```

Or equivalently, minimize negative log-likelihood:

```
J(θ) = -1/|T| ∑ₜ ∑ⱼ log P(wₜ₊ⱼ | wₜ; θ)
```

### The Model

For each word w, we learn two d-dimensional vectors:
- **vw**: input (center) embedding
- **uw**: output (context) embedding

The conditional probability using softmax:

```
P(wₒ | wc) = exp(uₒᵀvc) / ∑w exp(uwᵀvc)
```

### The Problem

Softmax normalization requires O(|V|) operations per training pair.

**Solution**: Negative Sampling

### Negative Sampling Objective

Replace multi-class softmax with binary logistic classification:

```
J_NS = -log σ(uₒᵀvc) - ∑ᵢ₌₁ᵏ 𝔼[log σ(-uᵢᵀvc)]
```

where:
- σ(x) = 1/(1 + exp(-x)) is the sigmoid function
- k negative samples drawn from noise distribution Pₙ(w) ∝ count(w)^0.75

### Gradients (Computed Manually)

```
∂J/∂uₒ = [σ(uₒᵀvc) - 1] vc
∂J/∂uᵢ = σ(uᵢᵀvc) vc
∂J/∂vc = [σ(uₒᵀvc) - 1] uₒ + ∑ᵢ σ(uᵢᵀvc) uᵢ
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Gaur-ang/Word2Vec.git
cd Word2Vec
pip install numpy  # That's it. No other dependencies.
```

### Basic Usage

```python
from word2vec import Word2Vec

# Your corpus
corpus = """
The king sat on the throne.
The queen stood beside the king.
Paris is the capital of France.
"""

# Initialize model
model = Word2Vec(
    embedding_dim=100,    # dimension d
    window_size=5,        # context window ±5
    neg_samples=5,        # k negative samples
    learning_rate=0.025,  # initial α
    min_count=5          # minimum word frequency
)

# Train
model.train(corpus, epochs=10)

# Get embeddings
king_vec = model.get_embedding('king')

# Find similar words
similar = model.most_similar('king', topn=5)
# [('queen', 0.876), ('prince', 0.834), ...]

# Solve analogies: king - man + woman ≈ ?
result = model.analogy('king', 'man', 'woman')
# [('queen', 0.891), ...]
```

---

## Implementation Details

### Architecture

```
Corpus → Tokenization → Vocabulary Building
    ↓
Training Pairs: (center, context)
    ↓
Parameters: V ∈ ℝ^(|V| × d), U ∈ ℝ^(|V| × d)
    ↓
For each pair (wc, wₒ):
    1. Sample k negative words from Pₙ
    2. Compute loss: J_NS
    3. Compute gradients (manually)
    4. SGD update: θ ← θ - α∇J
```

### Parameter Count

Total trainable parameters: **2 × |V| × d**

Example: 10,000 vocabulary × 100 dimensions = **2,000,000 parameters**

Per training step updates:
- 1 input embedding (vc)
- 1 positive output embedding (uₒ)  
- k negative output embeddings (uᵢ)

**Only (k+2) vectors updated per step** → sparse, efficient updates

### Noise Distribution

```python
Pₙ(w) = count(w)^0.75 / ∑w' count(w')^0.75
```

Why 0.75?
- Unigram (α=1.0): over-samples common words like "the"
- Uniform (α=0.0): over-samples rare words
- Smoothed (α=0.75): empirically optimal balance

---

## Why This Works

### Connection to PMI

Word2Vec implicitly factorizes a shifted PMI matrix:

```
uₒᵀvc ≈ PMI(wₒ, wc) - log k
```

where PMI (Pointwise Mutual Information):

```
PMI(wₒ, wc) = log[P(wₒ, wc) / (P(wₒ)P(wc))]
```

### Why Cosine Similarity Works

Embeddings encode **distributional statistics**:
- Words with similar contexts → similar vectors
- Direction encodes semantic category
- Magnitude encodes frequency (normalized away by cosine)

### Why Analogies Emerge

Vector arithmetic works when PMI differences are approximately parallel:

```
PMI(queen, c) - PMI(king, c) ≈ PMI(woman, c) - PMI(man, c)
```

This translates to:

```
vqueen - vking ≈ vwoman - vman
```

**This is approximate, not exact** - depends on dimensionality and training.

---

## Educational Components

### 1. Complete Mathematical Derivation

Every step from probability theory to working code:
- Corpus definition and notation
- Softmax formulation
- Gradient derivation (chain rule, step-by-step)
- Computational complexity analysis
- Negative sampling derivation
- PMI connection proof

### 2. Explicit Implementation

No hidden abstractions:
```python
def _update_embeddings(self, center_idx, context_idx):
    """ Manual SGD update with explicit gradients."""
    v_c = self.V[center_idx]
    u_o = self.U[context_idx]
    
    # Forward pass
    score_pos = np.dot(u_o, v_c)
    sigmoid_pos = self._sigmoid(score_pos)
    
    # Gradient computation (manual)
    grad_u_o = (sigmoid_pos - 1.0) * v_c
    grad_v_c = (sigmoid_pos - 1.0) * u_o
    
    # SGD update (manual)
    self.U[context_idx] -= self.alpha * grad_u_o
    # ... negative samples ...
    self.V[center_idx] -= self.alpha * grad_v_c
```

### 3. Statistical Interpretation

Clear connections to:
- Maximum likelihood estimation
- Exponential family models
- Matrix factorization
- Information theory (PMI, entropy)

---

## ⚡ Performance Notes

### Computational Complexity

- **Per training pair**: O(k × d) where k ≪ |V|
- **Full softmax**: O(|V| × d) → **infeasible for large vocabularies**
- **Speedup**: ~20,000× for |V|=100K, k=5

### Training Time

On modest hardware (CPU):
- 1M training pairs, 100-dim, 5 epochs: ~2-5 minutes
- 100M pairs: ~3-5 hours

For production use on large corpora, consider:
- Subsampling frequent words
- Phrase detection
- Parallel processing (pure NumPy is easily parallelizable)

---

## Validation

### Intrinsic Evaluation

1. **Similarity**: Cosine similarity correlates with human judgments
2. **Analogies**: Semantic and syntactic analogy accuracy
3. **Clustering**: K-means on embeddings recovers semantic categories

### Extrinsic Evaluation

Use learned embeddings for:
- Sentiment analysis
- Named entity recognition
- Document classification
- Machine translation

**Note**: This implementation prioritizes clarity over performance. For production, see limitations below.

---

## Limitations & Extensions

### Current Limitations

- **Static embeddings**: One vector per word type (no polysemy handling)
- **No subword information**: Can't handle OOV words
- **Context-independent**: Word order beyond the window is ignored
- **Single-threaded**: No parallelization

### Possible Extensions

1. **Subword embeddings** (FastText):
   ```
   vword = ∑ vn-gram / |n-grams|
   ```

2. **Dynamic context window**:
   Already implemented - randomly samples window size

3. **Phrase detection**:
   Identify multi-word expressions (e.g., "New York")

4. **Hierarchical softmax**:
   Alternative to negative sampling using a binary tree

5. **Contextual embeddings**:
   Use as initialization for LSTM/Transformer models

---

## References

### Original Papers

- **Mikolov et al. (2013)**: "Efficient Estimation of Word Representations in Vector Space."
- **Mikolov et al. (2013)**: "Distributed Representations of Words and Phrases and their Compositionality."

### Theoretical Analysis

- **Levy & Goldberg (2014)**: "Neural Word Embedding as Implicit Matrix Factorization"
- **Goldberg & Levy (2014)**: "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method"

### Related Work

- **GloVe** (Pennington et al., 2014): Explicit co-occurrence matrix factorization
- **FastText** (Bojanowski et al., 2017): Subword embeddings
- **ELMo, BERT**: Contextual embeddings (beyond this project's scope)

---

## Contributing

Contributions welcome! Areas of interest:

- **Performance optimizations** (while maintaining clarity)
- **Visualization tools** (t-SNE, PCA of embeddings)
- **Evaluation benchmarks** (word similarity, analogies)
- **Documentation improvements**
- **Additional examples**

Please maintain the principle: **every line of code must map to a mathematical operation**.

---

## Acknowledgments

This implementation is inspired by:
- The clarity of Mikolov et al.'s original papers
- Goldberg & Levy's theoretical analysis
- The principle that **understanding requires implementation**

Special thanks to the NLP community for decades of research on distributional semantics.

---

## FAQ

**Q: Why implement from scratch when libraries exist?**  
A: Understanding. You can't claim to understand Word2Vec if you can't derive and implement it.

**Q: Is this production-ready?**  
A: No. This prioritizes educational clarity. For production, use gensim or pre-trained embeddings.

**Q: Why NumPy only?**  
A: To remove all abstractions. Every operation is explicit and traceable.

**Q: What about BERT/GPT/transformers?**  
A: Those are different models. Word2Vec is a foundational technique that is worth understanding in depth.

**Q: Can I use this for my research?**  
A: Yes (MIT license), but cite appropriately and consider using established libraries for benchmarking.

---

Happy learning! 🚀

---

*Built with ❤️ and ∇ (gradients)*
