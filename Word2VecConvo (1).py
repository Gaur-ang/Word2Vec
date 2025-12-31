import numpy as np
from collections import Counter, defaultdict
import re

class Word2Vec:

    def __init__(self, embedding_dim=100, window_size=5, neg_samples=5,
                 learning_rate=0.025, min_count=5):
        """
        Parameters:
        -----------
        embedding_dim : int
            Dimension d of embedding vectors
        window_size : int
            Context window size c (looks c words left and right)
        neg_samples : int
            Number k of negative samples per positive pair
        learning_rate : float
            Initial learning rate α for SGD
        min_count : int
            Minimum word frequency to be included in vocabulary
        """
        self.d = embedding_dim
        self.c = window_size
        self.k = neg_samples
        self.alpha = learning_rate
        self.min_count = min_count

        # Parameters θ = {V, U}
        self.V = None  # Input embeddings: |V| x d
        self.U = None  # Output embeddings: |V| x d

        # Vocabulary
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

        # Noise distribution P_n(w) ∝ count(w)^0.75
        self.noise_dist = None

    def _tokenize(self, text):
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, corpus):
        """
        Build vocabulary from corpus.

        Parameters:
        -----------
        corpus : str or list of str
            Input text corpus

        Creates:
        --------
        - word2idx: word -> integer index mapping
        - idx2word: integer index -> word mapping
        - noise_dist: sampling distribution for negative samples
        """
        if isinstance(corpus, str):
            tokens = self._tokenize(corpus)
        else:
            tokens = []
            for text in corpus:
                tokens.extend(self._tokenize(text))

        # Count word frequencies
        word_counts = Counter(tokens)

        # Filter by minimum count
        vocab_words = [w for w, c in word_counts.items() if c >= self.min_count]
        vocab_words.sort()  # Deterministic ordering

        # Create mappings
        self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab_words)

        print(f"Vocabulary size: {self.vocab_size}")

        # Build noise distribution: P_n(w) ∝ count(w)^0.75
        counts = np.array([word_counts[w] for w in vocab_words], dtype=np.float64)
        counts_smooth = np.power(counts, 0.75)
        self.noise_dist = counts_smooth / counts_smooth.sum()

        return tokens

    def _initialize_embeddings(self):
        """
        Initialize embedding matrices V and U.

        Uses uniform distribution in [-0.5/d, 0.5/d] for small random values.
        """
        bound = 0.5 / self.d
        self.V = np.random.uniform(-bound, bound, (self.vocab_size, self.d))
        self.U = np.random.uniform(-bound, bound, (self.vocab_size, self.d))

    def _generate_training_pairs(self, tokens):
        """
        Generate (center, context) training pairs from tokenized corpus.

        For each position t, generates pairs (w_t, w_{t+j}) for j in [-c, c], j ≠ 0.

        Parameters:
        -----------
        tokens : list of str
            Tokenized corpus

        Returns:
        --------
        pairs : list of (int, int)
            List of (center_idx, context_idx) pairs
        """
        pairs = []
        T = len(tokens)

        for t in range(T):
            center_word = tokens[t]
            if center_word not in self.word2idx:
                continue

            center_idx = self.word2idx[center_word]

            # Dynamic window: randomly sample window size from [1, c]
            window = np.random.randint(1, self.c + 1)

            for j in range(-window, window + 1):
                if j == 0:
                    continue

                context_pos = t + j
                if context_pos < 0 or context_pos >= T:
                    continue

                context_word = tokens[context_pos]
                if context_word not in self.word2idx:
                    continue

                context_idx = self.word2idx[context_word]
                pairs.append((center_idx, context_idx))

        return pairs

    def _sigmoid(self, x):
        """
        Sigmoid function: σ(x) = 1 / (1 + exp(-x))

        Numerically stable implementation.
        """
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _negative_sampling_loss(self, center_idx, context_idx):
        """
        Compute negative sampling loss for one training pair.

        J_NS = -log σ(u_o^T v_c) - Σ log σ(-u_i^T v_c)

        Parameters:
        -----------
        center_idx : int
            Index of center word w_c
        context_idx : int
            Index of observed context word w_o

        Returns:
        --------
        loss : float
            Negative sampling loss for this pair
        """
        # Get embeddings
        v_c = self.V[center_idx]  # Shape: (d,)
        u_o = self.U[context_idx]  # Shape: (d,)

        # Positive score: s_o = u_o^T v_c
        score_pos = np.dot(u_o, v_c)

        # Positive term: log σ(s_o)
        loss = -np.log(self._sigmoid(score_pos))

        # Sample k negative examples
        neg_indices = np.random.choice(
            self.vocab_size,
            size=self.k,
            p=self.noise_dist
        )

        for neg_idx in neg_indices:
            u_i = self.U[neg_idx]
            score_neg = np.dot(u_i, v_c)

            # Negative term: log σ(-s_i)
            loss -= np.log(self._sigmoid(-score_neg))

        return loss

    def _update_embeddings(self, center_idx, context_idx):
        """
        Perform SGD update for one training pair using negative sampling.

        Gradients:
        - ∂J/∂u_o = [σ(s_o) - 1] v_c
        - ∂J/∂u_i = σ(s_i) v_c  for negative samples
        - ∂J/∂v_c = [σ(s_o) - 1] u_o + Σ σ(s_i) u_i

        Parameters:
        -----------
        center_idx : int
            Index of center word w_c
        context_idx : int
            Index of observed context word w_o
        """
        # Get embeddings
        v_c = self.V[center_idx].copy()  # Shape: (d,)
        u_o = self.U[context_idx]        # Shape: (d,)

        # Positive score
        score_pos = np.dot(u_o, v_c)
        sigmoid_pos = self._sigmoid(score_pos)

        # Gradient for positive sample
        # ∂J/∂u_o = [σ(s_o) - 1] v_c
        grad_u_o = (sigmoid_pos - 1.0) * v_c

        # Gradient accumulator for v_c
        # ∂J/∂v_c = [σ(s_o) - 1] u_o + Σ σ(s_i) u_i
        grad_v_c = (sigmoid_pos - 1.0) * u_o

        # Update positive context embedding
        self.U[context_idx] -= self.alpha * grad_u_o

        # Sample and update negative samples
        neg_indices = np.random.choice(
            self.vocab_size,
            size=self.k,
            p=self.noise_dist
        )

        for neg_idx in neg_indices:
            u_i = self.U[neg_idx]
            score_neg = np.dot(u_i, v_c)
            sigmoid_neg = self._sigmoid(score_neg)

            # Gradient for negative sample
            # ∂J/∂u_i = σ(s_i) v_c
            grad_u_i = sigmoid_neg * v_c

            # Accumulate gradient for v_c
            grad_v_c += sigmoid_neg * u_i

            # Update negative sample embedding
            self.U[neg_idx] -= self.alpha * grad_u_i

        # Update center embedding
        self.V[center_idx] -= self.alpha * grad_v_c

    def train(self, corpus, epochs=5, verbose=True):
        """
        Train Word2Vec model using SGD with negative sampling.

        Parameters:
        -----------
        corpus : str or list of str
            Training corpus
        epochs : int
            Number of passes through the data
        verbose : bool
            Print training progress
        """
        # Build vocabulary and tokenize
        tokens = self.build_vocab(corpus)

        # Initialize embeddings
        self._initialize_embeddings()

        # Generate training pairs
        pairs = self._generate_training_pairs(tokens)
        n_pairs = len(pairs)
        print(f"Generated {n_pairs} training pairs")

        # Training loop
        initial_alpha = self.alpha

        for epoch in range(epochs):
            # Shuffle pairs for stochastic gradient descent
            np.random.shuffle(pairs)

            total_loss = 0.0

            for i, (center_idx, context_idx) in enumerate(pairs):
                # Linear learning rate decay
                self.alpha = initial_alpha * (1.0 - i / (epochs * n_pairs))

                # Compute loss (for monitoring)
                if verbose and i % 100000 == 0:
                    loss = self._negative_sampling_loss(center_idx, context_idx)
                    total_loss += loss

                # SGD update
                self._update_embeddings(center_idx, context_idx)

            if verbose:
                avg_loss = total_loss / (n_pairs / 100000)
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        print("Training complete!")

    def get_embedding(self, word):
        """
        Get the learned embedding for a word.

        Returns the input (center) embedding v_w.
        """
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        return self.V[self.word2idx[word]]

    def similarity(self, word1, word2):
        """
        Compute cosine similarity between two words.

        sim(w1, w2) = (v_w1 · v_w2) / (||v_w1|| ||v_w2||)
        """
        v1 = self.get_embedding(word1)
        v2 = self.get_embedding(word2)

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def most_similar(self, word, topn=10):
        """
        Find the most similar words to a given word.

        Uses cosine similarity between input embeddings.
        """
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")

        word_idx = self.word2idx[word]
        query_vec = self.V[word_idx]

        # Normalize query vector
        query_norm = query_vec / np.linalg.norm(query_vec)

        # Compute similarities with all words
        # Normalize all vectors
        norms = np.linalg.norm(self.V, axis=1, keepdims=True)
        V_norm = self.V / (norms + 1e-10)

        # Cosine similarity = dot product of normalized vectors
        similarities = V_norm @ query_norm

        # Get top-n (excluding the word itself)
        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if idx != word_idx:
                results.append((self.idx2word[idx], similarities[idx]))
                if len(results) >= topn:
                    break

        return results

    def analogy(self, word_a, word_b, word_c, topn=5):
        """
        Solve word analogy: word_a is to word_b as word_c is to ?

        Uses vector arithmetic: v_b - v_a + v_c ≈ v_?

        Example: king - man + woman ≈ queen
        """
        if any(w not in self.word2idx for w in [word_a, word_b, word_c]):
            raise ValueError("All words must be in vocabulary")

        # Vector arithmetic
        v_a = self.get_embedding(word_a)
        v_b = self.get_embedding(word_b)
        v_c = self.get_embedding(word_c)

        target_vec = v_b - v_a + v_c
        target_norm = target_vec / np.linalg.norm(target_vec)

        # Find most similar
        norms = np.linalg.norm(self.V, axis=1, keepdims=True)
        V_norm = self.V / (norms + 1e-10)
        similarities = V_norm @ target_norm

        # Get top-n (excluding input words)
        top_indices = np.argsort(similarities)[::-1]

        exclude = {self.word2idx[w] for w in [word_a, word_b, word_c]}
        results = []
        for idx in top_indices:
            if idx not in exclude:
                results.append((self.idx2word[idx], similarities[idx]))
                if len(results) >= topn:
                    break

        return results


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Sample corpus (in practice, use much larger corpus)
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        corpus = f.read()

    print("=" * 70)
    print("WORD2VEC FROM FIRST PRINCIPLES — SHAKESPEARE")
    print("=" * 70)
    print(f"Corpus length (characters): {len(corpus)}")
    print()


    # Initialize model
    model = Word2Vec(
        embedding_dim=100,
        window_size=5,
        neg_samples=5,
        learning_rate=0.025,
        min_count=5
    )

    # Train
    print("Training...")
    model.train(corpus, epochs=10, verbose=True)
    print()

    # Test similarity
    print("=" * 70)
    print("SIMILARITY TESTS")
    print("=" * 70)

    test_words = ['king', 'queen', 'man', 'woman', 'rome', 'war']
    for word in test_words:
        if word in model.word2idx:
            print(f"\nMost similar to '{word}':")
            similar = model.most_similar(word, topn=5)
            for w, score in similar:
                print(f"  {w}: {score:.4f}")

    # Test analogy
    print()
    print("=" * 70)
    print("ANALOGY TESTS")
    print("=" * 70)

    analogies = [
        ('king', 'man', 'queen'),
        ('man', 'woman', 'king'),
    ]

    for word_a, word_b, word_c in analogies:
        try:
            print(f"\n{word_a} - {word_b} + {word_c} = ?")
            results = model.analogy(word_a, word_b, word_c, topn=3)
            for w, score in results:
                print(f"  {w}: {score:.4f}")
        except ValueError as e:
            print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("Note: For better results, train on much larger corpus")
    print("=" * 70)

