import numpy as np


def sigmoid(x):
    # clip for numerical stability
    """
    Numerically stable sigmoid.
    Clipping avoids overflow in exp for large positive/negative values.
    """
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int = 50, seed: int = 42):
        """
        Initialize two embedding matrices:

        W_in  : input / center word embeddings
        W_out : output / context word embeddings

        Both are randomly initialized with small values."""
        
        rng = np.random.default_rng(seed)

        # Input embeddings: center word vectors
        self.W_in = rng.normal(0, 0.01, size=(vocab_size, embedding_dim))

        # Output embeddings: context word vectors
        self.W_out = rng.normal(0, 0.01, size=(vocab_size, embedding_dim))

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def train_one_example(self, center_id: int, pos_context_id: int, neg_ids, lr: float = 0.025):
        """
        Perform one SGD update for a single skip-gram example.

        Objective:
        maximize log sigma(u_o^T v_c) + sum_k log sigma(-u_k^T v_c)

        Equivalently, minimize:
        -log sigma(u_o^T v_c) - sum_k log sigma(-u_k^T v_c)

        Parameters
        ----------
        center_id : int
            ID of center word
        pos_context_id : int
            ID of true context word
        neg_ids : list[int]
            IDs of negative sampled words
        lr : float
            Learning rate
        """

        # Input embedding of center word: shape (D,)
        v_c = self.W_in[center_id].copy()        
        
        # Output embedding of positive context word: shape (D,)
        u_o = self.W_out[pos_context_id].copy()  

        # ---------- Positive pair ----------
        # Score between center and true context
        pos_score = np.dot(u_o, v_c)
        pos_sig = sigmoid(pos_score)
        
        # Positive loss term: -log(sigmoid(pos_score))
        pos_loss = -np.log(pos_sig + 1e-10)

        # Gradient pieces
        # d(-log(sigmoid(x)))/dx = sigmoid(x) - 1
        grad_pos_score = pos_sig - 1.0

        # Chain rule:
        # d(pos_score)/d(v_c) = u_o
        # d(pos_score)/d(u_o) = v_c
        grad_v_c = grad_pos_score * u_o
        grad_u_o = grad_pos_score * v_c

        # ---------- Negative samples ----------
        neg_loss = 0.0
        neg_grads_u = []

        for neg_id in neg_ids:
            # Output embedding of negative sampled word
            u_k = self.W_out[neg_id].copy()

            # Score between center and negative word
            neg_score = np.dot(u_k, v_c)
            neg_sig = sigmoid(neg_score)

            # Negative loss term: -log(sigmoid(-neg_score))
            neg_loss += -np.log(sigmoid(-neg_score) + 1e-10)

            # d/dx [-log(sigmoid(-x))] = sigmoid(x)
            grad_neg_score = neg_sig

            # Accumulate gradient wrt center vector
            grad_v_c += grad_neg_score * u_k
            
            # Gradient wrt negative output vector
            grad_u_k = grad_neg_score * v_c
            neg_grads_u.append((neg_id, grad_u_k))

        total_loss = pos_loss + neg_loss

        # Parameter updates
        # ---------- SGD updates ----------
        # Update center word input vector
        self.W_in[center_id] -= lr * grad_v_c
        
        # Update positive context output vector
        self.W_out[pos_context_id] -= lr * grad_u_o

        # Update negative sampled output vectors
        for neg_id, grad_u_k in neg_grads_u:
            self.W_out[neg_id] -= lr * grad_u_k

        return total_loss

    def get_input_embeddings(self):
        """
        Return input embeddings.
        These are commonly used as the final word vectors.
        """
        return self.W_in

    def get_output_embeddings(self):
        """ Return output/context embeddings. """
        return self.W_out

    def get_word_vectors(self):
        # Common simple choice: use input embeddings
        """
        Return the chosen final word representations.
        Here we use the input embeddings directly.
        """
        return self.W_in