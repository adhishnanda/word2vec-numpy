import re
import random
import requests
from collections import Counter
from typing import List, Tuple, Dict

# Public Shakespeare dataset used in TensorFlow's Word2Vec tutorial
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"


def download_text(save_path: str = "data/shakespeare.txt") -> str:
    """
    Download the raw text dataset and save it locally.
    Returns the downloaded text as a string.
    """
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    text = response.text

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


def load_text(path: str = "data/shakespeare.txt") -> str:
    """
    Load the dataset from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def simple_tokenize(text: str) -> List[str]:
    """
    Minimal preprocessing:
    - lowercase
    - keep only letters and whitespace
    - split on whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return tokens


def build_vocab(tokens: List[str], min_count: int = 5) -> Tuple[List[str], Dict[str, int], Dict[int, str], Counter]:
    """
    Build vocabulary from tokens.
    Words appearing fewer than min_count times are removed.
    Returns:
    - vocab_words: sorted list of words kept in the vocabulary
    - word_to_id: mapping from word -> integer ID
    - id_to_word: reverse mapping from ID -> word
    - filtered_counts: frequency counts only for words in vocabulary
    """
    counts = Counter(tokens)
    vocab_words = [word for word, count in counts.items() if count >= min_count]
    vocab_words.sort()

    word_to_id = {word: i for i, word in enumerate(vocab_words)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    filtered_counts = Counter({w: counts[w] for w in vocab_words})
    return vocab_words, word_to_id, id_to_word, filtered_counts


def encode_tokens(tokens: List[str], word_to_id: Dict[str, int]) -> List[int]:
    """Convert tokens to integer IDs, skipping words not present in the vocabulary."""
    return [word_to_id[t] for t in tokens if t in word_to_id]


def generate_skipgram_pairs(token_ids: List[int], window_size: int = 2) -> List[Tuple[int, int]]:
    """
    Generate (center, context) training pairs for skip-gram.

    For each token position i, all tokens within the context window
    around i are used as positive context words.
    """
    pairs = []
    n = len(token_ids)

    for i, center in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)

        for j in range(left, right):
            if i == j:
                continue
            context = token_ids[j]
            pairs.append((center, context))

    return pairs


def build_negative_sampling_distribution(filtered_counts: Counter, word_to_id: Dict[str, int]):
    """
    Build the negative sampling distribution.

    In the original Word2Vec negative sampling setup, words are sampled with probability proportional to count(word)^0.75.
    This downweights extremely frequent words compared to raw frequency, while still sampling common words more often than rare ones."""
      
    # P(w) proportional to count(w)^0.75 as used in word2vec negative sampling.
    
    vocab_size = len(word_to_id)
    freqs = [0.0] * vocab_size

    for word, idx in word_to_id.items():
        freqs[idx] = filtered_counts[word] ** 0.75

    total = sum(freqs)
    probs = [f / total for f in freqs]
    return probs


def sample_negative_ids(
    num_samples: int,
    vocab_size: int,
    probs,
    forbidden_ids=None,
) -> List[int]:
    
    """
    Simple weighted sampling with rejection for forbidden ids. Sample negative word IDs using the negative sampling distribution.

    forbidden_ids is used to avoid sampling the center word or the true positive context word as a negative example. """


    if forbidden_ids is None:
        forbidden_ids = set()

    samples = []
    while len(samples) < num_samples:
        candidate = random.choices(range(vocab_size), weights=probs, k=1)[0]
        if candidate not in forbidden_ids:
            samples.append(candidate)
    return samples