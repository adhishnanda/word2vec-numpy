import os
import random
import numpy as np

from data_utils import (
    download_text,
    load_text,
    simple_tokenize,
    build_vocab,
    encode_tokens,
    generate_skipgram_pairs,
    build_negative_sampling_distribution,
    sample_negative_ids,
)
from model import SkipGramNegativeSampling


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def nearest_neighbors(word, word_to_id, id_to_word, vectors, top_k=5):
    """
    Return the top_k nearest neighbors of a word based on cosine similarity.
    """
    if word not in word_to_id:
        return []

    w_id = word_to_id[word]
    query = vectors[w_id]

    sims = []
    for i in range(len(vectors)):
        if i == w_id:
            continue
        sim = cosine_similarity(query, vectors[i])
        sims.append((id_to_word[i], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def main():
    random.seed(42)
    np.random.seed(42)

    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    data_path = "data/shakespeare.txt"
    if not os.path.exists(data_path):
        print("Downloading dataset...")
        download_text(data_path)

    print("Loading text...")
    text = load_text(data_path)

    print("Tokenizing...")
    tokens = simple_tokenize(text)

    # To make 1-day completion easy, reduce corpus size a bit if needed.
    # We can use all tokens too, but this keeps training faster.
    max_tokens = 120000
    tokens = tokens[:max_tokens]

    print(f"Total raw tokens used: {len(tokens)}")

    print("Building vocabulary...")
    vocab_words, word_to_id, id_to_word, filtered_counts = build_vocab(tokens, min_count=5)
    token_ids = encode_tokens(tokens, word_to_id)

    print(f"Vocabulary size: {len(vocab_words)}")
    print(f"Encoded tokens: {len(token_ids)}")

    print("Generating skip-gram pairs...")
    pairs = generate_skipgram_pairs(token_ids, window_size=2)
    print(f"Training pairs: {len(pairs)}")

    probs = build_negative_sampling_distribution(filtered_counts, word_to_id)

    model = SkipGramNegativeSampling(
        vocab_size=len(vocab_words),
        embedding_dim=50,
        seed=42,
    )

    epochs = 2
    lr = 0.025
    num_negative = 5

    print("Training...")
    for epoch in range(epochs):
        random.shuffle(pairs)

        total_loss = 0.0
        report_every = 20000

        for step, (center_id, pos_context_id) in enumerate(pairs, start=1):
            neg_ids = sample_negative_ids(
                num_samples=num_negative,
                vocab_size=len(vocab_words),
                probs=probs,
                forbidden_ids={center_id, pos_context_id},
            )

            loss = model.train_one_example(
                center_id=center_id,
                pos_context_id=pos_context_id,
                neg_ids=neg_ids,
                lr=lr,
            )
            total_loss += loss

            if step % report_every == 0:
                avg_loss = total_loss / report_every
                print(f"Epoch {epoch+1} | Step {step}/{len(pairs)} | Avg loss: {avg_loss:.4f}")
                total_loss = 0.0

    print("Saving embeddings...")
    vectors = model.get_word_vectors()
    np.save("results/embeddings.npy", vectors)

    print("Writing nearest neighbors...")
    probe_words = ["king", "queen", "love", "death", "man", "woman", "lord"]
    with open("results/nearest_neighbors.txt", "w", encoding="utf-8") as f:
        for word in probe_words:
            nns = nearest_neighbors(word, word_to_id, id_to_word, vectors, top_k=5)
            line = f"{word}: {nns}\n"
            print(line.strip())
            f.write(line)

    print("Done.")


if __name__ == "__main__":
    main()