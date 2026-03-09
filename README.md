# Word2Vec in Pure NumPy

This repository implements the core training loop of **Word2Vec** in **pure NumPy**, without using PyTorch, TensorFlow, or any other machine learning framework.

The implemented variant is:

- **Skip-gram**
- **Negative Sampling**

The goal of this project is to demonstrate a full understanding of the Word2Vec training procedure, including:

- forward pass
- loss computation
- gradient derivation
- parameter updates with SGD

---

## Task Summary

The assignment was to implement the optimization procedure of a standard Word2Vec variant in pure NumPy.

I chose to implement **skip-gram with negative sampling**, because it is a standard and efficient Word2Vec formulation that avoids the computational cost of full softmax over the entire vocabulary.

---

## Dataset

I used the public **Shakespeare text corpus**, specifically the same text file used in TensorFlow’s official Word2Vec tutorial.

This dataset was chosen because it is:

- publicly accessible
- easy to preprocess
- small enough for efficient training in pure NumPy
- suitable for demonstrating learned semantic/contextual word representations

---

## Implemented Method

### 1. Preprocessing
The text is:

- lowercased
- stripped of punctuation and non-alphabetic characters
- tokenized by whitespace
- filtered by minimum frequency (`min_count=5`)

### 2. Vocabulary Construction
A vocabulary is built from the filtered tokens, and each word is mapped to an integer ID.

### 3. Skip-gram Training Pairs
For each center word, surrounding words inside a fixed context window are used as positive context targets.

### 4. Negative Sampling
Instead of computing a full softmax across the whole vocabulary, the model samples a small number of negative words for each positive pair.

This makes training much more efficient.

### 5. Embedding Matrices
Two embedding matrices are learned:

- `W_in`: input / center word embeddings
- `W_out`: output / context word embeddings

For final word representations, the input embeddings (`W_in`) are used.

---

## Objective Function

For one training example with:

- center word embedding: `v_c`
- positive context embedding: `u_o`
- negative sample embeddings: `u_k`

the loss is:

L = -log(sigmoid(u_o^T v_c)) - sum_k log(sigmoid(-u_k^T v_c))

This objective:

- increases similarity between true center-context pairs
- decreases similarity between center words and randomly sampled negative words

---

## Gradient Intuition

For the positive pair:

- if the dot product is too small, the model should pull the center and positive context vectors closer

For the negative pairs:

- if the dot product is too large, the model should push the center and negative vectors apart

The gradients are implemented manually in NumPy and updated with SGD.

---

## Project Structure

```text
word2vec-numpy/
├── data/
│   └── shakespeare.txt
├── results/
│   ├── embeddings.npy
│   └── nearest_neighbors.txt
├── src/
│   ├── data_utils.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Hyperparameters

The implementation currently uses:

- **Embedding dimension:** `50`  
- **Context window size:** `2`  
- **Negative samples per positive pair:** `5`  
- **Learning rate:** `0.025`  
- **Epochs:** `2`  
- **Minimum word frequency (`min_count`):** `5`

To keep training manageable in pure NumPy, only the first **120000 tokens** of the corpus are used.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt

### 2. Run training
```bash
python src/train.py

---

## Output

The training script produces the following outputs:

`results/embeddings.npy`
Contains the learned word embeddings saved as a NumPy array.

`results/nearest_neighbors.txt`
Contains example nearest-neighbor results using cosine similarity between word vectors.

---

## Example Observations

During training, the average loss decreases significantly, which indicates that the implementation is learning useful word relationships.

Example nearest-neighbor outputs include corpus-specific associations such as:

- `king` → `richard`, `crown, `iv`, `vi`
- `queen` → `margaret`, `elizabeth`
- `lord` → `buckingham`, `hastings`

These results are consistent with the Shakespeare corpus and suggest that the embeddings capture contextual similarity between words appearing in similar contexts.

---

## Limitations

This implementation is intentionally simple and educational, focusing on demonstrating the full training loop in NumPy.

The current implementation does not include:
- Mini-batch training
- Subsampling of very frequent words
- Learning rate decay
- Hierarchical softmax
- GPU acceleration
- Advanced evaluation benchmarks

---

## Possible Improvements

Several extensions could improve this implementation:
- Add subsampling of frequent words to reduce the influence of extremely common tokens
- Implement learning rate scheduling
- Implement CBOW (Continuous Bag of Words) as an alternative training variant
- Experiment with combining input and output embeddings
- Add intrinsic evaluation tasks, such as word analogy tests
- Improve preprocessing and sentence segmentation

---

## Key Learning Outcome

This project demonstrates a full from-scratch implementation of Word2Vec in NumPy, including:
- Skip-gram training pair generation
- Negative sampling
- Manual gradient derivation
- SGD-based parameter updates
- Simple embedding evaluation using cosine similarity
