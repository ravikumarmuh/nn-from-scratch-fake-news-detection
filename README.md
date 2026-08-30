# Fake News Detection — Neural Network from First Principles (NLP)

A binary fake/real news classifier built around a trainable word-embedding
layer, mean-pooling, and a feedforward classifier — implemented entirely from
scratch in NumPy, including backpropagation through the embedding layer
itself.

**Test accuracy: 83.0%** (AUC 0.906), essentially tied with TF-IDF logistic
regression (AUC 0.913), evaluated with early stopping and a documented
limitation analysis rather than a headline number alone.

## Why from scratch, and an important note on the data

The original approach for this kind of task uses a downloaded labelled
dataset and frozen pretrained GloVe embeddings. This project instead:

- **Learns the embedding end-to-end** with the classifier, rather than
  loading it frozen — an embedding is just another parameter matrix, and
  implementing it that way makes the mechanism explicit.
- **Uses a synthetic, template-generated corpus** (900 examples, documented
  in [`synthetic_corpus.py`](synthetic_corpus.py)) built from two known
  stylistic registers (neutral/sourced vs. sensational/urgent), including a
  deliberate 20% subset of "sophisticated" fake examples written in neutral,
  real-news style — which is what keeps the task from being trivially
  separable on vocabulary alone. **This is not a real-world dataset**, and
  the report is explicit that the numbers validate the pipeline and
  implementation, not real-world fake-news detection rates. Swapping in a
  real labelled dataset requires no code changes beyond the loader function
  (`load_real_dataset_instead`).

## Results

| Model | Test accuracy | AUC |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.859 | 0.913 |
| Linear SVM (TF-IDF) | 0.859 | — |
| scikit-learn MLP (TF-IDF) | 0.748 | 0.885 |
| **From-scratch Embedding+MLP (this work)** | **0.830** | **0.906** |

## What the model actually learned

Feature-importance analysis (see the report) shows every informative signal
is a stylistic or sourcing marker (*leaked*, *insiders*, *you* vs. *review*,
*officials*, *residents*) — the classifier is doing register detection, not
fact-checking, which mirrors a genuine and important limitation of purely
stylistic fake-news detectors in practice.

## Contents

- [`report.pdf`](report.pdf) / [`report.tex`](report.tex) — full write-up: data-substitution rationale, architecture, embedding backprop derivation, results, and discussion
- [`synthetic_corpus.py`](synthetic_corpus.py) — labelled corpus generator, plus the real-dataset drop-in loader
- [`nn_fakenews.py`](nn_fakenews.py) — tokenizer, `EmbeddingMLP` class (forward/backward/step), training with early stopping, TF-IDF baselines, and all figures
- [`figures/`](figures) — learning curves, confusion matrix, ROC curves, top informative n-grams, baseline comparison

## Running it

```bash
pip install numpy scikit-learn matplotlib
python nn_fakenews.py
```

No internet access or GPU required.

## Author

Ravi Kumar
