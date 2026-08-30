"""
Fake News Detection -- a Neural Network Built from First Principles (NLP edition)
===================================================================================

Pipeline
--------
1. Data     : synthetic labelled corpus (synthetic_corpus.py) -- see that file's
              docstring for why, and how to swap in the real Kaggle/GFG dataset.
2. Model    : a from-scratch NumPy network --
                  tokens -> trainable Embedding -> mean-pool -> Dense+ReLU -> Dense+Sigmoid
              trained end-to-end by manually-derived backpropagation (no autograd,
              no ML framework). The embedding is learned jointly with the classifier,
              which stands in for the frozen pretrained GloVe vectors used in the
              original tutorial (GloVe requires a download not available here).
3. Baselines: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, TF-IDF + scikit-learn
              MLPClassifier (architecture-matched sanity check, as in the digit project).
4. Evaluation: accuracy, precision/recall/F1, confusion matrix, ROC/AUC, learning
              curves, most-informative-features, and qualitative sample predictions.
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              roc_curve, auc)

from synthetic_corpus import generate_corpus

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)
OUT = "/home/claude/fakenews_project"

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
rows = generate_corpus(n_per_class=450)
texts = [t for t, _ in rows]
labels = np.array([y for _, y in rows], dtype=np.int64)

X_train_txt, X_temp_txt, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.30, stratify=labels, random_state=RNG_SEED
)
X_val_txt, X_test_txt, y_val, y_test = train_test_split(
    X_temp_txt, y_temp, test_size=0.50, stratify=y_temp, random_state=RNG_SEED
)
print(f"Train: {len(X_train_txt)}  Val: {len(X_val_txt)}  Test: {len(X_test_txt)}")

# ---------------------------------------------------------------------------
# 2. Tokenizer (built from scratch: lowercase, word-level, vocab from train only)
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-zA-Z']+")
MAX_LEN = 24
PAD, OOV = 0, 1


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts_list, min_freq=1):
    freq = {}
    for t in texts_list:
        for tok in tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {"<PAD>": PAD, "<OOV>": OOV}
    for tok, c in sorted(freq.items(), key=lambda kv: -kv[1]):
        if c >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    ids = [vocab.get(tok, OOV) for tok in tokenize(text)][:max_len]
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    ids = ids + [PAD] * (max_len - len(ids))
    return ids, mask


vocab = build_vocab(X_train_txt)
vocab_size = len(vocab)
print("Vocabulary size (train only):", vocab_size)


def encode_batch(texts_list, vocab, max_len=MAX_LEN):
    ids_list, mask_list = [], []
    for t in texts_list:
        ids, mask = encode(t, vocab, max_len)
        ids_list.append(ids)
        mask_list.append(mask)
    return np.array(ids_list), np.array(mask_list, dtype=np.float64)


Xtr_ids, Xtr_mask = encode_batch(X_train_txt, vocab)
Xval_ids, Xval_mask = encode_batch(X_val_txt, vocab)
Xte_ids, Xte_mask = encode_batch(X_test_txt, vocab)

# ---------------------------------------------------------------------------
# 3. From-scratch embedding + MLP classifier
# ---------------------------------------------------------------------------
class EmbeddingMLP:
    """
    ids (batch, L) --Embedding(V,d)--> emb (batch, L, d)
                    --masked mean-pool--> pooled (batch, d)
                    --Linear(d,h)+ReLU--> h_act (batch, h)
                    --Linear(h,1)+Sigmoid--> yhat (batch,)

    Backprop is derived by hand for every stage, including scattering the
    pooled-vector gradient back into individual embedding rows.
    """

    def __init__(self, vocab_size, embed_dim, hidden, seed=0):
        g = np.random.default_rng(seed)
        self.E = g.normal(0, 0.1, size=(vocab_size, embed_dim))
        self.W1 = g.normal(0, np.sqrt(2.0 / embed_dim), size=(embed_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = g.normal(0, np.sqrt(2.0 / hidden), size=(hidden, 1))
        self.b2 = np.zeros(1)
        for name in ["E", "W1", "b1", "W2", "b2"]:
            setattr(self, "v" + name, np.zeros_like(getattr(self, name)))

    def forward(self, ids, mask):
        emb = self.E[ids]                                   # (m, L, d)
        n_valid = mask.sum(axis=1, keepdims=True).clip(min=1)  # (m, 1)
        pooled = (emb * mask[:, :, None]).sum(axis=1) / n_valid  # (m, d)
        h = pooled @ self.W1 + self.b1
        a = np.maximum(0, h)
        z = a @ self.W2 + self.b2
        yhat = 1.0 / (1.0 + np.exp(-z))
        return yhat.ravel(), (ids, mask, n_valid, pooled, h, a, yhat)

    def loss(self, yhat, y, l2):
        m = y.shape[0]
        eps = 1e-12
        data_loss = -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))
        reg = (l2 / (2 * m)) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg

    def backward(self, cache, y, l2):
        ids, mask, n_valid, pooled, h, a, yhat = cache
        m = y.shape[0]
        yhat_col = yhat.reshape(-1, 1)
        y_col = y.reshape(-1, 1)

        dz = (yhat_col - y_col) / m                          # (m,1)
        dW2 = a.T @ dz + (l2 / m) * self.W2
        db2 = dz.sum(axis=0)

        da = dz @ self.W2.T                                  # (m,hidden)
        dh = da * (h > 0)
        dW1 = pooled.T @ dh + (l2 / m) * self.W1
        db1 = dh.sum(axis=0)

        dpooled = dh @ self.W1.T                              # (m, d)
        dE = np.zeros_like(self.E)
        contrib = (dpooled[:, None, :] / n_valid[:, :, None]) * mask[:, :, None]  # (m,L,d)
        np.add.at(dE, ids, contrib)

        return dE, dW1, db1, dW2, db2

    def step(self, grads, lr, momentum):
        names = ["E", "W1", "b1", "W2", "b2"]
        for name, g in zip(names, grads):
            v = getattr(self, "v" + name)
            v[:] = momentum * v - lr * g
            setattr(self, name, getattr(self, name) + v)

    def predict_proba(self, ids, mask):
        yhat, _ = self.forward(ids, mask)
        return yhat


def train(net, Xtr_ids, Xtr_mask, y_train, Xval_ids, Xval_mask, y_val,
          epochs=150, batch_size=32, lr=0.5, l2=1e-3, momentum=0.9, lr_decay=0.99,
          patience=15):
    n = Xtr_ids.shape[0]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    cur_lr = lr
    best_val_loss = np.inf
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(epochs):
        perm = rng.permutation(n)
        ids_s, mask_s, y_s = Xtr_ids[perm], Xtr_mask[perm], y_train[perm]
        for start in range(0, n, batch_size):
            ib = ids_s[start:start + batch_size]
            mb = mask_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]
            yhat, cache = net.forward(ib, mb)
            grads = net.backward(cache, yb, l2)
            net.step(grads, cur_lr, momentum)
        cur_lr *= lr_decay

        yhat_tr, _ = net.forward(Xtr_ids, Xtr_mask)
        yhat_val, _ = net.forward(Xval_ids, Xval_mask)
        val_loss = net.loss(yhat_val, y_val, l2)
        history["train_loss"].append(net.loss(yhat_tr, y_train, l2))
        history["val_loss"].append(val_loss)
        history["train_acc"].append(accuracy_score(y_train, yhat_tr >= 0.5))
        history["val_acc"].append(accuracy_score(y_val, yhat_val >= 0.5))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: getattr(net, k).copy() for k in ["E", "W1", "b1", "W2", "b2"]}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    # restore best checkpoint (early stopping)
    for k, v in best_state.items():
        setattr(net, k, v)
    history["best_epoch"] = best_epoch
    history["stopped_epoch"] = epoch
    return history


net = EmbeddingMLP(vocab_size=vocab_size, embed_dim=32, hidden=32, seed=RNG_SEED)
history = train(net, Xtr_ids, Xtr_mask, y_train.astype(np.float64),
                 Xval_ids, Xval_mask, y_val.astype(np.float64),
                 epochs=150, batch_size=32, lr=0.5, l2=1e-3, momentum=0.9, lr_decay=0.99,
                 patience=15)
print(f"Early stopping: restored checkpoint from epoch {history['best_epoch']} "
      f"(stopped at epoch {history['stopped_epoch']})")

test_proba = net.predict_proba(Xte_ids, Xte_mask)
test_pred = (test_proba >= 0.5).astype(int)
test_acc_scratch = accuracy_score(y_test, test_pred)
cm_scratch = confusion_matrix(y_test, test_pred)
report_scratch = classification_report(y_test, test_pred, target_names=["REAL", "FAKE"], digits=3)

print("\n=== From-scratch Embedding+MLP net ===")
print("Test accuracy:", test_acc_scratch)
print(report_scratch)

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].axvline(history["best_epoch"], color="gray", linestyle="--", linewidth=1,
                 label=f"restored checkpoint (epoch {history['best_epoch']})")
axes[0].set_title("Loss (binary cross-entropy + L2)")
axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].axvline(history["best_epoch"], color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_learning_curves.png", dpi=150)
plt.close(fig)

# Confusion matrix
fig, ax = plt.subplots(figsize=(4.2, 3.8))
im = ax.imshow(cm_scratch, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["REAL", "FAKE"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["REAL", "FAKE"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix (test), acc={test_acc_scratch:.3f}")
for i in range(2):
    for j in range(2):
        v = cm_scratch[i, j]
        ax.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm_scratch.max() / 2 else "black", fontsize=12)
fig.colorbar(im, fraction=0.046)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_confusion.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Baselines (TF-IDF based)
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
Xtr_tfidf = tfidf.fit_transform(X_train_txt)
Xval_tfidf = tfidf.transform(X_val_txt)
Xte_tfidf = tfidf.transform(X_test_txt)

results = {}

logreg = LogisticRegression(max_iter=2000)
logreg.fit(Xtr_tfidf, y_train)
logreg_proba = logreg.predict_proba(Xte_tfidf)[:, 1]
results["Logistic Regression (TF-IDF)"] = accuracy_score(y_test, logreg.predict(Xte_tfidf))

svm = LinearSVC()
svm.fit(Xtr_tfidf, y_train)
results["Linear SVM (TF-IDF)"] = accuracy_score(y_test, svm.predict(Xte_tfidf))

sk_mlp = MLPClassifier(hidden_layer_sizes=(32,), activation="relu", solver="sgd",
                        momentum=0.9, learning_rate_init=0.5, alpha=1e-3,
                        max_iter=150, random_state=RNG_SEED)
sk_mlp.fit(Xtr_tfidf, y_train)
sk_mlp_proba = sk_mlp.predict_proba(Xte_tfidf)[:, 1]
results["scikit-learn MLP (TF-IDF)"] = accuracy_score(y_test, sk_mlp.predict(Xte_tfidf))

results["From-scratch Embedding+MLP (this work)"] = test_acc_scratch

print("\n=== Test accuracy comparison ===")
for k, v in results.items():
    print(f"{k:42s} {v:.4f}")

# Bar chart
fig, ax = plt.subplots(figsize=(7.5, 4))
names = list(results.keys())
vals = [results[n] for n in names]
bars = ax.bar(range(len(names)), vals, color=["#888"] * (len(names) - 1) + ["#2b6cb0"])
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.0, 1.02)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("Test accuracy: from-scratch embedding net vs. TF-IDF baselines")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_comparison.png", dpi=150)
plt.close(fig)

# ROC curves
fig, ax = plt.subplots(figsize=(5.5, 5))
for name, proba in [("From-scratch Embedding+MLP", test_proba),
                     ("Logistic Regression (TF-IDF)", logreg_proba),
                     ("scikit-learn MLP (TF-IDF)", sk_mlp_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (test set)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_roc.png", dpi=150)
plt.close(fig)

# Most informative TF-IDF features (interpretability)
feat_names = np.array(tfidf.get_feature_names_out())
coefs = logreg.coef_[0]
top_fake_idx = np.argsort(coefs)[-15:][::-1]
top_real_idx = np.argsort(coefs)[:15]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].barh(range(15), coefs[top_real_idx][::-1], color="#2b6cb0")
axes[0].set_yticks(range(15))
axes[0].set_yticklabels(feat_names[top_real_idx][::-1], fontsize=8)
axes[0].set_title("Top REAL-leaning n-grams")
axes[1].barh(range(15), coefs[top_fake_idx], color="#c53030")
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(feat_names[top_fake_idx], fontsize=8)
axes[1].set_title("Top FAKE-leaning n-grams")
fig.suptitle("Most informative n-grams (Logistic Regression coefficients)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_top_features.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Qualitative sample predictions
# ---------------------------------------------------------------------------
samples = [
    "The transit authority confirmed a 5% increase in rail funding after a routine budget review.",
    "SHOCKING: officials are secretly hiding the truth about the water supply, share before it's deleted!",
    "A new report found the energy subsidy programme had mixed results, according to a government audit.",
    "You won't believe what they did to the school budget \u2014 the mainstream media won't cover this!",
]
sample_ids, sample_mask = encode_batch(samples, vocab)
sample_proba = net.predict_proba(sample_ids, sample_mask)

print("\n=== Sample predictions (from-scratch model) ===")
sample_results = []
for s, p in zip(samples, sample_proba):
    label = "FAKE" if p >= 0.5 else "REAL"
    print(f"[{label}  p(fake)={p:.3f}]  {s}")
    sample_results.append({"text": s, "p_fake": float(p), "prediction": label})

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
with open(f"{OUT}/results.json", "w") as f:
    json.dump({
        "n_train": len(X_train_txt), "n_val": len(X_val_txt), "n_test": len(X_test_txt),
        "vocab_size": vocab_size,
        "test_accuracy_scratch": float(test_acc_scratch),
        "best_epoch": int(history["best_epoch"]),
        "stopped_epoch": int(history["stopped_epoch"]),
        "baseline_accuracies": {k: float(v) for k, v in results.items()},
        "classification_report_scratch": report_scratch,
        "sample_predictions": sample_results,
    }, f, indent=2)

print("\nAll figures and results.json written to", OUT)
