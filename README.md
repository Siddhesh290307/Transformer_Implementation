# Transformer-based Neural Machine Translation

This project implements a **sequence-to-sequence Transformer** model for machine translation, trained from scratch on a parallel corpus. It demonstrates how to build and train a Transformer encoder–decoder architecture for translating sentences from a source language to a target language, and how to evaluate translation quality using BLEU.[web:88][web:92]

---

## Description

The project builds a neural machine translation system based on the original Transformer architecture (“Attention Is All You Need”). It includes data preprocessing (tokenization, vocabulary creation, padding), model implementation (multi-head self-attention, encoder–decoder blocks, positional encoding), training with teacher forcing, and evaluation using corpus-level BLEU scores on a held-out test set.[web:88]

---

## Features

- Transformer encoder–decoder for NMT:
  - Multi-head self-attention in encoder and decoder
  - Encoder–decoder (cross) attention
  - Position-wise feed-forward networks
  - Positional encodings and dropout
- End-to-end translation pipeline:
  - Text cleaning, tokenization, and vocabulary building
  - Padding and masking (padding masks + look-ahead masks)
- Evaluation:
  - Corpus BLEU score on test data
  - Sample qualitative translations (source vs. reference vs. predicted)

---

## Project Structure

.
├── model/
│   ├── attention.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── fnn.py
│   ├── masking.py
│   ├── positional_encoding.py
│   ├── transformer.py
│   └── __pycache__/
│
├── training/
│   └── Transformer_Training.ipynb
│
└── README.md

## Observations

- This repository extends my previous Neural Machine Translation implementation based on an encoder–decoder LSTM architecture (with and without attention):
https://github.com/Siddhesh290307/Neural-Machine-Translation

- A comparative analysis between the LSTM-based model and the Transformer architecture reveals distinct behavioral differences under limited training conditions:
    - Grammatical Structure:
        The Transformer produced outputs with stronger syntactic consistency and more stable sentence structure, even with relatively limited training.
    - Semantic Coherence:
        The LSTM encoder–decoder demonstrated comparatively better semantic retention in certain cases, particularly when trained on smaller datasets.
    - Data Sensitivity:
        With constrained training data and epochs, the LSTM occasionally matched or exceeded Transformer performance in BLEU score and semantic fidelity. This suggests that Transformers may require larger datasets to fully leverage their representational capacity.

- These observations support the broader hypothesis that:
Under low-resource or limited-training regimes, recurrent architectures can remain competitive with self-attention-based models.

- This comparison motivates further investigation into:
    - Data efficiency of Transformers vs LSTMs
    - Scaling behavior with increasing dataset size
    - Interpretability of attention mechanisms across architectures

## Future Improvements

- I will work towards extending the same Transformer architecture to more domains of NLP as well as further train the same model on more epochs.