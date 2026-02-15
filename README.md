# Transformer Neural Machine Translation (English → French)

This project implements a full **Transformer Encoder–Decoder architecture** for Neural Machine Translation (NMT), inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

The objective is to build the Transformer model from scratch and evaluate its performance on an **English–French translation task** using the **BLEU score metric**. The implementation includes positional encoding, multi-head attention, masking strategies, and sequence generation during inference.

This repository also serves as a continuation of my previous [Seq2Seq LSTM-based NMT work](https://github.com/Siddhesh290307/Neural-Machine-Translation), enabling architectural comparison between recurrent and self-attention-based approaches.

## Dataset

- **Source Language**: English
- **Target Language**: French
- **Dataset Size**: 175,621 sentence pairs
- **Source**: [Kaggle – English–French Language Translation Dataset](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)

### Preprocessing Steps
- Lowercasing and normalization
- Tokenization
- Vocabulary construction
- Addition of special tokens (`<SOS>`, `<EOS>`)
- Padding to uniform sequence length
- Creation of padding and look-ahead masks

## Model Architecture

**Transformer Encoder–Decoder** following the original design:
Encoder: Multi-Head Self-Attention → Feed-Forward → Residual + LayerNorm
Decoder: Masked Multi-Head Self-Attention → Encoder-Decoder Attention → Feed-Forward → Residual + LayerNorm


**Key Components:**
- Multi-Head Self-Attention (Encoder)
- Masked Multi-Head Self-Attention (Decoder)
- Encoder–Decoder Cross Attention
- Position-wise Feed-Forward Networks
- Residual Connections + Layer Normalization
- Sinusoidal Positional Encoding

### Attention Mechanism
The scaled dot-product attention is defined as:
Attention(Q,K,V) = softmax(QK^T / √d_k) V


**Key Characteristics:**
- Fully parallelizable (no recurrence)
- Global dependency modeling via self-attention
- Higher representational capacity than traditional RNNs
- Better scalability with large datasets

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Masked Cross-Entropy |
| **Embedding Dimension** | 128 |
| **Number of Heads** | 8 |
| **Feedforward Dimension** | 512 |
| **Teacher Forcing** | Enabled |
| **Evaluation Metric** | Corpus-level BLEU Score |

## Project Structure

Transformer_Implementation/
│
├── model/
│ ├── attention.py
│ ├── encoder.py
│ ├── decoder.py
│ ├── fnn.py
│ ├── masking.py
│ ├── positional_encoding.py
│ ├── transformer.py
│ └── pycache/
│
├── training/
│ └── Transformer_Training.ipynb
│
└── README.md


## Comparative Context (LSTM vs Transformer)

This project extends my earlier implementation: [Neural Machine Translation (Encoder–Decoder LSTM with Attention)](https://github.com/Siddhesh290307/Neural-Machine-Translation)

### Observations Under Limited Training
- **Grammatical Consistency**: Transformer demonstrated stronger syntactic stability
- **Semantic Retention**: LSTM occasionally preserved semantic meaning better under limited data
- **Data Sensitivity**: Transformers required larger datasets and more epochs to consistently outperform recurrent models

### Research Insight
Under low-resource or constrained training regimes, recurrent architectures can remain competitive with self-attention-based models.

This motivates further investigation into:
- Data efficiency comparisons
- Scaling behavior analysis
- Interpretability of attention mechanisms
- Low-resource translation performance

## Future Improvements
- [ ] Increase training epochs
- [ ] Implement Beam Search decoding
- [ ] Add label smoothing
- [ ] Introduce learning rate warmup scheduling
- [ ] Conduct formal BLEU comparison with LSTM baseline
- [ ] Extend to multilingual translation

## References
1. **Attention Is All You Need**  
   Vaswani, A., et al. (2017).  
   *Advances in Neural Information Processing Systems (NeurIPS 2017)*.

2. **BLEU: a Method for Automatic Evaluation of Machine Translation**  
   Papineni, K., et al. (2002).  
   *ACL 2002*.
