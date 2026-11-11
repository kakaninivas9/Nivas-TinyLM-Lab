# 🧠 Aira-AI: Build Your Own LLM from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Maintained by:** Engineer Adam 💎

An educational, open-source repository containing the complete source code to build and train a **Small Language Model (SLM)** from scratch. Instead of shipping a black-box pre-trained model, this project opens up the entire "brain logic" — tokenization, embeddings, self-attention, and training loops — so learners and researchers can study it, modify it, and train it on their own custom data.

---

## 📖 Table of Contents

- [Why This Repo?](#-why-this-repo)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Training Your Model](#-training-your-model)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Citing This Project](#-citing-this-project)
- [License](#-license)

---

## 🚀 Why This Repo?

Most tutorials either hand you a pre-trained model or hide the internals behind an API call. **Adam-AI** is built for people who want to *see* how a language model actually learns:

- **Learn AI Internals** — Understand how Transformers, embeddings, and self-attention work by reading (and running) real, minimal code.
- **Custom Training** — Train the model on your own text corpus — Malayalam literature, Python source code, poetry, anything.
- **Lightweight by Design** — Built to run on consumer-grade hardware (a decent GPU or even CPU for small experiments), not a data center.
- **No Black Boxes** — Every component (tokenizer, attention head, optimizer loop) is implemented transparently and documented.

## ✨ Features

- 🔤 Simple, hackable tokenizer (character or BPE-based)
- 🧩 Transformer architecture implemented from first principles
- 🎯 Configurable model size (layers, heads, embedding dimension)
- 📊 Training loss/metrics logging
- 💾 Checkpointing and resume-from-checkpoint support
- 🌍 Language-agnostic — train on any UTF-8 text, including Malayalam

## 📁 Project Structure

```
adam-ai/
├── data/
│   └── data.txt              # Your training corpus goes here
├── model/
│   ├── attention.py           # Self-attention implementation
│   ├── transformer.py         # Core transformer blocks
│   └── tokenizer.py           # Tokenizer logic
├── train.py                   # Training entry point
├── generate.py                # Text generation / inference script
├── config.yaml                # Hyperparameters and settings
├── requirements.txt
├── paper.md                   # Academic writeup of the project
├── paper.bib                  # Bibliography for paper.md
├── LICENSE
└── README.md
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
git clone https://github.com/<your-username>/adam-ai.git
cd adam-ai
pip install -r requirements.txt
```

Or install PyTorch directly if you're setting things up manually:

```bash
pip install torch
```

## 🎓 Training Your Model

1. **Prepare your data**

   Place your training text into `data/data.txt`. This can be in any language — Malayalam, English, code, or a mix.

   ```bash
   cp your-text-file.txt data/data.txt
   ```

2. **Configure your model** (optional)

   Edit `config.yaml` to adjust model size, batch size, learning rate, etc.

3. **Start training**

   ```bash
   python train.py --config config.yaml
   ```

4. **Generate text from your trained model**

   ```bash
   python generate.py --prompt "Once upon a time"
   ```

## ⚙️ Configuration

Key parameters you can tune in `config.yaml`:

| Parameter | Description | Default |
|---|---|---|
| `n_layers` | Number of transformer blocks | 6 |
| `n_heads` | Number of attention heads | 6 |
| `embed_dim` | Embedding dimension | 384 |
| `block_size` | Context window length | 256 |
| `batch_size` | Training batch size | 64 |
| `learning_rate` | Optimizer learning rate | 3e-4 |
| `max_iters` | Total training iterations | 5000 |

## 🗺️ Roadmap

- [ ] Add BPE tokenizer option
- [ ] Add mixed-precision training support
- [ ] Add support for multi-GPU training
- [ ] Provide pre-tokenized sample datasets
- [ ] Add Colab notebook for zero-setup training

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## 📄 Citing This Project

If you use Adam-AI in your research or teaching, please cite it using the details in [`paper.md`](paper.md) and [`paper.bib`](paper.bib).

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with 💎 for the ML learning community</p>
