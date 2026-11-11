# Contributing to Aira-Ai

First off, thank you for considering contributing to Aira-AI! 🎉 This project exists to help people learn how language models work under the hood, and every contribution — code, docs, bug reports, or ideas — helps that mission.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project follows a simple principle: **be respectful, be constructive, be kind**. We welcome contributors of all experience levels — remember that many people using this repo are learning ML for the first time.

## How Can I Contribute?

There are many ways to help:

- 🐛 **Report bugs** you encounter while training or running the model
- 💡 **Suggest features** — new tokenizer options, training tricks, docs improvements
- 📝 **Improve documentation** — clearer explanations help beginners the most
- 🧪 **Add tests** for existing functionality
- 🔧 **Submit code** — bug fixes, new features, performance improvements
- 🌍 **Add example datasets** in different languages

## Development Setup

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/adam-ai.git
   cd adam-ai
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a new branch for your work:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Pull Request Process

1. Make sure your code runs without errors (`python train.py --config config.yaml` should complete a few iterations without crashing).
2. Update the README.md or relevant docs if you've changed behavior or added a feature.
3. Write clear, descriptive commit messages (e.g., `Add BPE tokenizer support`, not `fix stuff`).
4. Ensure your branch is up to date with `main` before opening the PR:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

5. Open a pull request with:
   - A clear title and description of what changed and why
   - Reference to any related issue (e.g., `Closes #12`)
   - Before/after behavior if applicable (e.g., loss curves, sample outputs)

6. A maintainer will review your PR. Please be patient and responsive to feedback — this is a community learning project, and reviews are meant to help, not gatekeep.

## Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable names — this is an educational codebase, clarity beats cleverness.
- Add docstrings to new functions and classes, explaining *what* they do and *why*.
- Keep functions small and focused where possible.
- Comment non-obvious math (e.g., attention score scaling, positional encoding) so learners can follow along.

## Reporting Bugs

When filing a bug report, please include:

- Your OS, Python version, and PyTorch version
- Steps to reproduce the issue
- Expected vs. actual behavior
- Full error traceback, if any
- Your `config.yaml` settings (if training-related)

## Suggesting Enhancements

Open an issue with:

- A clear description of the enhancement
- Why it would be useful (especially for learners)
- Any relevant references, papers, or prior art

---

Thank you for helping make Adam-AI a better learning resource for everyone! 💎
