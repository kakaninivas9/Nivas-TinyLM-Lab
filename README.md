# Nivas TinyLM Lab

An educational adaptation of [Aira-AI-Non-Trained](https://github.com/adameehanairagroup/Aira-AI-Non-Trained), preserving the original MIT license and credit. Prepared with Codex assistance; the transformer architecture is upstream work. See [UPSTREAM.md](UPSTREAM.md).

## Added in this adaptation

- Save evaluated checkpoints after training updates, including one-step runs.

- Import-safe dataset preparation: importing `main` does not read files, download data, or start training.
- Explicit local UTF-8 corpus selection and configurable CPU/CUDA training settings.
- Seeded runs, clear errors for undersized train/validation splits and unknown prompt characters.
- Sampling validation and checkpoints containing vocabulary and architecture settings.
- Tiny CPU regression tests and GitHub Actions.
- Console training messages compatible with Windows terminals using legacy encodings.

## Quick start

Use Python 3.11+ in a virtual environment and install dependencies:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --data examples/corpus.txt --steps 1 --batch-size 2 --block-size 8 --eval-iters 1
```

The included repetitive corpus is synthetic programming-practice text. One step is a smoke test, not a trained language model. The default device is CPU; `--device cuda` requires a compatible PyTorch/CUDA installation. No data downloads occur automatically.

The checkpoint stores model weights, characters, context size, embedding size, attention heads, layers and dropout. It does not store optimizer/RNG state or provide exact training resumption. Use only trusted checkpoint files.

## Validation and limits

Tests exercise shifted batches, short datasets, unknown characters, sampling validation, a one-step CPU run, checkpoint loading and generated token shape. No accuracy, benchmark, disease detection or production-readiness claims are made. Results depend on the corpus and training budget; a useful model needs substantially more data and evaluation. No pretrained weights are included.

[Original documentation](UPSTREAM-README.md) describes the upstream project; its branding and research claims are not claims about this adaptation.
