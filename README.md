# Jowett Plato-Trained (JPT) Language Model

An exercise for learning how a transformer language model works, by building one yourself.

JPT is a small language model that writes text in the style of Plato's dialogues. It has already been trained: the file `weights.npz` contains all of its learned numbers. What's missing is the code that uses those numbers to generate text, and that's your job. You'll write it using nothing but numpy, one step at a time, until the model writes its own (sort of) Platonic dialogue:

```
MENO:
Yes, Euthydemus, I do.

SOCRATES:
And is this wise, like fathers or brothers? ...
```

## What you'll do

The notebook `jpt.ipynb` contains the outline of the model, one function per cell. Fill in every cell except the first and the last, following the step-by-step instructions in [exercises.md](exercises.md). The first cell loads the vocabulary and the weights; the last asks the model to continue a prompt. Once all your functions work, run the notebook from top to bottom and the last cell will generate new text using your code.

## Setup

You need Python 3.10 or newer. Then either:

**With [uv](https://docs.astral.sh/uv/)** (from this folder):

```bash
uv sync
uv run jupyter lab
```

**Or with pip:**

```bash
pip install numpy jupyterlab
jupyter lab
```

Then open `jpt.ipynb`.

## Files

| File | What it is |
|---|---|
| `exercises.md` | Step-by-step instructions for filling in the notebook |
| `jpt.ipynb` | The notebook where you write your code |
| `weights.npz` | The trained model's weights, in numpy format |
| `plato.txt` | The text of Plato's dialogues. The model's vocabulary is the set of characters in this file. |
| `pyproject.toml`, `uv.lock` | Python package requirements (numpy and JupyterLab) |

## About the model

JPT reads and writes one character at a time. Its vocabulary is the 77 characters that appear in `plato.txt`: letters, digits, punctuation, spaces and newlines.

| | |
|---|---|
| Context length | 256 characters |
| Embedding size | 384 |
| Transformer blocks | 6 |
| Attention heads per block | 6 |
| Parameters | about 10.8 million |

It was trained in two stages: first on about 200 MB of philosophy and classics from [Project Gutenberg](https://www.gutenberg.org), to learn English, then on Plato's dialogues alone, to learn to sound like Plato.
