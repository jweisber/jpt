# Jowett Plato-Trained (JPT) Language Model

A project for learning how a transformer works by building one yourself.

JPT is a small language model that writes text in the style of Plato's dialogues, badly. It's already trained: `weights.npz` contains all the weights. What's missing is the code that uses those weights to generate text. You write that code using NumPy, until the model writes its own (sort of) Platonic dialogue:

```
MENO:
Yes, Euthydemus, I do.

SOCRATES:
And is this wise, like fathers or brothers? ...
```

## What you'll do

The notebook `jpt.ipynb` contains the outline of the model. Follow the step-by-step instructions in the notebook to fill in the missing functions. The first code cell loads the vocabulary and the weights; the last one asks the model to continue a prompt. Under each function you fill in is a test cell, with the expected output in a comment, so you can check your work as you go. Once all the functions are filled in, run the notebook from top to bottom and the last cell will generate new text using your code.

## Files

| File | What it is |
|---|---|
| `jpt.ipynb` | The notebook where you write your code |
| `weights.npz` | The trained model's weights, in numpy format |
| `plato.txt` | The text of Plato's dialogues. The model's vocabulary is the set of characters in this file. |

## About the model

JPT reads and writes one character at a time. Its vocabulary is the 77 characters that appear in `plato.txt`: letters, most digits, punctuation, spaces and newlines.

| | |
|---|---|
| Context length | 256 characters |
| Embedding size | 384 |
| Transformer blocks | 6 |
| Attention heads per block | 6 |
| Parameters | about 10.8 million |

It was trained in two stages: first on about 200 MB of philosophy and classics from [Project Gutenberg](https://www.gutenberg.org), to learn English, then on Plato's dialogues alone, to learn to sound like Plato.

## Shout Out

This project is inspired by [Andrej Karpathy's YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY). If you want to learn to train your own model, that tutorial is an excellent followup project.
