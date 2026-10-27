# Exercises: Build Your Own Transformer

> This project is based on Andrej Karpathy's excellent [YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY). If you want to learn how to train your own model, that tutorial is a good followup project.

The exercises in this section walk you through the steps of building your own transformer. The weight matrices are supplied for you, and have been trained on Plato's dialogues. At the end, you will have a transformer that babbles fake Plato.

The transformer you will build is smaller than a commercial LLM (about 10% the size of GPT-2), so that it can run without special hardware. That means our transformer's attempts at Plato won't be very good. In fact, a lot of the output will be ungrammatical; some words will even be made up.

However, the same code would generate perfectly grammatical text on a more powerful computer, if you just changed a few of the numbers at the top of the file.

## Exercise 1

Download [the starter notebook](jpt.ipynb), [the weights](weights.npz), and [the text of Plato's dialogues](plato.txt). Put them in the same folder on your computer or JupyterHub. Open the Jupyter notebook and look over the code.

- The model's vocabulary is the English alphabet, plus digits and a few punctuation symbols.
- The embedding space has 384 dimensions. There are 6 "blocks", in other words 6 attention layers each followed by an MLP layer. Each attention layer has 6 heads.
- The `generate()` function takes a prompt as input and uses the `transformer()` function to output 500 new tokens.

Run the code. For now it just repeats the same letter 500 times.

## Exercise 2

Fill in the `transformer()` function. Following the flow of the transformer diagram in the chapter (the "Corrections" section), the steps are:

- Call `tokenize()` on `prompt` and store the result as `tokens`.
- Call `embed()` on `tokens` and store the result as `X`.
- Create a `for` loop with `n_block` iterations. Each iteration `i` should do the following steps:
    - Call `attention(X, i)`, add the result to `X`, and store that sum as the new value of `X`.
    - Call `mlp(X, i)`, add the result to `X`, and store that sum as the new value of `X`.
- Finally, call `unembed()` on `X`, and return the result.

## Exercise 3

The `tokenize()` function needs to return a list of integers, one for each character in the `prompt`. For example, if the `prompt` is `Hello`, then `tokenize()` should return `[30, 55, 62, 62, 65]` because `H` is the 30th character in the vocabulary, `e` is 55th, etc.

Complete the `tokenize()` function so that it does this. Some tips:

> We use square brackets because `char_to_int` isn't a function, it's a different sort of Python object known as a `dict` (dictionary).

- I've provided `char_to_int` so you can find the integer corresponding to a given character. For example, `char_to_int['H']` returns `30`, and `char_to_int['e']` returns `55`. Note the square brackets here instead of parentheses.
- You can create an empty list called `foo` with `foo = []`.
- You can loop through the characters in a string by writing: `for c in my_string:`.
- You can append an element to the end of a list by writing: `foo.append(element)`.

## Exercise 4

Next, the `embed()` function should take a list of integers and return a matrix. Each column of that matrix should be the sum of the token embedding and position embedding for the corresponding integer in the input.

The easiest way to do this is to make two matrices and then add them together.

- The columns of the first matrix should have the token embeddings.
- The columns of the second matrix should have the position embeddings.

I've provided two matrices to help with this:

- `tok_emb` is a `384 x 77` matrix. Column `i` is the token embedding for the `i`th character in the vocabulary.
- `pos_emb` is a `384 x 256` matrix. Column `i` is the position embedding for the `i`th position in the prompt.

Tip: you can select the columns you want from a matrix using a list of indices, like this:

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
indices = [0, 2, 2, 1]
selection = matrix[:, indices]
print(selection)
# [[1 3 3 2]
#  [4 6 6 5]]
```

## Exercise 5

Fill in the `softmax()` function. This function should take a vector `x` as input and return a vector of probabilities.

Tips:

- The function `np.exp()` calculates $e^x$. You can pass it a vector and it will calculate $e^x$ for each element.
- The function `np.sum()` adds together a vector's elements.

## Exercise 6

This exercise introduces a technique that will greatly simplify the work below. In several places, we will need to perform the same linear operation on each column of some matrix `X`. That is, we'll have a matrix `W` and vector `b`, and we will need to calculate `W @ x + b`, for each column `x` of the matrix `X`. We could do this with a `for` loop, processing the columns of `X` one at a time. But there's a much simpler way.

**(a)** If we multiply `W @ X`, what will column `i` contain?

**(b)** Mathematically, you can't add a matrix and a vector. For example, the following operation is not defined:

$$
\begin{bmatrix}
  1 & 2\\
  3 & 4
\end{bmatrix} +
\begin{bmatrix}
  5\\
  6
\end{bmatrix}
$$

> NumPy's approach here is called *broadcasting*.

But NumPy won't give an error if you ask it to do this. Try it out and examine the result. What does NumPy do to make this operation work?

**(c)** Based on the above, what's a simple way to calculate `W @ x + b` for each column `x` of `X`, without having to loop over the columns? Hint: you only need one line of very simple code (in fact it's just five characters of code, not counting whitespace).

## Exercise 7

Implement the attention heads by filling in the `head()` function. This is the trickiest part, so let's step through it in three parts.

**(a)** Each column of `X` is an embedding vector. Multiply each embedding by `W_Q` to get its query vector, and store these query vectors as the columns of a new matrix `Q`.

Likewise, use `W_K` and `W_V` to create matrices `K` and `V`, whose columns are (respectively) the key and value vectors.

**(b)** Create a matrix called `scores`. The value at row `i`, column `j` should be the dot-product of column `i` of `Q` and column `j` of `K`, divided by `scale`. (Hint: transpose `Q`.)

I've provided some code to handle the remaining processing that needs to be done on `scores`, since this part is kind of fiddly. First, this code makes sure each token only averages information from previous ones, since the prediction of the next token should only draw on preceding tokens. Second, it applies `softmax()` to convert `scores` to proper weights, storing them in the matrix `A`.

**(c)** Each row of `A` contains a sequence of weights. Create a matrix where column `i` is a weighted average of the columns of `V`, using the weights from row `i` of `A`. Return that matrix.

## Exercise 8

Fill in the `attention()` function. There are two parts here.

**(a)** Pass the input `X` through each attention head, then concatenate the results. You can do this with a `for` loop that runs `n_head` times. Each time, call `head(X, i, h)`, where `i` is the current attention layer and `h` is the number of the attention head. Be sure to store the output from each head as it's processed. When you have all the outputs stored in a list called `head_outputs`, you can concatenate them together with `np.concatenate(head_outputs, axis=0)`.

**(b)** Apply the linear projection $W \mathbf{x} + \mathbf{b}$ to each column of the result from step (a), and return the result.

## Exercise 9

Next we'll fill in the `relu()` function. This function takes the role the sigmoid function played in the previous chapter.

The hidden layer of our our penguin-classification network applied the sigmoid function to each intermediate value before passing it on to the final, output layer. But here we'll use a different operation known as *ReLU* (short for *Rectified Linear Unit*).

> The ReLU function is often used in place of sigmoid in modern neural networks. It has a broadly similar shape to sigmoid, but is simpler and tends to work better in training.

The chapter's ReLU figure shows a graph of the ReLU function. The definition is:

$$
\text{ReLU}(x) =
  \begin{cases}
    0 & \text{ if } x \leq 0, \\
    x & \text{ if } x > 0.
  \end{cases}
$$

Fill in `relu()` so that it computes this function. It should take a matrix as input and apply ReLU to each entry. Tip: the function `np.maximum(a, X)` takes a number `a` and matrix `X` and returns the same matrix but with entries smaller than `a` replaced by `a`.

## Exercise 10

Fill in the `mlp()` function. The chapter's MLP diagram shows the basic structure of an MLP layer. Mathematically, the sequence of operations this network performs on an input vector $\mathbf{x}$ is:

$$
\begin{aligned}
\mathbf{y} &= \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) \\
\mathbf{z} &= W_2 \mathbf{y} + \mathbf{b}_2
\end{aligned}
$$

The `mlp()` function should take a matrix `X` as input, pass its columns through this sequence of operations, and return a matrix whose columns are the vectors that come out the other end.

## Exercise 11

Fill in the `unembed()` function. This function takes a matrix `X` as input, and it returns an integer: the new token to be output. There are three basic steps to this process:

**(a)** Compare the embedding of each token in the vocabulary to the last column of `X`, by taking dot-products.

**(b)** Convert these dot-products to probabilities using `softmax()`.

**(c)** Choose the next token according to those probabilities.

Tips:

- You can get the last column of `X` with `X[:, -1]`.
- If `probs` is a vector of `n` probabilities, then

  ```python
  np.random.choice(len(probs), p=probs)
  ```

  will choose a number between `0` and `n` (excluding `n`) according to those probabilities.

You should now have a working transformer. Test it by running the `generate()` function on a prompt of your choice.
