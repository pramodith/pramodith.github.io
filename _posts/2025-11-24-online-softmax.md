---
layout: post
title: "The Math Behind Online Softmax"
date: 2025-11-24 00:00:00 +0000
categories: [LLM, AI, Kernels, GPU, ML]
tags: [LLM, AI, Kernels, GPU, ML]
math: true  # Enable math equation rendering
author: Pramodith B
pin: true
description: "Understanding the mathematical principles behind online softmax, an optimization technique used in Flash Attention to efficiently compute softmax in chunks."
---

# Online Softmax
Flash Attention has been responsible for reducing the runtime of transformer models. Flash Attention can be broken down into two categories of optimizations:
1. GPU-aware I/O optimizations. These relate to how data moves between GPU high-bandwidth memory (HBM) and on-chip SRAM. These are not discussed here.
2. Online softmax. This is an algorithmic optimization that allows us to compute softmax in chunks. Each chunk is sized to fit in GPU SRAM and can be processed in parallel across multiple streaming multiprocessors (SMs).

In this blog post, we will discuss the online softmax algorithm and the simple mathematical tricks that make it possible.

## Softmax Formula

Given a vector of scores $z = [z_1, z_2, \ldots, z_n]$, the softmax function is defined as:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Since this blog focuses on the math behind online softmax, we will assume that we have one query vector and multiple key vectors. Thus, we focus on computing the softmax of a single row of the attention matrix. The same logic extends to multiple query vectors and batch sizes > 1.

Let's initialize a query vector and a random number of key vectors of a random hidden size to compute the attention scores and the softmax scores.

### Setup
```bash
!pip3 install --quiet torch
```
```python
import torch
from torch import nn
```

```python   
hidden_dim = 2 ** torch.randint(2, 12, (1, ))
num_keys = 2 ** torch.randint(2, 8, (1, ))
query = torch.randn(1, hidden_dim)
keys = torch.randn(num_keys, hidden_dim)

dot_products = torch.matmul(query, keys.T)
softmax_scores = nn.functional.softmax(dot_products, dim=-1)
print(f"Softmax scores: {softmax_scores}")
```

## Softmax: Subtracting the Max
To avoid overflow, implementations of softmax often subtract the maximum value from all the dot products. Mathematically this operation is equivalent to the original softmax because of the following:

In the numerator and denominator, we can factor out a constant  $e^{-c}$ where 
$$
c = \max(dot\_products)
$$

Numerator:
$$
e^{dot\_product_i - c} = e^{dot\_product_i} \cdot e^{-c}
$$

Denominator:
$$
\sum_{j=1}^{n} e^{dot\_product_j - c} = \sum_{j=1}^{n} e^{dot\_product_j} \cdot e^{-c}
$$

Thus we have:
$$
\text{softmax}(dot\_product_i - c) = \frac{e^{dot\_product_i - c}}{\sum_{j=1}^{n} e^{dot\_product_j - c}} = 
\frac{e^{dot\_product_i} \cdot \cancel{e^{-c}}}{\sum_{j=1}^{n} e^{dot\_product_j} \cdot \cancel{e^{-c}}} = 
\frac{e^{dot\_product_i}}{\sum_{j=1}^{n} e^{dot\_product_j}} = \text{softmax}(dot\_product_i)
$$

**This shows us that subtracting by the max does not change the output of softmax.**

```python
maxs = torch.max(dot_products, dim=-1, keepdim=True)[0]
dot_products -= maxs
softmax_scores_post_max = torch.nn.functional.softmax(dot_products, -1)
```

Let's verify that these two softmax computations are equivalent:

```python
assert torch.allclose(softmax_scores, softmax_scores_post_max)
```

## Online Softmax
Great, it's time for us to move on to online softmax. To make things simple we'll break down the computation of online softmax into two parts:

1. Computing the numerator of softmax in an online fashion.
2. Computing the denominator of softmax in an online fashion.

## Online Softmax Numerator

Let's assume that we have limited memory and cannot compute all the dot products at once. Instead, we need to process them in the maximum number of chunks that fit in memory.


Hold on a minute: in the numerator of softmax we need to subtract the global maximum dot product from each dot product for a given query. If we only have access to a chunk of dot products at a time, how can we compute the global maximum?


We can compute the maximum in an online fashion by keeping track of the maximum value seen so far as we process each chunk. Let's assume that we have two chunks of dot products:


$$
\begin{align*}
\text{chunk}_1 &= [d_1, d_2, \ldots, d_k] \\
\text{chunk_max}_1 &= \max(\text{chunk}_1) \\
\text{chunk_1_numerator} &= \text{softmax}(\text{chunk}_1 - \text{chunk_max}_1)
\end{align*}
$$


$$
\begin{align*}
\text{chunk}_2 &= [d_{k+1}, d_{k+2}, \ldots, d_n] \\
\text{chunk_max}_2 &= \max(\text{chunk}_2) \\
\text{chunk_2_numerator} &= \text{softmax}(\text{chunk}_2 - \text{chunk_max}_2)
\end{align*}
$$


To compute the overall softmax, **we need to scale the chunk softmaxes based on the difference between the chunk maximums and the global maximum**. We get to this with a little bit of high-school math:


$$
\begin{align*}
s &= s_0 + s_1 \\
e^{s} &= e^{s_0 + s_1} = e^{s_0} \cdot e^{s_1}
\end{align*}
$$


The max can be re-written as:


$$
\text{global_max} = \max(\text{chunk_max}_1, \text{chunk_max}_2)
$$


The correct numerator for each chunk is:
$$
\mathrm{corrected\_chunk\_1\_numerator} = e^{\mathrm{dot\_{products\_chunk}}_1 - \mathrm{global\_{max}}}
$$


Let's add and subtract the term `chunk_max_1` to re-write the above as:

$$
e^{\mathrm{dot\_{products\_chunk}}_1 - \mathrm{global\_{max}}} = e^{\mathrm{dot\_{products\_chunk}}_1 - \mathrm{global\_{max}} \boldsymbol{- \mathrm{chunk\_{max}}_1 + \mathrm{chunk\_{max}}_1}}
$$


Moving terms around we get:

$$
e^{\mathrm{dot\_{products\_chunk}}_1 - \mathrm{global\_{max}}} = e^{\mathrm{dot\_{products\_chunk}}_1 - \mathrm{chunk\_{max}}_1} \cdot e^{\mathrm{chunk\_{max}}_1 - \mathrm{global\_{max}}}
$$

$$
correction\_factor\_chunk\_1 = e^{\mathrm{chunk\_{max}}_1 - \mathrm{global\_{max}}}
$$


Tada! If we need to compute the softmax scores in chunks all we need to do is keep track of all the maximums for each chunk and then compute the global maximum. We'll then use the above formula to adjust each chunk numerator by multiplying it with the appropriate correction factor.

## Online Softmax Denominator
Okay, now that we've worked out how to compute the numerator in an online fashion, let's look at the denominator.

The denominator of the softmax function is the sum of exponentials of all dot-products.

$$
\begin{align*}
\text{denominator} &= \sum_{j=1}^{n} e^{dot\_product_j - \text{global_max}}
\end{align*}
$$

If we split the dot-products into chunks, we can compute the denominator for each chunk separately and then sum them up:

$$
\begin{align*}
\text{denominator_chunk}_i &= \sum_{j \in \text{chunk}_i} e^{dot\_product_j - \text{global_max}} \\
\text{denominator} &= \sum_{i} \text{denominator_chunk}_i
\end{align*}
$$

Since each chunk doesn't know the global maximum we'll actually have:

$$
\text{denominator_chunk}_i = \sum_{j \in \text{chunk}_i} e^{dot\_product_j - \text{chunk_max}_i}
$$

The correct denominator for each chunk can be computed as:

$$
\text{corrected_denominator_chunk}_i = \sum_{j \in \text{chunk}_i} e^{\text{dot_product}_j - \text{global_max}} \\
$$

Using the same trick as we did for the numerator, we can add and subtract the term `chunk_max_i`:

$$
= \sum_{j \in \text{chunk}_i} e^{\text{dot_product}_j - \text{chunk_max}_i} \cdot e^{\text{chunk_max}_i - \text{global_max}}
$$

The correction factor is the same for all elements within a chunk, so we can factor it out of the sum:

$$
\text{corrected_denominator_chunk}_i = e^{\text{chunk_max}_i - \text{global_max}} \cdot \sum_{j \in \text{chunk}_i} e^{dot\_product_j - \text{chunk_max}_i}
$$

The overall denominator can then be computed by summing up the corrected denominators from each chunk:

$$
\text{denominator} = \sum_{i} \text{corrected_denominator_chunk}_i = \sum_{i} e^{\text{chunk_max}_i - \text{global_max}} \cdot \sum_{j \in \text{chunk}_i} e^{dot\_product_j - \text{chunk_max}_i}
$$

We can store the sum of exponentials for each chunk as we compute them, and then apply the correction factor based on the global maximum when we compute the final denominator.

## Code Implementation (Naive Version)
Alright, it's time to convert all this math into code! Lucky for us, all of this can be implemented in under 20 lines of PyTorch code. Let's get to it!

As a reminder, this implementation is a naive version where we don't worry about multiple queries or batch sizes greater than 1. The purpose of this code is to show how we can easily map all the math we've discussed into code.

```python
num_chunks = 2 ** torch.randint(2, min(num_keys, 5), (1, )).item()
print(f"Number of chunks: {num_chunks}")
key_chunks = keys.chunk(num_chunks, dim=0)
```

We'll divide the keys into `num_chunks` chunks. Now we can compute the dot-products, maximums and sum of exponentials for each chunk.

```python
maxs = []
dot_products_chunks = []
dot_products_sum = []
for k in key_chunks:
    # compute the dot-product against all keys in the chunk
    dot_products_chunks.append(query @ k.T)
    # store the max
    maxs.append(torch.max(dot_products_chunks[-1], dim=-1)[0])
    # subtract the dot-product with the max of the chunk
    dot_products_chunks[-1] -= maxs[-1]
    # store the sum of dot-product scores for the chunk
    dot_products_sum.append(torch.exp(dot_products_chunks[-1]).sum())
```

In the code above, you see that we:

1. For a given chunk we compute the dot-product scores.
2. Find the max for the chunk.
3. Subtract the dot-products with the max of the chunk.
4. Store the sum of all the dot-product scores within the given chunk.

1-3 is needed for the numerator and denominator since both of them require us to compute

$$
e^{dot\_product_j - \text{chunk_max}_i}
$$

4 is needed for the denominator to compute the term

$$
\sum_{j \in \text{chunk}_i} e^{dot\_product_j - \text{chunk_max}_i}
$$

Now that we've computed the maximum for each chunk we can compute the global maximum and our correction factor.
Remember, the correction factor is:

$$
\text{correction_factor} = e^{\mathrm{chunk\_{max}}_j - \mathrm{global\_{max}}}
$$

```python
global_max = torch.max(torch.cat(maxs), dim=-1, keepdim=True)[0]
correction_factor = [(local_max - global_max) for local_max in maxs]
```

We have everything we need to compute the denominator for the softmax function now!

$$
\text{denominator} = \sum_{i} \text{corrected_denominator_chunk}_i = \sum_{i} e^{\text{chunk_max}_i - \text{global_max}} \cdot \sum_{j \in \text{chunk}_i} e^{dot\_product_j - \text{chunk_max}_i}
$$

$$
\text{correction_factor} = \sum_{i} e^{\text{chunk_max}_i - \text{global_max}} \\
$$

$$
\text{dot_products_sum}_i = \sum_{j \in \text{chunk}_i} e^{\text{dot_product}_j - \text{chunk_max}_i}
$$

```python
denominator = sum([
    torch.exp(correction_factor[i]) * dot_products_sum[i]
    for i in range(len(dot_products_chunks))
])
```

We can now finally compute the correct softmax scores!
The numerator for each chunk is computed by multiplying the chunk numerator with the correction factor.

```python
online_softmax = [torch.exp(dp + cf)/denominator for dp, cf in zip(dot_products_chunks, correction_factor)]
```

Let's verify that our online softmax implementation matches the standard softmax implementation from PyTorch:

```python
assert torch.allclose(torch.cat(online_softmax, dim=-1), softmax_scores)
```

Woohoo! Our online softmax implementation matches the standard softmax implementation from PyTorch.

## Conclusion

In practice, the chunks of queries and keys can be processed in parallel across multiple streaming multiprocessors (SMs) on the GPU, allowing for efficient computation of the attention mechanism even with limited memory. If you want to take a look at what a production ready implementation looks like, check out the [liger-kernels](https://github.com/linkedin/Liger-Kernel/blob/0a62700752e8592c0bf16916d1e4dbf598cee8c1/src/liger_kernel/ops/softmax.py#L116) implementation.

If you'd like to understand the GPU-related I/O optimizations and see how online softmax is integrated into a full attention mechanism, check out Aleksa Gordic's great blog post [here](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad/).

The notebook containing this exact blog can be found [here](https://github.com/pramodith/llm_exploration/blob/main/transformer_from_scratch/online_softmax.ipynb).