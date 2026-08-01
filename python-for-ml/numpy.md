# NumPy

## What is it?

NumPy is the foundational Python library for numerical computing. Its core object is the **ndarray** — a multi-dimensional array where every element shares the same fixed type and sits in a single contiguous block of memory. Nearly every code example throughout this handbook so far — `LinearRegression` in [What is Machine Learning](../machine-learning/what-is-machine-learning.md), `LogisticRegression` and `KMeans` in [Types of Machine Learning](../machine-learning/types-of-machine-learning.md) — has already been passing NumPy arrays into scikit-learn without this chapter naming what they were.

Think of a NumPy array as a C array wearing a Python-friendly interface — one block of memory, one type, tightly packed. A Python list, by contrast, is really an array of pointers to separately allocated Python objects scattered across memory, where every single element carries its own type information and reference count. That difference in memory layout is the entire reason NumPy exists.

## Why does it exist?

Plain Python lists are flexible but slow for numeric work. Looping over a list in Python means the interpreter re-checks each element's type and re-dispatches the operation one element at a time — overhead that has nothing to do with the actual math being done. NumPy exists to remove that overhead: it pushes the loop down into pre-compiled, typed C code operating on contiguous memory, instead of looping in the Python interpreter itself.

**When NumPy is worth reaching for, and when a plain list is fine:** for small collections, heterogeneous data, or work that isn't fundamentally numeric, a Python list is simpler and there's no real cost to using one. The moment you're doing math over a collection of numbers at any real scale, a vectorized NumPy operation is both faster and more concise than the equivalent Python loop — verified directly below, not just asserted.

## How does it work?

Two ideas do most of the work:

- **Vectorization** — an operation like `arr * arr` or `arr.sum()` applies across the entire array in one call, executed in compiled C, rather than looping over elements in Python.
- **Broadcasting** — rules that let arrays of different shapes combine directly (like adding a single row to every row of a matrix) without writing an explicit loop or manually duplicating data.

## Example

**Vectorization, measured directly.** Summing the squares of 1,000,000 numbers, both ways:

```python
import numpy as np
import timeit

n = 1_000_000
data = list(range(n))
arr = np.arange(n)

def python_loop():
    total = 0
    for x in data:
        total += x * x
    return total

def numpy_vectorized():
    return (arr * arr).sum()

t_python = timeit.timeit(python_loop, number=5) / 5
t_numpy = timeit.timeit(numpy_vectorized, number=5) / 5
print(f"python loop: {t_python*1000:.2f} ms")
print(f"numpy vectorized: {t_numpy*1000:.2f} ms")
```

```text
python loop: 68.93 ms
numpy vectorized: 2.67 ms
```

Both compute the identical number; the vectorized version ran about 26x faster on this machine, doing the same amount of work with no Python-level loop at all.

**Broadcasting**, adding a single row to every row of a matrix:

```python
import numpy as np

matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
row = np.array([100, 200, 300, 400])

print(matrix + row)
```

```text
[[101 202 303 404]
 [105 206 307 408]
 [109 210 311 412]]
```

No loop was written — NumPy applied `row` to every row of `matrix` automatically, because the shapes are compatible (a `(3, 4)` array and a `(4,)` array).

## Where is it used?

Directly underneath nearly every numeric example in this handbook: the [What is Machine Learning](../machine-learning/what-is-machine-learning.md) and [Types of Machine Learning](../machine-learning/types-of-machine-learning.md) examples all take NumPy arrays as input. It's also the foundation pandas is built on top of, and the standard input format for scikit-learn, PyTorch, and most of the Python data ecosystem.

## Advantages

- **Dramatically faster than pure Python for numeric work**, as measured directly above.
- **Broadcasting removes a large class of manual looping code**, making numeric operations shorter and less error-prone than the equivalent hand-written loop.
- **The de facto standard array interface** most of the Python data and ML ecosystem is built on — learning it transfers directly to pandas, scikit-learn, and beyond.

## Limitations

- **Every element shares one fixed type.** That's exactly what makes NumPy fast, but it means mixed-type data (some numeric columns, some text) doesn't fit naturally the way a Python list or a pandas DataFrame handles it.
- **Not naturally suited to labeled, tabular data.** A NumPy array has no concept of named columns or row labels — that's specifically what pandas adds on top of it, covered in the next chapter.
- **Single-core by default.** Vectorized operations are fast per-call, but still run on one CPU core; genuinely large workloads eventually need a distributed or GPU-backed alternative once a single machine's memory or a single core's throughput becomes the bottleneck.

## Production considerations

- **Copies vs. views affect memory at scale.** Some operations (basic slicing) return a view into the same memory; others (fancy indexing, many reshapes) return a full copy. On large arrays, an unexpected copy can silently double memory usage.
- **dtype choice affects both memory and correctness.** Defaulting every array to `float64` doubles memory use for large datasets where `float32` would have been accurate enough; conversely, choosing too small an integer type can silently overflow instead of raising an error.
- **A single machine's CPU and memory are the real ceiling.** Vectorization makes each operation fast, but workloads that outgrow one machine typically move to a distributed array library (e.g. Dask) or GPU-backed arrays rather than trying to out-optimize single-core NumPy further.

## Common mistakes

- **Looping over a NumPy array element by element in Python.** This throws away essentially all of NumPy's performance benefit, since the loop is back in the Python interpreter — exactly the cost the vectorized example above shows NumPy exists to avoid.
- **Assuming two differently-shaped arrays will combine without checking broadcasting compatibility.** Trying to add a `(3, 4)` matrix to a `(3,)` array (instead of the compatible `(4,)` shape used above) raises `ValueError: operands could not be broadcast together with shapes (3,4) (3,)` — a clear failure, but only after being confused about why it happened.
- **Not knowing whether an operation returns a view or a copy.** Basic slicing (`arr[1:4]`) returns a view — modifying it modifies the original array. Fancy indexing (`arr[[1, 2, 3]]`) returns a copy — modifying it leaves the original untouched. Assuming the wrong one is a classic source of a bug where "changing a copy" quietly changes the original, or vice versa.

## Interview questions

### Basic

- What is the core difference between a NumPy array and a Python list?
- What does "vectorization" mean in the context of NumPy?

### Intermediate

- What is broadcasting, and what determines whether two arrays are broadcast-compatible?
- Why can a NumPy array use significantly less memory than a Python list holding the same numbers?

### Advanced

- Why does looping over a NumPy array in pure Python erase most of its performance advantage?
- What's the difference between a NumPy view and a copy, and why does that distinction matter for correctness in real code?
