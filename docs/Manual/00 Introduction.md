[//]: # (Project: Accera)
[//]: # (Version: 1.2)

# Introduction
Accera is a programming model, a domain-specific programming language embedded in Python (eDSL), and an optimizing cross-compiler for compute-intensive code. Accera currently supports CPU and GPU targets and focuses on optimization of nested for-loops.

Writing highly optimized compute-intensive code in a traditional programming language is a difficult and time-consuming process. It requires special engineering skills, such as fluency in Assembly language and a deep understanding of computer architecture. Manually optimizing the simplest numerical algorithms already requires a significant engineering effort. Moreover, highly optimized numerical code is prone to bugs, is often hard to read and maintain, and needs to be reimplemented every time a new target architecture is introduced. Accera aims to solve these problems.

Accera has three goals:

* **Performance**: generate the fastest implementation of any compute-intensive algorithm.
* **Readability**: do so without sacrificing code readability and maintainability.
* **Writability**: a user-friendly programming model, designed for agility.

The Accera language was designed with the following guiding principles in mind:

### 1: Strict separation of logic from implementation
Traditional programming languages tend to tightly couple the code logic (*what* the program does) with its implementation (*how* the program is implemented). For example, consider the simple example of multiplying a 16&times;11 matrix *A* by a 11&times;10 matrix *B*. The logic of the algorithm is to calculate, for each value of *i* and *j*, the sum over *k* of *A[i,k]&middot;B[k,j]*. In Python, this logic can be expressed as
```python
# C += A @ B
for i in range(16):
    for j in range(10):
        for k in range(11):
            C[i, j] += A[i, k] * B[k, j]
```
However, the code above expresses more than just the logic of matrix multiplication, it also specifies a concrete plan for executing this logic: first perform all the work required to calculate `C(0,0)` in ascending order of `k`; then proceed to `C(0,1)`; etc. In principle, the iterations of this loop could be performed in any order and the logic would remain intact, but the code above insists on one specific order. Moreover, this code doesn't take advantage of important optimization techniques, such as double-buffered caching or vectorization.

In contrast, the Accera programming model draws a strict distinction between the logic and its implementation. Namely, the programmer first writes the logic using a pseudocode-like syntax, independent of the target platform and without any consideration for performance. After the abstract logic is specified, the programmer moves on to define the concrete implementation details.

### 2: Mindfully trade-off safety versus expressivity
The Accera programming model starts with a default implementation of the specified logic and allows the programmer to transform and manipulate that implementation in different ways. When used correctly, these transformations should be *safe*, which means that they do not influence the underlying logic. Using safe transformations allows the programmer to focus on performance, without having to worry about correctness. Moreover, safe transformations allow automatic search algorithms to search the space of transformations more aggressively, converge faster, and find better optima.

However, safety is usually achieved by restricting and constraining a programming language. Excessive restrictions can limit the expressivity and the power of a language and prevent its users from creating highly-sophisticated and highly-optimized implementations.

The Accera programming model navigates the trade-off between safety and expressivity by being very explicit about the safety guarantees provided by each transformation under different circumstances. Some situations are safer than others, but in all cases, the programmer knows exactly what safety guarantees are given.

### 3: The programmer is in control
The Accera language gives the programmer maximal control over the generated code and avoids under-the-hood magic that cannot be overridden or controlled. Convenience methods and carefully chosen default values prevent verbosity, but the programmer can always override these and fine-tune the implementation as they see fit.


<div style="page-break-after: always;"></div>
