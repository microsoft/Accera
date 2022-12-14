[//]: # (Project: Accera)
[//]: # (Version: v1.2.13)

# Introduction
Accera is a framework with a Python-based Domain-specific Language (eDSL) that produces optimized compute-intensive code. Accera's primary focus is the optimization of affine and semi-affine nested for-loops for CPU and GPU targets.

Optimization of compute-intensive code in a traditional programming language is not only challenging and time-consuming, but manual optimization of the simplest numerical algorithms demands significant engineering effort and requires an advanced understanding of computer architecture and fluency in C++, C, or Assembly Language. Even with all these efforts, implemented code is prone to critical bugs and requires extensive engineering effort for maintenance. Accera aims at resolving all these issues by providing optimized solutions for compute-intensive algorithms that are highly efficient, readable, and maintainable. 

Accera has THREE primary goals:

* **Performance**: To generate the fastest implementation for any compute-intensive algorithm.
* **Readability**: To ensure effective implementation of algorithms without sacrificing the readability of code.
* **Writability**: To provide a user-friendly programming model designed for agility and maintainability.

Accera is designed based on the following guiding principles: 

### 1: Strict separation of logic from implementation
Traditional programming languages are prone to the tight coupling of code logic (*what* the program does) with its implementation (*how* the program is implemented). Consider an example of multiplying a 16&times;11 matrix *A* by an 11&times;10 matrix *B*. The algorithm's logic calculates the sum over *k* of *A[i,k]&middot;B[k,j]* for each value of *i* and *j*. In Python, this logic can be expressed as:
```python
# C += A @ B
for i in range(16):
    for j in range(10):
        for k in range(11):
            C[i, j] += A[i, k] * B[k, j]
```
The above code expresses more than just the logic of matrix multiplication. It insists on a specific execution flow: first perform all the steps required to calculate `C(0,0)` in ascending order of `k`; then proceed to `C(0,1)`. However, in principle, a single order of execution should not be imposed because the iterations of this loop can be performed in any order while keeping the logic intact. Moreover, the above logic doesn't utilize important optimization techniques, such as double-buffered caching or vectorization.

Accera, on the other hand, provides a strict distinction between logic and its implementation. The programmer first implements the logic without performance considerations using a pseudocode-like syntax independent of the target platform. Once the logic is specified, only then does the programmer move to define the concrete implementation details. 

### 2: Mindfully trade-off safety versus expressivity
Accera offers a programming model where a default implementation of the specified logic can be transformed and manipulated in different ways. If used correctly, these transformations are *safe*, which means that the underlying logic remains intact. This allows the programmer to entirely focus on the performance of the logic without worrying about its correctness. Moreover, these safe transformations allow automatic search algorithms to aggressively search the space of transformations to converge faster and find better optima. 

Traditionally, this safety is achieved by trading off the true potential of a programming language since it demands restricting its scope. Nevertheless, extensive constraints significantly restrict the expressivity and the power of the programming language, eventually preventing the end-users from developing highly-optimized and sophisticated implementations. 

Accera moderates this trade-off between safety and expressivity by explicitly defining what level of safety guarantees are being given by each transformation under different circumstances. Some situations are safer than others. However, the programmer knows exactly what safeties are being guaranteed in all cases. 

### 3: The programmer is in control
Accera gives the programmer maximum control over the generated logic by providing access to the underlying knobs that determine how algorithms are optimized. Convenience methods and carefully used default values can prevent verbosity. As per the use case, these helper methods can always be tuned, even overridden. 

<div style="page-break-after: always;"></div>

