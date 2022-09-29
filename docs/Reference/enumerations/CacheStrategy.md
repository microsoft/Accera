[//]: # (Project: Accera)
[//]: # (Version: v1.2.10)

# Accera v1.2.10 Reference
## `accera.CacheStrategy`

type | description
--- | ---
`accera.CacheStrategy.BLOCKED` | Every thread copies a contiguous block of memory based on their thread index. e.g. If 100 elements are cached by 10 threads, thread 0 copies elements [0, 10), thread 1 copies elements [10, 20) and so on.
`accera.CacheStrategy.STRIPED` | Every thread copies a part of their contribution in a round-robin fashion. e.g. In the previous example, thread 0 will now copy elements [0, 2), [20, 22), [40, 42), [60, 62) and [80, 82), thread 1 will copy [2, 4), [22, 24), [42, 44), [62, 64) and [82, 84) and so on. The minimum number of contiguous elements that each thread copies is governed by the vectorization parameter, which in this example is 2.

The effects of different caching strategies can be noticed as performance characteristics arising out of overhead caused by bank conflicts, memory coalescing etc.

<div style="page-break-after: always;"></div>
