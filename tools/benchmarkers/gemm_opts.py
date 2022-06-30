####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

from dataclasses import dataclass

CONFIG_HEADERS = ["m", "n", "k", "transA", "transB", "alpha", "beta", "lda", "ldb", "ldc"]

@dataclass(eq=True, unsafe_hash=True)
class GemmOpts:
    m: int = 0
    n: int = 0
    k: int = 0
    transA: bool = False
    transB: bool = False
    alpha: float = 1.
    beta: float = 0.
    lda: int = 0
    ldb: int = 0
    ldc: int = 0

    def __post_init__(self):
        self.m = int(self.m)
        self.n = int(self.n)
        self.k = int(self.k)
        self.transA = bool(int(self.transA))
        self.transB = bool(int(self.transB))
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.lda = int(self.lda)
        self.ldb = int(self.ldb)
        self.ldc = int(self.ldc)

    def __str__(self):
        return f'{self.m}, {self.n}, {self.k}, {self.transA}, {self.transB}, {self.alpha}, {self.beta}, {self.lda}, {self.ldb}, {self.ldc}'