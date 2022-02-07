// RUN: acc-opt --show-dialects | FileCheck %s
// CHECK: Registered Dialects:
// CHECK: accera
// CHECK-NEXT: accln
// CHECK-NEXT: accv
// CHECK-NEXT: accxp
// CHECK-NEXT: affine
// CHECK-NEXT: gpu
// CHECK-NEXT: linalg
// CHECK-NEXT: llvm
// CHECK-NEXT: math
// CHECK-NEXT: memref
// CHECK-NEXT: nvvm
// CHECK-NEXT: rocdl
// CHECK-NEXT: scf
// CHECK-NEXT: spv
// CHECK-NEXT: std
// CHECK-NEXT: tensor
// CHECK-NEXT: vector
