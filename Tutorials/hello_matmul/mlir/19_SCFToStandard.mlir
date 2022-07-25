module @hello_matmul {
  func @hello_matmul_py_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
    %c128 = constant 128 : index
    %c0 = constant 0 : index
    %c256 = constant 256 : index
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
    %1 = cmpi "slt", %0, %c128 : index
    cond_br %1, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
    %3 = cmpi "slt", %2, %c256 : index
    cond_br %3, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    br ^bb5(%c0 : index)
  ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
    %5 = cmpi "slt", %4, %c256 : index
    cond_br %5, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %6 = load %arg0[%0, %4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %7 = load %arg1[%4, %2] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %8 = mulf %6, %7 {RelaxedPrecision} : f32
    %9 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %10 = addf %9, %8 {RelaxedPrecision} : f32
    store %10, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %11 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    store %11, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %12 = addi %4, %c1 : index
    %13 = load %arg0[%0, %12] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %14 = load %arg1[%12, %2] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %15 = mulf %13, %14 {RelaxedPrecision} : f32
    %16 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %17 = addf %16, %15 {RelaxedPrecision} : f32
    store %17, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %18 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    store %18, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %19 = addi %4, %c2 : index
    %20 = load %arg0[%0, %19] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %21 = load %arg1[%19, %2] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %22 = mulf %20, %21 {RelaxedPrecision} : f32
    %23 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %24 = addf %23, %22 {RelaxedPrecision} : f32
    store %24, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %25 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    store %25, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %26 = addi %4, %c3 : index
    %27 = load %arg0[%0, %26] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %28 = load %arg1[%26, %2] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %29 = mulf %27, %28 {RelaxedPrecision} : f32
    %30 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %31 = addf %30, %29 {RelaxedPrecision} : f32
    store %31, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %32 = load %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    store %32, %arg2[%0, %2] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
    %33 = addi %4, %c4 : index
    br ^bb5(%33 : index)
  ^bb7:  // pred: ^bb5
    %34 = addi %2, %c1 : index
    br ^bb3(%34 : index)
  ^bb8:  // pred: ^bb3
    %35 = addi %0, %c1 : index
    br ^bb1(%35 : index)
  ^bb9:  // pred: ^bb1
    return
  }
  func @hello_matmul_py(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.emit_header_decl, accv.emit_raw_pointer_api} {
    call @hello_matmul_py_impl_16252232176815793891(%arg0, %arg1, %arg2) : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
    return
  }
}
