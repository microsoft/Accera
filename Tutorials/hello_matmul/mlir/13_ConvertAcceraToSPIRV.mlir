module @hello_matmul {
  func @hello_matmul_py_impl_16252232176815793891(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
    %c128 = constant 128 : index
    %c0 = constant 0 : index
    %c256 = constant 256 : index
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    scf.for %arg3 = %c0 to %c128 step %c1 {
      scf.for %arg4 = %c0 to %c256 step %c1 {
        scf.for %arg5 = %c0 to %c256 step %c4 {
          %0 = load %arg0[%arg3, %arg5] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %1 = load %arg1[%arg5, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %2 = mulf %0, %1 {RelaxedPrecision} : f32
          %3 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %4 = addf %3, %2 {RelaxedPrecision} : f32
          store %4, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %5 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          store %5, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %6 = addi %arg5, %c1 : index
          %7 = load %arg0[%arg3, %6] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %8 = load %arg1[%6, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %9 = mulf %7, %8 {RelaxedPrecision} : f32
          %10 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %11 = addf %10, %9 {RelaxedPrecision} : f32
          store %11, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %12 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          store %12, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %13 = addi %arg5, %c2 : index
          %14 = load %arg0[%arg3, %13] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %15 = load %arg1[%13, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %16 = mulf %14, %15 {RelaxedPrecision} : f32
          %17 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %18 = addf %17, %16 {RelaxedPrecision} : f32
          store %18, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %19 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          store %19, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %20 = addi %arg5, %c3 : index
          %21 = load %arg0[%arg3, %20] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %22 = load %arg1[%20, %arg4] : memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %23 = mulf %21, %22 {RelaxedPrecision} : f32
          %24 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %25 = addf %24, %23 {RelaxedPrecision} : f32
          store %25, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          %26 = load %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
          store %26, %arg2[%arg3, %arg4] : memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>
        }
      }
    }
    return
  }
  func @hello_matmul_py(%arg0: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg1: memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, %arg2: memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) attributes {exec_target = 0 : i64, accv.emit_header_decl, accv.emit_raw_pointer_api} {
    call @hello_matmul_py_impl_16252232176815793891(%arg0, %arg1, %arg2) : (memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<256x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>, memref<128x256xf32, affine_map<(d0, d1) -> (d0 * 256 + d1)>>) -> ()
    return
  }
}
