module @optimized_matmul {
  func @optimized_matmul_py_impl_17630232307017152746(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
    %c781 = constant 781 : index
    %c782 = constant 782 : index
    %c783 = constant 783 : index
    %c512 = constant 512 : index
    %c780 = constant 780 : index
    %c256 = constant 256 : index
    %c16 = constant 16 : index
    %c128 = constant 128 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %c5 = constant 5 : index
    %c6 = constant 6 : index
    %c7 = constant 7 : index
    %c8 = constant 8 : index
    %c9 = constant 9 : index
    %c10 = constant 10 : index
    %c11 = constant 11 : index
    %c12 = constant 12 : index
    %c13 = constant 13 : index
    %c14 = constant 14 : index
    %c15 = constant 15 : index
    scf.for %arg3 = %c0 to %c512 step %c256 {
      scf.for %arg4 = %c0 to %c780 step %c6 {
        scf.for %arg5 = %c0 to %c256 step %c16 {
          scf.for %arg6 = %c0 to %c128 step %c4 {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %0 = addi %arg3, %arg5 : index
              %1 = addi %arg6, %arg7 : index
              %2 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %3 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %4 = mulf %2, %3 {RelaxedPrecision} : f32
              %5 = load %arg2[%arg4, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %6 = addf %5, %4 {RelaxedPrecision} : f32
              store %6, %arg2[%arg4, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %7 = load %arg2[%arg4, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %7, %arg2[%arg4, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %8 = addi %0, %c1 : index
              %9 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %10 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %11 = mulf %9, %10 {RelaxedPrecision} : f32
              %12 = load %arg2[%arg4, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %13 = addf %12, %11 {RelaxedPrecision} : f32
              store %13, %arg2[%arg4, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %14 = load %arg2[%arg4, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %14, %arg2[%arg4, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %15 = addi %0, %c2 : index
              %16 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %17 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %18 = mulf %16, %17 {RelaxedPrecision} : f32
              %19 = load %arg2[%arg4, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %20 = addf %19, %18 {RelaxedPrecision} : f32
              store %20, %arg2[%arg4, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %21 = load %arg2[%arg4, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %21, %arg2[%arg4, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %22 = addi %0, %c3 : index
              %23 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %24 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %25 = mulf %23, %24 {RelaxedPrecision} : f32
              %26 = load %arg2[%arg4, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %27 = addf %26, %25 {RelaxedPrecision} : f32
              store %27, %arg2[%arg4, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %28 = load %arg2[%arg4, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %28, %arg2[%arg4, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %29 = addi %0, %c4 : index
              %30 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %31 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %32 = mulf %30, %31 {RelaxedPrecision} : f32
              %33 = load %arg2[%arg4, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %34 = addf %33, %32 {RelaxedPrecision} : f32
              store %34, %arg2[%arg4, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %35 = load %arg2[%arg4, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %35, %arg2[%arg4, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %36 = addi %0, %c5 : index
              %37 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %38 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %39 = mulf %37, %38 {RelaxedPrecision} : f32
              %40 = load %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %41 = addf %40, %39 {RelaxedPrecision} : f32
              store %41, %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %42 = load %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %42, %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %43 = addi %0, %c6 : index
              %44 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %45 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %46 = mulf %44, %45 {RelaxedPrecision} : f32
              %47 = load %arg2[%arg4, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %48 = addf %47, %46 {RelaxedPrecision} : f32
              store %48, %arg2[%arg4, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %49 = load %arg2[%arg4, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %49, %arg2[%arg4, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %50 = addi %0, %c7 : index
              %51 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %52 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %53 = mulf %51, %52 {RelaxedPrecision} : f32
              %54 = load %arg2[%arg4, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %55 = addf %54, %53 {RelaxedPrecision} : f32
              store %55, %arg2[%arg4, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %56 = load %arg2[%arg4, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %56, %arg2[%arg4, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %57 = addi %0, %c8 : index
              %58 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %59 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %60 = mulf %58, %59 {RelaxedPrecision} : f32
              %61 = load %arg2[%arg4, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %62 = addf %61, %60 {RelaxedPrecision} : f32
              store %62, %arg2[%arg4, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %63 = load %arg2[%arg4, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %63, %arg2[%arg4, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %64 = addi %0, %c9 : index
              %65 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %66 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %67 = mulf %65, %66 {RelaxedPrecision} : f32
              %68 = load %arg2[%arg4, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %69 = addf %68, %67 {RelaxedPrecision} : f32
              store %69, %arg2[%arg4, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %70 = load %arg2[%arg4, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %70, %arg2[%arg4, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %71 = addi %0, %c10 : index
              %72 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %73 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %74 = mulf %72, %73 {RelaxedPrecision} : f32
              %75 = load %arg2[%arg4, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %76 = addf %75, %74 {RelaxedPrecision} : f32
              store %76, %arg2[%arg4, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %77 = load %arg2[%arg4, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %77, %arg2[%arg4, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %78 = addi %0, %c11 : index
              %79 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %80 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %81 = mulf %79, %80 {RelaxedPrecision} : f32
              %82 = load %arg2[%arg4, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %83 = addf %82, %81 {RelaxedPrecision} : f32
              store %83, %arg2[%arg4, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %84 = load %arg2[%arg4, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %84, %arg2[%arg4, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %85 = addi %0, %c12 : index
              %86 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %87 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %88 = mulf %86, %87 {RelaxedPrecision} : f32
              %89 = load %arg2[%arg4, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %90 = addf %89, %88 {RelaxedPrecision} : f32
              store %90, %arg2[%arg4, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %91 = load %arg2[%arg4, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %91, %arg2[%arg4, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %92 = addi %0, %c13 : index
              %93 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %94 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %95 = mulf %93, %94 {RelaxedPrecision} : f32
              %96 = load %arg2[%arg4, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %97 = addf %96, %95 {RelaxedPrecision} : f32
              store %97, %arg2[%arg4, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %98 = load %arg2[%arg4, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %98, %arg2[%arg4, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %99 = addi %0, %c14 : index
              %100 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %101 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %102 = mulf %100, %101 {RelaxedPrecision} : f32
              %103 = load %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %104 = addf %103, %102 {RelaxedPrecision} : f32
              store %104, %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %105 = load %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %105, %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %106 = addi %0, %c15 : index
              %107 = load %arg0[%arg4, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %108 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %109 = mulf %107, %108 {RelaxedPrecision} : f32
              %110 = load %arg2[%arg4, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %111 = addf %110, %109 {RelaxedPrecision} : f32
              store %111, %arg2[%arg4, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %112 = load %arg2[%arg4, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %112, %arg2[%arg4, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %113 = addi %arg4, %c1 : index
              %114 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %115 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %116 = mulf %114, %115 {RelaxedPrecision} : f32
              %117 = load %arg2[%113, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %118 = addf %117, %116 {RelaxedPrecision} : f32
              store %118, %arg2[%113, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %119 = load %arg2[%113, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %119, %arg2[%113, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %120 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %121 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %122 = mulf %120, %121 {RelaxedPrecision} : f32
              %123 = load %arg2[%113, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %124 = addf %123, %122 {RelaxedPrecision} : f32
              store %124, %arg2[%113, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %125 = load %arg2[%113, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %125, %arg2[%113, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %126 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %127 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %128 = mulf %126, %127 {RelaxedPrecision} : f32
              %129 = load %arg2[%113, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %130 = addf %129, %128 {RelaxedPrecision} : f32
              store %130, %arg2[%113, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %131 = load %arg2[%113, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %131, %arg2[%113, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %132 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %133 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %134 = mulf %132, %133 {RelaxedPrecision} : f32
              %135 = load %arg2[%113, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %136 = addf %135, %134 {RelaxedPrecision} : f32
              store %136, %arg2[%113, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %137 = load %arg2[%113, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %137, %arg2[%113, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %138 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %139 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %140 = mulf %138, %139 {RelaxedPrecision} : f32
              %141 = load %arg2[%113, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %142 = addf %141, %140 {RelaxedPrecision} : f32
              store %142, %arg2[%113, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %143 = load %arg2[%113, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %143, %arg2[%113, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %144 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %145 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %146 = mulf %144, %145 {RelaxedPrecision} : f32
              %147 = load %arg2[%113, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %148 = addf %147, %146 {RelaxedPrecision} : f32
              store %148, %arg2[%113, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %149 = load %arg2[%113, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %149, %arg2[%113, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %150 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %151 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %152 = mulf %150, %151 {RelaxedPrecision} : f32
              %153 = load %arg2[%113, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %154 = addf %153, %152 {RelaxedPrecision} : f32
              store %154, %arg2[%113, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %155 = load %arg2[%113, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %155, %arg2[%113, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %156 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %157 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %158 = mulf %156, %157 {RelaxedPrecision} : f32
              %159 = load %arg2[%113, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %160 = addf %159, %158 {RelaxedPrecision} : f32
              store %160, %arg2[%113, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %161 = load %arg2[%113, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %161, %arg2[%113, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %162 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %163 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %164 = mulf %162, %163 {RelaxedPrecision} : f32
              %165 = load %arg2[%113, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %166 = addf %165, %164 {RelaxedPrecision} : f32
              store %166, %arg2[%113, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %167 = load %arg2[%113, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %167, %arg2[%113, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %168 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %169 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %170 = mulf %168, %169 {RelaxedPrecision} : f32
              %171 = load %arg2[%113, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %172 = addf %171, %170 {RelaxedPrecision} : f32
              store %172, %arg2[%113, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %173 = load %arg2[%113, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %173, %arg2[%113, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %174 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %175 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %176 = mulf %174, %175 {RelaxedPrecision} : f32
              %177 = load %arg2[%113, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %178 = addf %177, %176 {RelaxedPrecision} : f32
              store %178, %arg2[%113, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %179 = load %arg2[%113, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %179, %arg2[%113, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %180 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %181 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %182 = mulf %180, %181 {RelaxedPrecision} : f32
              %183 = load %arg2[%113, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %184 = addf %183, %182 {RelaxedPrecision} : f32
              store %184, %arg2[%113, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %185 = load %arg2[%113, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %185, %arg2[%113, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %186 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %187 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %188 = mulf %186, %187 {RelaxedPrecision} : f32
              %189 = load %arg2[%113, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %190 = addf %189, %188 {RelaxedPrecision} : f32
              store %190, %arg2[%113, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %191 = load %arg2[%113, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %191, %arg2[%113, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %192 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %193 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %194 = mulf %192, %193 {RelaxedPrecision} : f32
              %195 = load %arg2[%113, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %196 = addf %195, %194 {RelaxedPrecision} : f32
              store %196, %arg2[%113, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %197 = load %arg2[%113, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %197, %arg2[%113, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %198 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %199 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %200 = mulf %198, %199 {RelaxedPrecision} : f32
              %201 = load %arg2[%113, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %202 = addf %201, %200 {RelaxedPrecision} : f32
              store %202, %arg2[%113, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %203 = load %arg2[%113, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %203, %arg2[%113, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %204 = load %arg0[%113, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %205 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %206 = mulf %204, %205 {RelaxedPrecision} : f32
              %207 = load %arg2[%113, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %208 = addf %207, %206 {RelaxedPrecision} : f32
              store %208, %arg2[%113, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %209 = load %arg2[%113, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %209, %arg2[%113, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %210 = addi %arg4, %c2 : index
              %211 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %212 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %213 = mulf %211, %212 {RelaxedPrecision} : f32
              %214 = load %arg2[%210, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %215 = addf %214, %213 {RelaxedPrecision} : f32
              store %215, %arg2[%210, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %216 = load %arg2[%210, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %216, %arg2[%210, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %217 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %218 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %219 = mulf %217, %218 {RelaxedPrecision} : f32
              %220 = load %arg2[%210, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %221 = addf %220, %219 {RelaxedPrecision} : f32
              store %221, %arg2[%210, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %222 = load %arg2[%210, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %222, %arg2[%210, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %223 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %224 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %225 = mulf %223, %224 {RelaxedPrecision} : f32
              %226 = load %arg2[%210, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %227 = addf %226, %225 {RelaxedPrecision} : f32
              store %227, %arg2[%210, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %228 = load %arg2[%210, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %228, %arg2[%210, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %229 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %230 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %231 = mulf %229, %230 {RelaxedPrecision} : f32
              %232 = load %arg2[%210, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %233 = addf %232, %231 {RelaxedPrecision} : f32
              store %233, %arg2[%210, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %234 = load %arg2[%210, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %234, %arg2[%210, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %235 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %236 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %237 = mulf %235, %236 {RelaxedPrecision} : f32
              %238 = load %arg2[%210, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %239 = addf %238, %237 {RelaxedPrecision} : f32
              store %239, %arg2[%210, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %240 = load %arg2[%210, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %240, %arg2[%210, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %241 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %242 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %243 = mulf %241, %242 {RelaxedPrecision} : f32
              %244 = load %arg2[%210, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %245 = addf %244, %243 {RelaxedPrecision} : f32
              store %245, %arg2[%210, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %246 = load %arg2[%210, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %246, %arg2[%210, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %247 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %248 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %249 = mulf %247, %248 {RelaxedPrecision} : f32
              %250 = load %arg2[%210, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %251 = addf %250, %249 {RelaxedPrecision} : f32
              store %251, %arg2[%210, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %252 = load %arg2[%210, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %252, %arg2[%210, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %253 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %254 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %255 = mulf %253, %254 {RelaxedPrecision} : f32
              %256 = load %arg2[%210, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %257 = addf %256, %255 {RelaxedPrecision} : f32
              store %257, %arg2[%210, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %258 = load %arg2[%210, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %258, %arg2[%210, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %259 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %260 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %261 = mulf %259, %260 {RelaxedPrecision} : f32
              %262 = load %arg2[%210, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %263 = addf %262, %261 {RelaxedPrecision} : f32
              store %263, %arg2[%210, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %264 = load %arg2[%210, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %264, %arg2[%210, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %265 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %266 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %267 = mulf %265, %266 {RelaxedPrecision} : f32
              %268 = load %arg2[%210, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %269 = addf %268, %267 {RelaxedPrecision} : f32
              store %269, %arg2[%210, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %270 = load %arg2[%210, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %270, %arg2[%210, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %271 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %272 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %273 = mulf %271, %272 {RelaxedPrecision} : f32
              %274 = load %arg2[%210, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %275 = addf %274, %273 {RelaxedPrecision} : f32
              store %275, %arg2[%210, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %276 = load %arg2[%210, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %276, %arg2[%210, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %277 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %278 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %279 = mulf %277, %278 {RelaxedPrecision} : f32
              %280 = load %arg2[%210, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %281 = addf %280, %279 {RelaxedPrecision} : f32
              store %281, %arg2[%210, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %282 = load %arg2[%210, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %282, %arg2[%210, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %283 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %284 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %285 = mulf %283, %284 {RelaxedPrecision} : f32
              %286 = load %arg2[%210, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %287 = addf %286, %285 {RelaxedPrecision} : f32
              store %287, %arg2[%210, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %288 = load %arg2[%210, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %288, %arg2[%210, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %289 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %290 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %291 = mulf %289, %290 {RelaxedPrecision} : f32
              %292 = load %arg2[%210, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %293 = addf %292, %291 {RelaxedPrecision} : f32
              store %293, %arg2[%210, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %294 = load %arg2[%210, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %294, %arg2[%210, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %295 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %296 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %297 = mulf %295, %296 {RelaxedPrecision} : f32
              %298 = load %arg2[%210, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %299 = addf %298, %297 {RelaxedPrecision} : f32
              store %299, %arg2[%210, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %300 = load %arg2[%210, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %300, %arg2[%210, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %301 = load %arg0[%210, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %302 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %303 = mulf %301, %302 {RelaxedPrecision} : f32
              %304 = load %arg2[%210, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %305 = addf %304, %303 {RelaxedPrecision} : f32
              store %305, %arg2[%210, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %306 = load %arg2[%210, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %306, %arg2[%210, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %307 = addi %arg4, %c3 : index
              %308 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %309 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %310 = mulf %308, %309 {RelaxedPrecision} : f32
              %311 = load %arg2[%307, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %312 = addf %311, %310 {RelaxedPrecision} : f32
              store %312, %arg2[%307, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %313 = load %arg2[%307, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %313, %arg2[%307, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %314 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %315 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %316 = mulf %314, %315 {RelaxedPrecision} : f32
              %317 = load %arg2[%307, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %318 = addf %317, %316 {RelaxedPrecision} : f32
              store %318, %arg2[%307, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %319 = load %arg2[%307, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %319, %arg2[%307, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %320 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %321 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %322 = mulf %320, %321 {RelaxedPrecision} : f32
              %323 = load %arg2[%307, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %324 = addf %323, %322 {RelaxedPrecision} : f32
              store %324, %arg2[%307, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %325 = load %arg2[%307, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %325, %arg2[%307, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %326 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %327 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %328 = mulf %326, %327 {RelaxedPrecision} : f32
              %329 = load %arg2[%307, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %330 = addf %329, %328 {RelaxedPrecision} : f32
              store %330, %arg2[%307, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %331 = load %arg2[%307, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %331, %arg2[%307, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %332 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %333 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %334 = mulf %332, %333 {RelaxedPrecision} : f32
              %335 = load %arg2[%307, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %336 = addf %335, %334 {RelaxedPrecision} : f32
              store %336, %arg2[%307, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %337 = load %arg2[%307, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %337, %arg2[%307, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %338 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %339 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %340 = mulf %338, %339 {RelaxedPrecision} : f32
              %341 = load %arg2[%307, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %342 = addf %341, %340 {RelaxedPrecision} : f32
              store %342, %arg2[%307, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %343 = load %arg2[%307, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %343, %arg2[%307, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %344 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %345 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %346 = mulf %344, %345 {RelaxedPrecision} : f32
              %347 = load %arg2[%307, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %348 = addf %347, %346 {RelaxedPrecision} : f32
              store %348, %arg2[%307, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %349 = load %arg2[%307, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %349, %arg2[%307, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %350 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %351 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %352 = mulf %350, %351 {RelaxedPrecision} : f32
              %353 = load %arg2[%307, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %354 = addf %353, %352 {RelaxedPrecision} : f32
              store %354, %arg2[%307, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %355 = load %arg2[%307, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %355, %arg2[%307, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %356 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %357 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %358 = mulf %356, %357 {RelaxedPrecision} : f32
              %359 = load %arg2[%307, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %360 = addf %359, %358 {RelaxedPrecision} : f32
              store %360, %arg2[%307, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %361 = load %arg2[%307, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %361, %arg2[%307, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %362 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %363 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %364 = mulf %362, %363 {RelaxedPrecision} : f32
              %365 = load %arg2[%307, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %366 = addf %365, %364 {RelaxedPrecision} : f32
              store %366, %arg2[%307, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %367 = load %arg2[%307, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %367, %arg2[%307, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %368 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %369 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %370 = mulf %368, %369 {RelaxedPrecision} : f32
              %371 = load %arg2[%307, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %372 = addf %371, %370 {RelaxedPrecision} : f32
              store %372, %arg2[%307, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %373 = load %arg2[%307, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %373, %arg2[%307, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %374 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %375 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %376 = mulf %374, %375 {RelaxedPrecision} : f32
              %377 = load %arg2[%307, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %378 = addf %377, %376 {RelaxedPrecision} : f32
              store %378, %arg2[%307, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %379 = load %arg2[%307, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %379, %arg2[%307, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %380 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %381 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %382 = mulf %380, %381 {RelaxedPrecision} : f32
              %383 = load %arg2[%307, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %384 = addf %383, %382 {RelaxedPrecision} : f32
              store %384, %arg2[%307, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %385 = load %arg2[%307, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %385, %arg2[%307, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %386 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %387 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %388 = mulf %386, %387 {RelaxedPrecision} : f32
              %389 = load %arg2[%307, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %390 = addf %389, %388 {RelaxedPrecision} : f32
              store %390, %arg2[%307, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %391 = load %arg2[%307, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %391, %arg2[%307, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %392 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %393 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %394 = mulf %392, %393 {RelaxedPrecision} : f32
              %395 = load %arg2[%307, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %396 = addf %395, %394 {RelaxedPrecision} : f32
              store %396, %arg2[%307, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %397 = load %arg2[%307, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %397, %arg2[%307, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %398 = load %arg0[%307, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %399 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %400 = mulf %398, %399 {RelaxedPrecision} : f32
              %401 = load %arg2[%307, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %402 = addf %401, %400 {RelaxedPrecision} : f32
              store %402, %arg2[%307, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %403 = load %arg2[%307, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %403, %arg2[%307, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %404 = addi %arg4, %c4 : index
              %405 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %406 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %407 = mulf %405, %406 {RelaxedPrecision} : f32
              %408 = load %arg2[%404, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %409 = addf %408, %407 {RelaxedPrecision} : f32
              store %409, %arg2[%404, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %410 = load %arg2[%404, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %410, %arg2[%404, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %411 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %412 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %413 = mulf %411, %412 {RelaxedPrecision} : f32
              %414 = load %arg2[%404, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %415 = addf %414, %413 {RelaxedPrecision} : f32
              store %415, %arg2[%404, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %416 = load %arg2[%404, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %416, %arg2[%404, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %417 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %418 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %419 = mulf %417, %418 {RelaxedPrecision} : f32
              %420 = load %arg2[%404, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %421 = addf %420, %419 {RelaxedPrecision} : f32
              store %421, %arg2[%404, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %422 = load %arg2[%404, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %422, %arg2[%404, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %423 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %424 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %425 = mulf %423, %424 {RelaxedPrecision} : f32
              %426 = load %arg2[%404, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %427 = addf %426, %425 {RelaxedPrecision} : f32
              store %427, %arg2[%404, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %428 = load %arg2[%404, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %428, %arg2[%404, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %429 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %430 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %431 = mulf %429, %430 {RelaxedPrecision} : f32
              %432 = load %arg2[%404, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %433 = addf %432, %431 {RelaxedPrecision} : f32
              store %433, %arg2[%404, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %434 = load %arg2[%404, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %434, %arg2[%404, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %435 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %436 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %437 = mulf %435, %436 {RelaxedPrecision} : f32
              %438 = load %arg2[%404, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %439 = addf %438, %437 {RelaxedPrecision} : f32
              store %439, %arg2[%404, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %440 = load %arg2[%404, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %440, %arg2[%404, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %441 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %442 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %443 = mulf %441, %442 {RelaxedPrecision} : f32
              %444 = load %arg2[%404, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %445 = addf %444, %443 {RelaxedPrecision} : f32
              store %445, %arg2[%404, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %446 = load %arg2[%404, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %446, %arg2[%404, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %447 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %448 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %449 = mulf %447, %448 {RelaxedPrecision} : f32
              %450 = load %arg2[%404, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %451 = addf %450, %449 {RelaxedPrecision} : f32
              store %451, %arg2[%404, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %452 = load %arg2[%404, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %452, %arg2[%404, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %453 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %454 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %455 = mulf %453, %454 {RelaxedPrecision} : f32
              %456 = load %arg2[%404, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %457 = addf %456, %455 {RelaxedPrecision} : f32
              store %457, %arg2[%404, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %458 = load %arg2[%404, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %458, %arg2[%404, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %459 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %460 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %461 = mulf %459, %460 {RelaxedPrecision} : f32
              %462 = load %arg2[%404, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %463 = addf %462, %461 {RelaxedPrecision} : f32
              store %463, %arg2[%404, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %464 = load %arg2[%404, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %464, %arg2[%404, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %465 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %466 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %467 = mulf %465, %466 {RelaxedPrecision} : f32
              %468 = load %arg2[%404, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %469 = addf %468, %467 {RelaxedPrecision} : f32
              store %469, %arg2[%404, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %470 = load %arg2[%404, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %470, %arg2[%404, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %471 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %472 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %473 = mulf %471, %472 {RelaxedPrecision} : f32
              %474 = load %arg2[%404, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %475 = addf %474, %473 {RelaxedPrecision} : f32
              store %475, %arg2[%404, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %476 = load %arg2[%404, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %476, %arg2[%404, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %477 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %478 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %479 = mulf %477, %478 {RelaxedPrecision} : f32
              %480 = load %arg2[%404, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %481 = addf %480, %479 {RelaxedPrecision} : f32
              store %481, %arg2[%404, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %482 = load %arg2[%404, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %482, %arg2[%404, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %483 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %484 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %485 = mulf %483, %484 {RelaxedPrecision} : f32
              %486 = load %arg2[%404, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %487 = addf %486, %485 {RelaxedPrecision} : f32
              store %487, %arg2[%404, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %488 = load %arg2[%404, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %488, %arg2[%404, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %489 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %490 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %491 = mulf %489, %490 {RelaxedPrecision} : f32
              %492 = load %arg2[%404, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %493 = addf %492, %491 {RelaxedPrecision} : f32
              store %493, %arg2[%404, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %494 = load %arg2[%404, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %494, %arg2[%404, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %495 = load %arg0[%404, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %496 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %497 = mulf %495, %496 {RelaxedPrecision} : f32
              %498 = load %arg2[%404, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %499 = addf %498, %497 {RelaxedPrecision} : f32
              store %499, %arg2[%404, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %500 = load %arg2[%404, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %500, %arg2[%404, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %501 = addi %arg4, %c5 : index
              %502 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %503 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %504 = mulf %502, %503 {RelaxedPrecision} : f32
              %505 = load %arg2[%501, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %506 = addf %505, %504 {RelaxedPrecision} : f32
              store %506, %arg2[%501, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %507 = load %arg2[%501, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %507, %arg2[%501, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %508 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %509 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %510 = mulf %508, %509 {RelaxedPrecision} : f32
              %511 = load %arg2[%501, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %512 = addf %511, %510 {RelaxedPrecision} : f32
              store %512, %arg2[%501, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %513 = load %arg2[%501, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %513, %arg2[%501, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %514 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %515 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %516 = mulf %514, %515 {RelaxedPrecision} : f32
              %517 = load %arg2[%501, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %518 = addf %517, %516 {RelaxedPrecision} : f32
              store %518, %arg2[%501, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %519 = load %arg2[%501, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %519, %arg2[%501, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %520 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %521 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %522 = mulf %520, %521 {RelaxedPrecision} : f32
              %523 = load %arg2[%501, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %524 = addf %523, %522 {RelaxedPrecision} : f32
              store %524, %arg2[%501, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %525 = load %arg2[%501, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %525, %arg2[%501, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %526 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %527 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %528 = mulf %526, %527 {RelaxedPrecision} : f32
              %529 = load %arg2[%501, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %530 = addf %529, %528 {RelaxedPrecision} : f32
              store %530, %arg2[%501, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %531 = load %arg2[%501, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %531, %arg2[%501, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %532 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %533 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %534 = mulf %532, %533 {RelaxedPrecision} : f32
              %535 = load %arg2[%501, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %536 = addf %535, %534 {RelaxedPrecision} : f32
              store %536, %arg2[%501, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %537 = load %arg2[%501, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %537, %arg2[%501, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %538 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %539 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %540 = mulf %538, %539 {RelaxedPrecision} : f32
              %541 = load %arg2[%501, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %542 = addf %541, %540 {RelaxedPrecision} : f32
              store %542, %arg2[%501, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %543 = load %arg2[%501, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %543, %arg2[%501, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %544 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %545 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %546 = mulf %544, %545 {RelaxedPrecision} : f32
              %547 = load %arg2[%501, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %548 = addf %547, %546 {RelaxedPrecision} : f32
              store %548, %arg2[%501, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %549 = load %arg2[%501, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %549, %arg2[%501, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %550 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %551 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %552 = mulf %550, %551 {RelaxedPrecision} : f32
              %553 = load %arg2[%501, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %554 = addf %553, %552 {RelaxedPrecision} : f32
              store %554, %arg2[%501, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %555 = load %arg2[%501, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %555, %arg2[%501, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %556 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %557 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %558 = mulf %556, %557 {RelaxedPrecision} : f32
              %559 = load %arg2[%501, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %560 = addf %559, %558 {RelaxedPrecision} : f32
              store %560, %arg2[%501, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %561 = load %arg2[%501, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %561, %arg2[%501, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %562 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %563 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %564 = mulf %562, %563 {RelaxedPrecision} : f32
              %565 = load %arg2[%501, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %566 = addf %565, %564 {RelaxedPrecision} : f32
              store %566, %arg2[%501, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %567 = load %arg2[%501, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %567, %arg2[%501, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %568 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %569 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %570 = mulf %568, %569 {RelaxedPrecision} : f32
              %571 = load %arg2[%501, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %572 = addf %571, %570 {RelaxedPrecision} : f32
              store %572, %arg2[%501, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %573 = load %arg2[%501, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %573, %arg2[%501, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %574 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %575 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %576 = mulf %574, %575 {RelaxedPrecision} : f32
              %577 = load %arg2[%501, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %578 = addf %577, %576 {RelaxedPrecision} : f32
              store %578, %arg2[%501, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %579 = load %arg2[%501, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %579, %arg2[%501, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %580 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %581 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %582 = mulf %580, %581 {RelaxedPrecision} : f32
              %583 = load %arg2[%501, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %584 = addf %583, %582 {RelaxedPrecision} : f32
              store %584, %arg2[%501, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %585 = load %arg2[%501, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %585, %arg2[%501, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %586 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %587 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %588 = mulf %586, %587 {RelaxedPrecision} : f32
              %589 = load %arg2[%501, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %590 = addf %589, %588 {RelaxedPrecision} : f32
              store %590, %arg2[%501, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %591 = load %arg2[%501, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %591, %arg2[%501, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %592 = load %arg0[%501, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %593 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %594 = mulf %592, %593 {RelaxedPrecision} : f32
              %595 = load %arg2[%501, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %596 = addf %595, %594 {RelaxedPrecision} : f32
              store %596, %arg2[%501, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %597 = load %arg2[%501, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %597, %arg2[%501, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            }
          }
        }
      }
      scf.for %arg4 = %c0 to %c256 step %c16 {
        scf.for %arg5 = %c0 to %c128 step %c4 {
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %0 = addi %arg3, %arg4 : index
            %1 = addi %arg5, %arg6 : index
            %2 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %3 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %4 = mulf %2, %3 {RelaxedPrecision} : f32
            %5 = load %arg2[%c780, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %6 = addf %5, %4 {RelaxedPrecision} : f32
            store %6, %arg2[%c780, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %7 = load %arg2[%c780, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %7, %arg2[%c780, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %8 = addi %0, %c1 : index
            %9 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %10 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %11 = mulf %9, %10 {RelaxedPrecision} : f32
            %12 = load %arg2[%c780, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %13 = addf %12, %11 {RelaxedPrecision} : f32
            store %13, %arg2[%c780, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %14 = load %arg2[%c780, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %14, %arg2[%c780, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %15 = addi %0, %c2 : index
            %16 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %17 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %18 = mulf %16, %17 {RelaxedPrecision} : f32
            %19 = load %arg2[%c780, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %20 = addf %19, %18 {RelaxedPrecision} : f32
            store %20, %arg2[%c780, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %21 = load %arg2[%c780, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %21, %arg2[%c780, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %22 = addi %0, %c3 : index
            %23 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %24 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %25 = mulf %23, %24 {RelaxedPrecision} : f32
            %26 = load %arg2[%c780, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %27 = addf %26, %25 {RelaxedPrecision} : f32
            store %27, %arg2[%c780, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %28 = load %arg2[%c780, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %28, %arg2[%c780, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %29 = addi %0, %c4 : index
            %30 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %31 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %32 = mulf %30, %31 {RelaxedPrecision} : f32
            %33 = load %arg2[%c780, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %34 = addf %33, %32 {RelaxedPrecision} : f32
            store %34, %arg2[%c780, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %35 = load %arg2[%c780, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %35, %arg2[%c780, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %36 = addi %0, %c5 : index
            %37 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %38 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %39 = mulf %37, %38 {RelaxedPrecision} : f32
            %40 = load %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %41 = addf %40, %39 {RelaxedPrecision} : f32
            store %41, %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %42 = load %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %42, %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %43 = addi %0, %c6 : index
            %44 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %45 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %46 = mulf %44, %45 {RelaxedPrecision} : f32
            %47 = load %arg2[%c780, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %48 = addf %47, %46 {RelaxedPrecision} : f32
            store %48, %arg2[%c780, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %49 = load %arg2[%c780, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %49, %arg2[%c780, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %50 = addi %0, %c7 : index
            %51 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %52 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %53 = mulf %51, %52 {RelaxedPrecision} : f32
            %54 = load %arg2[%c780, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %55 = addf %54, %53 {RelaxedPrecision} : f32
            store %55, %arg2[%c780, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %56 = load %arg2[%c780, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %56, %arg2[%c780, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %57 = addi %0, %c8 : index
            %58 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %59 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %60 = mulf %58, %59 {RelaxedPrecision} : f32
            %61 = load %arg2[%c780, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %62 = addf %61, %60 {RelaxedPrecision} : f32
            store %62, %arg2[%c780, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %63 = load %arg2[%c780, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %63, %arg2[%c780, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %64 = addi %0, %c9 : index
            %65 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %66 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %67 = mulf %65, %66 {RelaxedPrecision} : f32
            %68 = load %arg2[%c780, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %69 = addf %68, %67 {RelaxedPrecision} : f32
            store %69, %arg2[%c780, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %70 = load %arg2[%c780, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %70, %arg2[%c780, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %71 = addi %0, %c10 : index
            %72 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %73 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %74 = mulf %72, %73 {RelaxedPrecision} : f32
            %75 = load %arg2[%c780, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %76 = addf %75, %74 {RelaxedPrecision} : f32
            store %76, %arg2[%c780, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %77 = load %arg2[%c780, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %77, %arg2[%c780, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %78 = addi %0, %c11 : index
            %79 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %80 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %81 = mulf %79, %80 {RelaxedPrecision} : f32
            %82 = load %arg2[%c780, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %83 = addf %82, %81 {RelaxedPrecision} : f32
            store %83, %arg2[%c780, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %84 = load %arg2[%c780, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %84, %arg2[%c780, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %85 = addi %0, %c12 : index
            %86 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %87 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %88 = mulf %86, %87 {RelaxedPrecision} : f32
            %89 = load %arg2[%c780, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %90 = addf %89, %88 {RelaxedPrecision} : f32
            store %90, %arg2[%c780, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %91 = load %arg2[%c780, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %91, %arg2[%c780, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %92 = addi %0, %c13 : index
            %93 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %94 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %95 = mulf %93, %94 {RelaxedPrecision} : f32
            %96 = load %arg2[%c780, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %97 = addf %96, %95 {RelaxedPrecision} : f32
            store %97, %arg2[%c780, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %98 = load %arg2[%c780, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %98, %arg2[%c780, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %99 = addi %0, %c14 : index
            %100 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %101 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %102 = mulf %100, %101 {RelaxedPrecision} : f32
            %103 = load %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %104 = addf %103, %102 {RelaxedPrecision} : f32
            store %104, %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %105 = load %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %105, %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %106 = addi %0, %c15 : index
            %107 = load %arg0[%c780, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %108 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %109 = mulf %107, %108 {RelaxedPrecision} : f32
            %110 = load %arg2[%c780, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %111 = addf %110, %109 {RelaxedPrecision} : f32
            store %111, %arg2[%c780, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %112 = load %arg2[%c780, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %112, %arg2[%c780, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %113 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %114 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %115 = mulf %113, %114 {RelaxedPrecision} : f32
            %116 = load %arg2[%c781, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %117 = addf %116, %115 {RelaxedPrecision} : f32
            store %117, %arg2[%c781, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %118 = load %arg2[%c781, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %118, %arg2[%c781, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %119 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %120 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %121 = mulf %119, %120 {RelaxedPrecision} : f32
            %122 = load %arg2[%c781, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %123 = addf %122, %121 {RelaxedPrecision} : f32
            store %123, %arg2[%c781, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %124 = load %arg2[%c781, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %124, %arg2[%c781, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %125 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %126 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %127 = mulf %125, %126 {RelaxedPrecision} : f32
            %128 = load %arg2[%c781, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %129 = addf %128, %127 {RelaxedPrecision} : f32
            store %129, %arg2[%c781, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %130 = load %arg2[%c781, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %130, %arg2[%c781, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %131 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %132 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %133 = mulf %131, %132 {RelaxedPrecision} : f32
            %134 = load %arg2[%c781, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %135 = addf %134, %133 {RelaxedPrecision} : f32
            store %135, %arg2[%c781, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %136 = load %arg2[%c781, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %136, %arg2[%c781, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %137 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %138 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %139 = mulf %137, %138 {RelaxedPrecision} : f32
            %140 = load %arg2[%c781, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %141 = addf %140, %139 {RelaxedPrecision} : f32
            store %141, %arg2[%c781, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %142 = load %arg2[%c781, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %142, %arg2[%c781, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %143 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %144 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %145 = mulf %143, %144 {RelaxedPrecision} : f32
            %146 = load %arg2[%c781, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %147 = addf %146, %145 {RelaxedPrecision} : f32
            store %147, %arg2[%c781, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %148 = load %arg2[%c781, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %148, %arg2[%c781, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %149 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %150 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %151 = mulf %149, %150 {RelaxedPrecision} : f32
            %152 = load %arg2[%c781, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %153 = addf %152, %151 {RelaxedPrecision} : f32
            store %153, %arg2[%c781, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %154 = load %arg2[%c781, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %154, %arg2[%c781, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %155 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %156 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %157 = mulf %155, %156 {RelaxedPrecision} : f32
            %158 = load %arg2[%c781, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %159 = addf %158, %157 {RelaxedPrecision} : f32
            store %159, %arg2[%c781, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %160 = load %arg2[%c781, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %160, %arg2[%c781, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %161 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %162 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %163 = mulf %161, %162 {RelaxedPrecision} : f32
            %164 = load %arg2[%c781, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %165 = addf %164, %163 {RelaxedPrecision} : f32
            store %165, %arg2[%c781, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %166 = load %arg2[%c781, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %166, %arg2[%c781, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %167 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %168 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %169 = mulf %167, %168 {RelaxedPrecision} : f32
            %170 = load %arg2[%c781, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %171 = addf %170, %169 {RelaxedPrecision} : f32
            store %171, %arg2[%c781, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %172 = load %arg2[%c781, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %172, %arg2[%c781, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %173 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %174 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %175 = mulf %173, %174 {RelaxedPrecision} : f32
            %176 = load %arg2[%c781, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %177 = addf %176, %175 {RelaxedPrecision} : f32
            store %177, %arg2[%c781, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %178 = load %arg2[%c781, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %178, %arg2[%c781, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %179 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %180 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %181 = mulf %179, %180 {RelaxedPrecision} : f32
            %182 = load %arg2[%c781, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %183 = addf %182, %181 {RelaxedPrecision} : f32
            store %183, %arg2[%c781, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %184 = load %arg2[%c781, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %184, %arg2[%c781, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %185 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %186 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %187 = mulf %185, %186 {RelaxedPrecision} : f32
            %188 = load %arg2[%c781, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %189 = addf %188, %187 {RelaxedPrecision} : f32
            store %189, %arg2[%c781, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %190 = load %arg2[%c781, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %190, %arg2[%c781, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %191 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %192 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %193 = mulf %191, %192 {RelaxedPrecision} : f32
            %194 = load %arg2[%c781, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %195 = addf %194, %193 {RelaxedPrecision} : f32
            store %195, %arg2[%c781, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %196 = load %arg2[%c781, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %196, %arg2[%c781, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %197 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %198 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %199 = mulf %197, %198 {RelaxedPrecision} : f32
            %200 = load %arg2[%c781, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %201 = addf %200, %199 {RelaxedPrecision} : f32
            store %201, %arg2[%c781, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %202 = load %arg2[%c781, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %202, %arg2[%c781, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %203 = load %arg0[%c781, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %204 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %205 = mulf %203, %204 {RelaxedPrecision} : f32
            %206 = load %arg2[%c781, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %207 = addf %206, %205 {RelaxedPrecision} : f32
            store %207, %arg2[%c781, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %208 = load %arg2[%c781, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %208, %arg2[%c781, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %209 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %210 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %211 = mulf %209, %210 {RelaxedPrecision} : f32
            %212 = load %arg2[%c782, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %213 = addf %212, %211 {RelaxedPrecision} : f32
            store %213, %arg2[%c782, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %214 = load %arg2[%c782, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %214, %arg2[%c782, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %215 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %216 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %217 = mulf %215, %216 {RelaxedPrecision} : f32
            %218 = load %arg2[%c782, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %219 = addf %218, %217 {RelaxedPrecision} : f32
            store %219, %arg2[%c782, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %220 = load %arg2[%c782, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %220, %arg2[%c782, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %221 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %222 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %223 = mulf %221, %222 {RelaxedPrecision} : f32
            %224 = load %arg2[%c782, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %225 = addf %224, %223 {RelaxedPrecision} : f32
            store %225, %arg2[%c782, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %226 = load %arg2[%c782, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %226, %arg2[%c782, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %227 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %228 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %229 = mulf %227, %228 {RelaxedPrecision} : f32
            %230 = load %arg2[%c782, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %231 = addf %230, %229 {RelaxedPrecision} : f32
            store %231, %arg2[%c782, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %232 = load %arg2[%c782, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %232, %arg2[%c782, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %233 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %234 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %235 = mulf %233, %234 {RelaxedPrecision} : f32
            %236 = load %arg2[%c782, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %237 = addf %236, %235 {RelaxedPrecision} : f32
            store %237, %arg2[%c782, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %238 = load %arg2[%c782, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %238, %arg2[%c782, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %239 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %240 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %241 = mulf %239, %240 {RelaxedPrecision} : f32
            %242 = load %arg2[%c782, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %243 = addf %242, %241 {RelaxedPrecision} : f32
            store %243, %arg2[%c782, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %244 = load %arg2[%c782, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %244, %arg2[%c782, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %245 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %246 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %247 = mulf %245, %246 {RelaxedPrecision} : f32
            %248 = load %arg2[%c782, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %249 = addf %248, %247 {RelaxedPrecision} : f32
            store %249, %arg2[%c782, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %250 = load %arg2[%c782, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %250, %arg2[%c782, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %251 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %252 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %253 = mulf %251, %252 {RelaxedPrecision} : f32
            %254 = load %arg2[%c782, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %255 = addf %254, %253 {RelaxedPrecision} : f32
            store %255, %arg2[%c782, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %256 = load %arg2[%c782, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %256, %arg2[%c782, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %257 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %258 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %259 = mulf %257, %258 {RelaxedPrecision} : f32
            %260 = load %arg2[%c782, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %261 = addf %260, %259 {RelaxedPrecision} : f32
            store %261, %arg2[%c782, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %262 = load %arg2[%c782, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %262, %arg2[%c782, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %263 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %264 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %265 = mulf %263, %264 {RelaxedPrecision} : f32
            %266 = load %arg2[%c782, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %267 = addf %266, %265 {RelaxedPrecision} : f32
            store %267, %arg2[%c782, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %268 = load %arg2[%c782, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %268, %arg2[%c782, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %269 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %270 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %271 = mulf %269, %270 {RelaxedPrecision} : f32
            %272 = load %arg2[%c782, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %273 = addf %272, %271 {RelaxedPrecision} : f32
            store %273, %arg2[%c782, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %274 = load %arg2[%c782, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %274, %arg2[%c782, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %275 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %276 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %277 = mulf %275, %276 {RelaxedPrecision} : f32
            %278 = load %arg2[%c782, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %279 = addf %278, %277 {RelaxedPrecision} : f32
            store %279, %arg2[%c782, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %280 = load %arg2[%c782, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %280, %arg2[%c782, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %281 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %282 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %283 = mulf %281, %282 {RelaxedPrecision} : f32
            %284 = load %arg2[%c782, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %285 = addf %284, %283 {RelaxedPrecision} : f32
            store %285, %arg2[%c782, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %286 = load %arg2[%c782, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %286, %arg2[%c782, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %287 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %288 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %289 = mulf %287, %288 {RelaxedPrecision} : f32
            %290 = load %arg2[%c782, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %291 = addf %290, %289 {RelaxedPrecision} : f32
            store %291, %arg2[%c782, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %292 = load %arg2[%c782, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %292, %arg2[%c782, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %293 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %294 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %295 = mulf %293, %294 {RelaxedPrecision} : f32
            %296 = load %arg2[%c782, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %297 = addf %296, %295 {RelaxedPrecision} : f32
            store %297, %arg2[%c782, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %298 = load %arg2[%c782, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %298, %arg2[%c782, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %299 = load %arg0[%c782, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %300 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %301 = mulf %299, %300 {RelaxedPrecision} : f32
            %302 = load %arg2[%c782, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %303 = addf %302, %301 {RelaxedPrecision} : f32
            store %303, %arg2[%c782, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %304 = load %arg2[%c782, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %304, %arg2[%c782, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %305 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %306 = load %arg1[%1, %0] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %307 = mulf %305, %306 {RelaxedPrecision} : f32
            %308 = load %arg2[%c783, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %309 = addf %308, %307 {RelaxedPrecision} : f32
            store %309, %arg2[%c783, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %310 = load %arg2[%c783, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %310, %arg2[%c783, %0] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %311 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %312 = load %arg1[%1, %8] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %313 = mulf %311, %312 {RelaxedPrecision} : f32
            %314 = load %arg2[%c783, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %315 = addf %314, %313 {RelaxedPrecision} : f32
            store %315, %arg2[%c783, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %316 = load %arg2[%c783, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %316, %arg2[%c783, %8] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %317 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %318 = load %arg1[%1, %15] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %319 = mulf %317, %318 {RelaxedPrecision} : f32
            %320 = load %arg2[%c783, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %321 = addf %320, %319 {RelaxedPrecision} : f32
            store %321, %arg2[%c783, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %322 = load %arg2[%c783, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %322, %arg2[%c783, %15] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %323 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %324 = load %arg1[%1, %22] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %325 = mulf %323, %324 {RelaxedPrecision} : f32
            %326 = load %arg2[%c783, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %327 = addf %326, %325 {RelaxedPrecision} : f32
            store %327, %arg2[%c783, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %328 = load %arg2[%c783, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %328, %arg2[%c783, %22] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %329 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %330 = load %arg1[%1, %29] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %331 = mulf %329, %330 {RelaxedPrecision} : f32
            %332 = load %arg2[%c783, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %333 = addf %332, %331 {RelaxedPrecision} : f32
            store %333, %arg2[%c783, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %334 = load %arg2[%c783, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %334, %arg2[%c783, %29] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %335 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %336 = load %arg1[%1, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %337 = mulf %335, %336 {RelaxedPrecision} : f32
            %338 = load %arg2[%c783, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %339 = addf %338, %337 {RelaxedPrecision} : f32
            store %339, %arg2[%c783, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %340 = load %arg2[%c783, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %340, %arg2[%c783, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %341 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %342 = load %arg1[%1, %43] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %343 = mulf %341, %342 {RelaxedPrecision} : f32
            %344 = load %arg2[%c783, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %345 = addf %344, %343 {RelaxedPrecision} : f32
            store %345, %arg2[%c783, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %346 = load %arg2[%c783, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %346, %arg2[%c783, %43] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %347 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %348 = load %arg1[%1, %50] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %349 = mulf %347, %348 {RelaxedPrecision} : f32
            %350 = load %arg2[%c783, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %351 = addf %350, %349 {RelaxedPrecision} : f32
            store %351, %arg2[%c783, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %352 = load %arg2[%c783, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %352, %arg2[%c783, %50] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %353 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %354 = load %arg1[%1, %57] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %355 = mulf %353, %354 {RelaxedPrecision} : f32
            %356 = load %arg2[%c783, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %357 = addf %356, %355 {RelaxedPrecision} : f32
            store %357, %arg2[%c783, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %358 = load %arg2[%c783, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %358, %arg2[%c783, %57] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %359 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %360 = load %arg1[%1, %64] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %361 = mulf %359, %360 {RelaxedPrecision} : f32
            %362 = load %arg2[%c783, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %363 = addf %362, %361 {RelaxedPrecision} : f32
            store %363, %arg2[%c783, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %364 = load %arg2[%c783, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %364, %arg2[%c783, %64] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %365 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %366 = load %arg1[%1, %71] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %367 = mulf %365, %366 {RelaxedPrecision} : f32
            %368 = load %arg2[%c783, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %369 = addf %368, %367 {RelaxedPrecision} : f32
            store %369, %arg2[%c783, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %370 = load %arg2[%c783, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %370, %arg2[%c783, %71] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %371 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %372 = load %arg1[%1, %78] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %373 = mulf %371, %372 {RelaxedPrecision} : f32
            %374 = load %arg2[%c783, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %375 = addf %374, %373 {RelaxedPrecision} : f32
            store %375, %arg2[%c783, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %376 = load %arg2[%c783, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %376, %arg2[%c783, %78] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %377 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %378 = load %arg1[%1, %85] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %379 = mulf %377, %378 {RelaxedPrecision} : f32
            %380 = load %arg2[%c783, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %381 = addf %380, %379 {RelaxedPrecision} : f32
            store %381, %arg2[%c783, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %382 = load %arg2[%c783, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %382, %arg2[%c783, %85] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %383 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %384 = load %arg1[%1, %92] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %385 = mulf %383, %384 {RelaxedPrecision} : f32
            %386 = load %arg2[%c783, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %387 = addf %386, %385 {RelaxedPrecision} : f32
            store %387, %arg2[%c783, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %388 = load %arg2[%c783, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %388, %arg2[%c783, %92] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %389 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %390 = load %arg1[%1, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %391 = mulf %389, %390 {RelaxedPrecision} : f32
            %392 = load %arg2[%c783, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %393 = addf %392, %391 {RelaxedPrecision} : f32
            store %393, %arg2[%c783, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %394 = load %arg2[%c783, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %394, %arg2[%c783, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %395 = load %arg0[%c783, %1] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %396 = load %arg1[%1, %106] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %397 = mulf %395, %396 {RelaxedPrecision} : f32
            %398 = load %arg2[%c783, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %399 = addf %398, %397 {RelaxedPrecision} : f32
            store %399, %arg2[%c783, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %400 = load %arg2[%c783, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %400, %arg2[%c783, %106] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
          }
        }
      }
    }
    return
  }
  func @optimized_matmul_py(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, accv.emit_header_decl, accv.emit_raw_pointer_api} {
    call @optimized_matmul_py_impl_17630232307017152746(%arg0, %arg1, %arg2) : (memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) -> ()
    return
  }
}
