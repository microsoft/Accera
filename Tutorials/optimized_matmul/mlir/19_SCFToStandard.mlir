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
    br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb23
    %1 = cmpi "slt", %0, %c512 : index
    cond_br %1, ^bb2, ^bb24
  ^bb2:  // pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb13
    %3 = cmpi "slt", %2, %c780 : index
    cond_br %3, ^bb4, ^bb14
  ^bb4:  // pred: ^bb3
    br ^bb5(%c0 : index)
  ^bb5(%4: index):  // 2 preds: ^bb4, ^bb12
    %5 = cmpi "slt", %4, %c256 : index
    cond_br %5, ^bb6, ^bb13
  ^bb6:  // pred: ^bb5
    br ^bb7(%c0 : index)
  ^bb7(%6: index):  // 2 preds: ^bb6, ^bb11
    %7 = cmpi "slt", %6, %c128 : index
    cond_br %7, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    br ^bb9(%c0 : index)
  ^bb9(%8: index):  // 2 preds: ^bb8, ^bb10
    %9 = cmpi "slt", %8, %c4 : index
    cond_br %9, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %10 = addi %0, %4 : index
    %11 = addi %6, %8 : index
    %12 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %13 = load %arg1[%11, %10] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %14 = mulf %12, %13 {RelaxedPrecision} : f32
    %15 = load %arg2[%2, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %16 = addf %15, %14 {RelaxedPrecision} : f32
    store %16, %arg2[%2, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %17 = load %arg2[%2, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %17, %arg2[%2, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %18 = addi %10, %c1 : index
    %19 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %20 = load %arg1[%11, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %21 = mulf %19, %20 {RelaxedPrecision} : f32
    %22 = load %arg2[%2, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %23 = addf %22, %21 {RelaxedPrecision} : f32
    store %23, %arg2[%2, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %24 = load %arg2[%2, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %24, %arg2[%2, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %25 = addi %10, %c2 : index
    %26 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %27 = load %arg1[%11, %25] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %28 = mulf %26, %27 {RelaxedPrecision} : f32
    %29 = load %arg2[%2, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %30 = addf %29, %28 {RelaxedPrecision} : f32
    store %30, %arg2[%2, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %31 = load %arg2[%2, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %31, %arg2[%2, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %32 = addi %10, %c3 : index
    %33 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %34 = load %arg1[%11, %32] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %35 = mulf %33, %34 {RelaxedPrecision} : f32
    %36 = load %arg2[%2, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %37 = addf %36, %35 {RelaxedPrecision} : f32
    store %37, %arg2[%2, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %38 = load %arg2[%2, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %38, %arg2[%2, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %39 = addi %10, %c4 : index
    %40 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %41 = load %arg1[%11, %39] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %42 = mulf %40, %41 {RelaxedPrecision} : f32
    %43 = load %arg2[%2, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %44 = addf %43, %42 {RelaxedPrecision} : f32
    store %44, %arg2[%2, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %45 = load %arg2[%2, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %45, %arg2[%2, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %46 = addi %10, %c5 : index
    %47 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %48 = load %arg1[%11, %46] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %49 = mulf %47, %48 {RelaxedPrecision} : f32
    %50 = load %arg2[%2, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %51 = addf %50, %49 {RelaxedPrecision} : f32
    store %51, %arg2[%2, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %52 = load %arg2[%2, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %52, %arg2[%2, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %53 = addi %10, %c6 : index
    %54 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %55 = load %arg1[%11, %53] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %56 = mulf %54, %55 {RelaxedPrecision} : f32
    %57 = load %arg2[%2, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %58 = addf %57, %56 {RelaxedPrecision} : f32
    store %58, %arg2[%2, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %59 = load %arg2[%2, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %59, %arg2[%2, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %60 = addi %10, %c7 : index
    %61 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %62 = load %arg1[%11, %60] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %63 = mulf %61, %62 {RelaxedPrecision} : f32
    %64 = load %arg2[%2, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %65 = addf %64, %63 {RelaxedPrecision} : f32
    store %65, %arg2[%2, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %66 = load %arg2[%2, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %66, %arg2[%2, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %67 = addi %10, %c8 : index
    %68 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %69 = load %arg1[%11, %67] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %70 = mulf %68, %69 {RelaxedPrecision} : f32
    %71 = load %arg2[%2, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %72 = addf %71, %70 {RelaxedPrecision} : f32
    store %72, %arg2[%2, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %73 = load %arg2[%2, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %73, %arg2[%2, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %74 = addi %10, %c9 : index
    %75 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %76 = load %arg1[%11, %74] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %77 = mulf %75, %76 {RelaxedPrecision} : f32
    %78 = load %arg2[%2, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %79 = addf %78, %77 {RelaxedPrecision} : f32
    store %79, %arg2[%2, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %80 = load %arg2[%2, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %80, %arg2[%2, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %81 = addi %10, %c10 : index
    %82 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %83 = load %arg1[%11, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %84 = mulf %82, %83 {RelaxedPrecision} : f32
    %85 = load %arg2[%2, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %86 = addf %85, %84 {RelaxedPrecision} : f32
    store %86, %arg2[%2, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %87 = load %arg2[%2, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %87, %arg2[%2, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %88 = addi %10, %c11 : index
    %89 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %90 = load %arg1[%11, %88] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %91 = mulf %89, %90 {RelaxedPrecision} : f32
    %92 = load %arg2[%2, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %93 = addf %92, %91 {RelaxedPrecision} : f32
    store %93, %arg2[%2, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %94 = load %arg2[%2, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %94, %arg2[%2, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %95 = addi %10, %c12 : index
    %96 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %97 = load %arg1[%11, %95] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %98 = mulf %96, %97 {RelaxedPrecision} : f32
    %99 = load %arg2[%2, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %100 = addf %99, %98 {RelaxedPrecision} : f32
    store %100, %arg2[%2, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %101 = load %arg2[%2, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %101, %arg2[%2, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %102 = addi %10, %c13 : index
    %103 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %104 = load %arg1[%11, %102] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %105 = mulf %103, %104 {RelaxedPrecision} : f32
    %106 = load %arg2[%2, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %107 = addf %106, %105 {RelaxedPrecision} : f32
    store %107, %arg2[%2, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %108 = load %arg2[%2, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %108, %arg2[%2, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %109 = addi %10, %c14 : index
    %110 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %111 = load %arg1[%11, %109] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %112 = mulf %110, %111 {RelaxedPrecision} : f32
    %113 = load %arg2[%2, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %114 = addf %113, %112 {RelaxedPrecision} : f32
    store %114, %arg2[%2, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %115 = load %arg2[%2, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %115, %arg2[%2, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %116 = addi %10, %c15 : index
    %117 = load %arg0[%2, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %118 = load %arg1[%11, %116] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %119 = mulf %117, %118 {RelaxedPrecision} : f32
    %120 = load %arg2[%2, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %121 = addf %120, %119 {RelaxedPrecision} : f32
    store %121, %arg2[%2, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %122 = load %arg2[%2, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %122, %arg2[%2, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %123 = addi %2, %c1 : index
    %124 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %125 = load %arg1[%11, %10] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %126 = mulf %124, %125 {RelaxedPrecision} : f32
    %127 = load %arg2[%123, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %128 = addf %127, %126 {RelaxedPrecision} : f32
    store %128, %arg2[%123, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %129 = load %arg2[%123, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %129, %arg2[%123, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %130 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %131 = load %arg1[%11, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %132 = mulf %130, %131 {RelaxedPrecision} : f32
    %133 = load %arg2[%123, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %134 = addf %133, %132 {RelaxedPrecision} : f32
    store %134, %arg2[%123, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %135 = load %arg2[%123, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %135, %arg2[%123, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %136 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %137 = load %arg1[%11, %25] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %138 = mulf %136, %137 {RelaxedPrecision} : f32
    %139 = load %arg2[%123, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %140 = addf %139, %138 {RelaxedPrecision} : f32
    store %140, %arg2[%123, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %141 = load %arg2[%123, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %141, %arg2[%123, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %142 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %143 = load %arg1[%11, %32] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %144 = mulf %142, %143 {RelaxedPrecision} : f32
    %145 = load %arg2[%123, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %146 = addf %145, %144 {RelaxedPrecision} : f32
    store %146, %arg2[%123, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %147 = load %arg2[%123, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %147, %arg2[%123, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %148 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %149 = load %arg1[%11, %39] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %150 = mulf %148, %149 {RelaxedPrecision} : f32
    %151 = load %arg2[%123, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %152 = addf %151, %150 {RelaxedPrecision} : f32
    store %152, %arg2[%123, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %153 = load %arg2[%123, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %153, %arg2[%123, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %154 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %155 = load %arg1[%11, %46] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %156 = mulf %154, %155 {RelaxedPrecision} : f32
    %157 = load %arg2[%123, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %158 = addf %157, %156 {RelaxedPrecision} : f32
    store %158, %arg2[%123, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %159 = load %arg2[%123, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %159, %arg2[%123, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %160 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %161 = load %arg1[%11, %53] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %162 = mulf %160, %161 {RelaxedPrecision} : f32
    %163 = load %arg2[%123, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %164 = addf %163, %162 {RelaxedPrecision} : f32
    store %164, %arg2[%123, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %165 = load %arg2[%123, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %165, %arg2[%123, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %166 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %167 = load %arg1[%11, %60] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %168 = mulf %166, %167 {RelaxedPrecision} : f32
    %169 = load %arg2[%123, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %170 = addf %169, %168 {RelaxedPrecision} : f32
    store %170, %arg2[%123, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %171 = load %arg2[%123, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %171, %arg2[%123, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %172 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %173 = load %arg1[%11, %67] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %174 = mulf %172, %173 {RelaxedPrecision} : f32
    %175 = load %arg2[%123, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %176 = addf %175, %174 {RelaxedPrecision} : f32
    store %176, %arg2[%123, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %177 = load %arg2[%123, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %177, %arg2[%123, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %178 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %179 = load %arg1[%11, %74] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %180 = mulf %178, %179 {RelaxedPrecision} : f32
    %181 = load %arg2[%123, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %182 = addf %181, %180 {RelaxedPrecision} : f32
    store %182, %arg2[%123, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %183 = load %arg2[%123, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %183, %arg2[%123, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %184 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %185 = load %arg1[%11, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %186 = mulf %184, %185 {RelaxedPrecision} : f32
    %187 = load %arg2[%123, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %188 = addf %187, %186 {RelaxedPrecision} : f32
    store %188, %arg2[%123, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %189 = load %arg2[%123, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %189, %arg2[%123, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %190 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %191 = load %arg1[%11, %88] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %192 = mulf %190, %191 {RelaxedPrecision} : f32
    %193 = load %arg2[%123, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %194 = addf %193, %192 {RelaxedPrecision} : f32
    store %194, %arg2[%123, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %195 = load %arg2[%123, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %195, %arg2[%123, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %196 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %197 = load %arg1[%11, %95] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %198 = mulf %196, %197 {RelaxedPrecision} : f32
    %199 = load %arg2[%123, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %200 = addf %199, %198 {RelaxedPrecision} : f32
    store %200, %arg2[%123, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %201 = load %arg2[%123, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %201, %arg2[%123, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %202 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %203 = load %arg1[%11, %102] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %204 = mulf %202, %203 {RelaxedPrecision} : f32
    %205 = load %arg2[%123, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %206 = addf %205, %204 {RelaxedPrecision} : f32
    store %206, %arg2[%123, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %207 = load %arg2[%123, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %207, %arg2[%123, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %208 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %209 = load %arg1[%11, %109] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %210 = mulf %208, %209 {RelaxedPrecision} : f32
    %211 = load %arg2[%123, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %212 = addf %211, %210 {RelaxedPrecision} : f32
    store %212, %arg2[%123, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %213 = load %arg2[%123, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %213, %arg2[%123, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %214 = load %arg0[%123, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %215 = load %arg1[%11, %116] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %216 = mulf %214, %215 {RelaxedPrecision} : f32
    %217 = load %arg2[%123, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %218 = addf %217, %216 {RelaxedPrecision} : f32
    store %218, %arg2[%123, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %219 = load %arg2[%123, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %219, %arg2[%123, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %220 = addi %2, %c2 : index
    %221 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %222 = load %arg1[%11, %10] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %223 = mulf %221, %222 {RelaxedPrecision} : f32
    %224 = load %arg2[%220, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %225 = addf %224, %223 {RelaxedPrecision} : f32
    store %225, %arg2[%220, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %226 = load %arg2[%220, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %226, %arg2[%220, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %227 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %228 = load %arg1[%11, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %229 = mulf %227, %228 {RelaxedPrecision} : f32
    %230 = load %arg2[%220, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %231 = addf %230, %229 {RelaxedPrecision} : f32
    store %231, %arg2[%220, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %232 = load %arg2[%220, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %232, %arg2[%220, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %233 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %234 = load %arg1[%11, %25] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %235 = mulf %233, %234 {RelaxedPrecision} : f32
    %236 = load %arg2[%220, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %237 = addf %236, %235 {RelaxedPrecision} : f32
    store %237, %arg2[%220, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %238 = load %arg2[%220, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %238, %arg2[%220, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %239 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %240 = load %arg1[%11, %32] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %241 = mulf %239, %240 {RelaxedPrecision} : f32
    %242 = load %arg2[%220, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %243 = addf %242, %241 {RelaxedPrecision} : f32
    store %243, %arg2[%220, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %244 = load %arg2[%220, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %244, %arg2[%220, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %245 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %246 = load %arg1[%11, %39] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %247 = mulf %245, %246 {RelaxedPrecision} : f32
    %248 = load %arg2[%220, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %249 = addf %248, %247 {RelaxedPrecision} : f32
    store %249, %arg2[%220, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %250 = load %arg2[%220, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %250, %arg2[%220, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %251 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %252 = load %arg1[%11, %46] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %253 = mulf %251, %252 {RelaxedPrecision} : f32
    %254 = load %arg2[%220, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %255 = addf %254, %253 {RelaxedPrecision} : f32
    store %255, %arg2[%220, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %256 = load %arg2[%220, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %256, %arg2[%220, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %257 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %258 = load %arg1[%11, %53] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %259 = mulf %257, %258 {RelaxedPrecision} : f32
    %260 = load %arg2[%220, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %261 = addf %260, %259 {RelaxedPrecision} : f32
    store %261, %arg2[%220, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %262 = load %arg2[%220, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %262, %arg2[%220, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %263 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %264 = load %arg1[%11, %60] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %265 = mulf %263, %264 {RelaxedPrecision} : f32
    %266 = load %arg2[%220, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %267 = addf %266, %265 {RelaxedPrecision} : f32
    store %267, %arg2[%220, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %268 = load %arg2[%220, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %268, %arg2[%220, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %269 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %270 = load %arg1[%11, %67] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %271 = mulf %269, %270 {RelaxedPrecision} : f32
    %272 = load %arg2[%220, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %273 = addf %272, %271 {RelaxedPrecision} : f32
    store %273, %arg2[%220, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %274 = load %arg2[%220, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %274, %arg2[%220, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %275 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %276 = load %arg1[%11, %74] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %277 = mulf %275, %276 {RelaxedPrecision} : f32
    %278 = load %arg2[%220, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %279 = addf %278, %277 {RelaxedPrecision} : f32
    store %279, %arg2[%220, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %280 = load %arg2[%220, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %280, %arg2[%220, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %281 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %282 = load %arg1[%11, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %283 = mulf %281, %282 {RelaxedPrecision} : f32
    %284 = load %arg2[%220, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %285 = addf %284, %283 {RelaxedPrecision} : f32
    store %285, %arg2[%220, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %286 = load %arg2[%220, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %286, %arg2[%220, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %287 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %288 = load %arg1[%11, %88] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %289 = mulf %287, %288 {RelaxedPrecision} : f32
    %290 = load %arg2[%220, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %291 = addf %290, %289 {RelaxedPrecision} : f32
    store %291, %arg2[%220, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %292 = load %arg2[%220, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %292, %arg2[%220, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %293 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %294 = load %arg1[%11, %95] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %295 = mulf %293, %294 {RelaxedPrecision} : f32
    %296 = load %arg2[%220, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %297 = addf %296, %295 {RelaxedPrecision} : f32
    store %297, %arg2[%220, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %298 = load %arg2[%220, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %298, %arg2[%220, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %299 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %300 = load %arg1[%11, %102] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %301 = mulf %299, %300 {RelaxedPrecision} : f32
    %302 = load %arg2[%220, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %303 = addf %302, %301 {RelaxedPrecision} : f32
    store %303, %arg2[%220, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %304 = load %arg2[%220, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %304, %arg2[%220, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %305 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %306 = load %arg1[%11, %109] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %307 = mulf %305, %306 {RelaxedPrecision} : f32
    %308 = load %arg2[%220, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %309 = addf %308, %307 {RelaxedPrecision} : f32
    store %309, %arg2[%220, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %310 = load %arg2[%220, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %310, %arg2[%220, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %311 = load %arg0[%220, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %312 = load %arg1[%11, %116] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %313 = mulf %311, %312 {RelaxedPrecision} : f32
    %314 = load %arg2[%220, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %315 = addf %314, %313 {RelaxedPrecision} : f32
    store %315, %arg2[%220, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %316 = load %arg2[%220, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %316, %arg2[%220, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %317 = addi %2, %c3 : index
    %318 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %319 = load %arg1[%11, %10] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %320 = mulf %318, %319 {RelaxedPrecision} : f32
    %321 = load %arg2[%317, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %322 = addf %321, %320 {RelaxedPrecision} : f32
    store %322, %arg2[%317, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %323 = load %arg2[%317, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %323, %arg2[%317, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %324 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %325 = load %arg1[%11, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %326 = mulf %324, %325 {RelaxedPrecision} : f32
    %327 = load %arg2[%317, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %328 = addf %327, %326 {RelaxedPrecision} : f32
    store %328, %arg2[%317, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %329 = load %arg2[%317, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %329, %arg2[%317, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %330 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %331 = load %arg1[%11, %25] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %332 = mulf %330, %331 {RelaxedPrecision} : f32
    %333 = load %arg2[%317, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %334 = addf %333, %332 {RelaxedPrecision} : f32
    store %334, %arg2[%317, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %335 = load %arg2[%317, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %335, %arg2[%317, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %336 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %337 = load %arg1[%11, %32] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %338 = mulf %336, %337 {RelaxedPrecision} : f32
    %339 = load %arg2[%317, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %340 = addf %339, %338 {RelaxedPrecision} : f32
    store %340, %arg2[%317, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %341 = load %arg2[%317, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %341, %arg2[%317, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %342 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %343 = load %arg1[%11, %39] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %344 = mulf %342, %343 {RelaxedPrecision} : f32
    %345 = load %arg2[%317, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %346 = addf %345, %344 {RelaxedPrecision} : f32
    store %346, %arg2[%317, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %347 = load %arg2[%317, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %347, %arg2[%317, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %348 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %349 = load %arg1[%11, %46] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %350 = mulf %348, %349 {RelaxedPrecision} : f32
    %351 = load %arg2[%317, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %352 = addf %351, %350 {RelaxedPrecision} : f32
    store %352, %arg2[%317, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %353 = load %arg2[%317, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %353, %arg2[%317, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %354 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %355 = load %arg1[%11, %53] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %356 = mulf %354, %355 {RelaxedPrecision} : f32
    %357 = load %arg2[%317, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %358 = addf %357, %356 {RelaxedPrecision} : f32
    store %358, %arg2[%317, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %359 = load %arg2[%317, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %359, %arg2[%317, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %360 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %361 = load %arg1[%11, %60] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %362 = mulf %360, %361 {RelaxedPrecision} : f32
    %363 = load %arg2[%317, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %364 = addf %363, %362 {RelaxedPrecision} : f32
    store %364, %arg2[%317, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %365 = load %arg2[%317, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %365, %arg2[%317, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %366 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %367 = load %arg1[%11, %67] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %368 = mulf %366, %367 {RelaxedPrecision} : f32
    %369 = load %arg2[%317, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %370 = addf %369, %368 {RelaxedPrecision} : f32
    store %370, %arg2[%317, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %371 = load %arg2[%317, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %371, %arg2[%317, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %372 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %373 = load %arg1[%11, %74] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %374 = mulf %372, %373 {RelaxedPrecision} : f32
    %375 = load %arg2[%317, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %376 = addf %375, %374 {RelaxedPrecision} : f32
    store %376, %arg2[%317, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %377 = load %arg2[%317, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %377, %arg2[%317, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %378 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %379 = load %arg1[%11, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %380 = mulf %378, %379 {RelaxedPrecision} : f32
    %381 = load %arg2[%317, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %382 = addf %381, %380 {RelaxedPrecision} : f32
    store %382, %arg2[%317, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %383 = load %arg2[%317, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %383, %arg2[%317, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %384 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %385 = load %arg1[%11, %88] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %386 = mulf %384, %385 {RelaxedPrecision} : f32
    %387 = load %arg2[%317, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %388 = addf %387, %386 {RelaxedPrecision} : f32
    store %388, %arg2[%317, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %389 = load %arg2[%317, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %389, %arg2[%317, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %390 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %391 = load %arg1[%11, %95] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %392 = mulf %390, %391 {RelaxedPrecision} : f32
    %393 = load %arg2[%317, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %394 = addf %393, %392 {RelaxedPrecision} : f32
    store %394, %arg2[%317, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %395 = load %arg2[%317, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %395, %arg2[%317, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %396 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %397 = load %arg1[%11, %102] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %398 = mulf %396, %397 {RelaxedPrecision} : f32
    %399 = load %arg2[%317, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %400 = addf %399, %398 {RelaxedPrecision} : f32
    store %400, %arg2[%317, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %401 = load %arg2[%317, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %401, %arg2[%317, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %402 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %403 = load %arg1[%11, %109] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %404 = mulf %402, %403 {RelaxedPrecision} : f32
    %405 = load %arg2[%317, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %406 = addf %405, %404 {RelaxedPrecision} : f32
    store %406, %arg2[%317, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %407 = load %arg2[%317, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %407, %arg2[%317, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %408 = load %arg0[%317, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %409 = load %arg1[%11, %116] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %410 = mulf %408, %409 {RelaxedPrecision} : f32
    %411 = load %arg2[%317, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %412 = addf %411, %410 {RelaxedPrecision} : f32
    store %412, %arg2[%317, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %413 = load %arg2[%317, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %413, %arg2[%317, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %414 = addi %2, %c4 : index
    %415 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %416 = load %arg1[%11, %10] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %417 = mulf %415, %416 {RelaxedPrecision} : f32
    %418 = load %arg2[%414, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %419 = addf %418, %417 {RelaxedPrecision} : f32
    store %419, %arg2[%414, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %420 = load %arg2[%414, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %420, %arg2[%414, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %421 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %422 = load %arg1[%11, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %423 = mulf %421, %422 {RelaxedPrecision} : f32
    %424 = load %arg2[%414, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %425 = addf %424, %423 {RelaxedPrecision} : f32
    store %425, %arg2[%414, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %426 = load %arg2[%414, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %426, %arg2[%414, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %427 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %428 = load %arg1[%11, %25] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %429 = mulf %427, %428 {RelaxedPrecision} : f32
    %430 = load %arg2[%414, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %431 = addf %430, %429 {RelaxedPrecision} : f32
    store %431, %arg2[%414, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %432 = load %arg2[%414, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %432, %arg2[%414, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %433 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %434 = load %arg1[%11, %32] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %435 = mulf %433, %434 {RelaxedPrecision} : f32
    %436 = load %arg2[%414, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %437 = addf %436, %435 {RelaxedPrecision} : f32
    store %437, %arg2[%414, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %438 = load %arg2[%414, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %438, %arg2[%414, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %439 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %440 = load %arg1[%11, %39] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %441 = mulf %439, %440 {RelaxedPrecision} : f32
    %442 = load %arg2[%414, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %443 = addf %442, %441 {RelaxedPrecision} : f32
    store %443, %arg2[%414, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %444 = load %arg2[%414, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %444, %arg2[%414, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %445 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %446 = load %arg1[%11, %46] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %447 = mulf %445, %446 {RelaxedPrecision} : f32
    %448 = load %arg2[%414, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %449 = addf %448, %447 {RelaxedPrecision} : f32
    store %449, %arg2[%414, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %450 = load %arg2[%414, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %450, %arg2[%414, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %451 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %452 = load %arg1[%11, %53] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %453 = mulf %451, %452 {RelaxedPrecision} : f32
    %454 = load %arg2[%414, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %455 = addf %454, %453 {RelaxedPrecision} : f32
    store %455, %arg2[%414, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %456 = load %arg2[%414, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %456, %arg2[%414, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %457 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %458 = load %arg1[%11, %60] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %459 = mulf %457, %458 {RelaxedPrecision} : f32
    %460 = load %arg2[%414, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %461 = addf %460, %459 {RelaxedPrecision} : f32
    store %461, %arg2[%414, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %462 = load %arg2[%414, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %462, %arg2[%414, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %463 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %464 = load %arg1[%11, %67] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %465 = mulf %463, %464 {RelaxedPrecision} : f32
    %466 = load %arg2[%414, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %467 = addf %466, %465 {RelaxedPrecision} : f32
    store %467, %arg2[%414, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %468 = load %arg2[%414, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %468, %arg2[%414, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %469 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %470 = load %arg1[%11, %74] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %471 = mulf %469, %470 {RelaxedPrecision} : f32
    %472 = load %arg2[%414, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %473 = addf %472, %471 {RelaxedPrecision} : f32
    store %473, %arg2[%414, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %474 = load %arg2[%414, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %474, %arg2[%414, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %475 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %476 = load %arg1[%11, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %477 = mulf %475, %476 {RelaxedPrecision} : f32
    %478 = load %arg2[%414, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %479 = addf %478, %477 {RelaxedPrecision} : f32
    store %479, %arg2[%414, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %480 = load %arg2[%414, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %480, %arg2[%414, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %481 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %482 = load %arg1[%11, %88] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %483 = mulf %481, %482 {RelaxedPrecision} : f32
    %484 = load %arg2[%414, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %485 = addf %484, %483 {RelaxedPrecision} : f32
    store %485, %arg2[%414, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %486 = load %arg2[%414, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %486, %arg2[%414, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %487 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %488 = load %arg1[%11, %95] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %489 = mulf %487, %488 {RelaxedPrecision} : f32
    %490 = load %arg2[%414, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %491 = addf %490, %489 {RelaxedPrecision} : f32
    store %491, %arg2[%414, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %492 = load %arg2[%414, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %492, %arg2[%414, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %493 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %494 = load %arg1[%11, %102] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %495 = mulf %493, %494 {RelaxedPrecision} : f32
    %496 = load %arg2[%414, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %497 = addf %496, %495 {RelaxedPrecision} : f32
    store %497, %arg2[%414, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %498 = load %arg2[%414, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %498, %arg2[%414, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %499 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %500 = load %arg1[%11, %109] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %501 = mulf %499, %500 {RelaxedPrecision} : f32
    %502 = load %arg2[%414, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %503 = addf %502, %501 {RelaxedPrecision} : f32
    store %503, %arg2[%414, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %504 = load %arg2[%414, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %504, %arg2[%414, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %505 = load %arg0[%414, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %506 = load %arg1[%11, %116] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %507 = mulf %505, %506 {RelaxedPrecision} : f32
    %508 = load %arg2[%414, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %509 = addf %508, %507 {RelaxedPrecision} : f32
    store %509, %arg2[%414, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %510 = load %arg2[%414, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %510, %arg2[%414, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %511 = addi %2, %c5 : index
    %512 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %513 = load %arg1[%11, %10] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %514 = mulf %512, %513 {RelaxedPrecision} : f32
    %515 = load %arg2[%511, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %516 = addf %515, %514 {RelaxedPrecision} : f32
    store %516, %arg2[%511, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %517 = load %arg2[%511, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %517, %arg2[%511, %10] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %518 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %519 = load %arg1[%11, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %520 = mulf %518, %519 {RelaxedPrecision} : f32
    %521 = load %arg2[%511, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %522 = addf %521, %520 {RelaxedPrecision} : f32
    store %522, %arg2[%511, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %523 = load %arg2[%511, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %523, %arg2[%511, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %524 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %525 = load %arg1[%11, %25] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %526 = mulf %524, %525 {RelaxedPrecision} : f32
    %527 = load %arg2[%511, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %528 = addf %527, %526 {RelaxedPrecision} : f32
    store %528, %arg2[%511, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %529 = load %arg2[%511, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %529, %arg2[%511, %25] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %530 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %531 = load %arg1[%11, %32] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %532 = mulf %530, %531 {RelaxedPrecision} : f32
    %533 = load %arg2[%511, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %534 = addf %533, %532 {RelaxedPrecision} : f32
    store %534, %arg2[%511, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %535 = load %arg2[%511, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %535, %arg2[%511, %32] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %536 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %537 = load %arg1[%11, %39] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %538 = mulf %536, %537 {RelaxedPrecision} : f32
    %539 = load %arg2[%511, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %540 = addf %539, %538 {RelaxedPrecision} : f32
    store %540, %arg2[%511, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %541 = load %arg2[%511, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %541, %arg2[%511, %39] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %542 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %543 = load %arg1[%11, %46] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %544 = mulf %542, %543 {RelaxedPrecision} : f32
    %545 = load %arg2[%511, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %546 = addf %545, %544 {RelaxedPrecision} : f32
    store %546, %arg2[%511, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %547 = load %arg2[%511, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %547, %arg2[%511, %46] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %548 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %549 = load %arg1[%11, %53] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %550 = mulf %548, %549 {RelaxedPrecision} : f32
    %551 = load %arg2[%511, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %552 = addf %551, %550 {RelaxedPrecision} : f32
    store %552, %arg2[%511, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %553 = load %arg2[%511, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %553, %arg2[%511, %53] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %554 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %555 = load %arg1[%11, %60] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %556 = mulf %554, %555 {RelaxedPrecision} : f32
    %557 = load %arg2[%511, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %558 = addf %557, %556 {RelaxedPrecision} : f32
    store %558, %arg2[%511, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %559 = load %arg2[%511, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %559, %arg2[%511, %60] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %560 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %561 = load %arg1[%11, %67] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %562 = mulf %560, %561 {RelaxedPrecision} : f32
    %563 = load %arg2[%511, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %564 = addf %563, %562 {RelaxedPrecision} : f32
    store %564, %arg2[%511, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %565 = load %arg2[%511, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %565, %arg2[%511, %67] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %566 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %567 = load %arg1[%11, %74] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %568 = mulf %566, %567 {RelaxedPrecision} : f32
    %569 = load %arg2[%511, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %570 = addf %569, %568 {RelaxedPrecision} : f32
    store %570, %arg2[%511, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %571 = load %arg2[%511, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %571, %arg2[%511, %74] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %572 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %573 = load %arg1[%11, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %574 = mulf %572, %573 {RelaxedPrecision} : f32
    %575 = load %arg2[%511, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %576 = addf %575, %574 {RelaxedPrecision} : f32
    store %576, %arg2[%511, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %577 = load %arg2[%511, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %577, %arg2[%511, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %578 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %579 = load %arg1[%11, %88] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %580 = mulf %578, %579 {RelaxedPrecision} : f32
    %581 = load %arg2[%511, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %582 = addf %581, %580 {RelaxedPrecision} : f32
    store %582, %arg2[%511, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %583 = load %arg2[%511, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %583, %arg2[%511, %88] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %584 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %585 = load %arg1[%11, %95] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %586 = mulf %584, %585 {RelaxedPrecision} : f32
    %587 = load %arg2[%511, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %588 = addf %587, %586 {RelaxedPrecision} : f32
    store %588, %arg2[%511, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %589 = load %arg2[%511, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %589, %arg2[%511, %95] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %590 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %591 = load %arg1[%11, %102] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %592 = mulf %590, %591 {RelaxedPrecision} : f32
    %593 = load %arg2[%511, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %594 = addf %593, %592 {RelaxedPrecision} : f32
    store %594, %arg2[%511, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %595 = load %arg2[%511, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %595, %arg2[%511, %102] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %596 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %597 = load %arg1[%11, %109] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %598 = mulf %596, %597 {RelaxedPrecision} : f32
    %599 = load %arg2[%511, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %600 = addf %599, %598 {RelaxedPrecision} : f32
    store %600, %arg2[%511, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %601 = load %arg2[%511, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %601, %arg2[%511, %109] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %602 = load %arg0[%511, %11] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %603 = load %arg1[%11, %116] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %604 = mulf %602, %603 {RelaxedPrecision} : f32
    %605 = load %arg2[%511, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %606 = addf %605, %604 {RelaxedPrecision} : f32
    store %606, %arg2[%511, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %607 = load %arg2[%511, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %607, %arg2[%511, %116] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %608 = addi %8, %c1 : index
    br ^bb9(%608 : index)
  ^bb11:  // pred: ^bb9
    %609 = addi %6, %c4 : index
    br ^bb7(%609 : index)
  ^bb12:  // pred: ^bb7
    %610 = addi %4, %c16 : index
    br ^bb5(%610 : index)
  ^bb13:  // pred: ^bb5
    %611 = addi %2, %c6 : index
    br ^bb3(%611 : index)
  ^bb14:  // pred: ^bb3
    br ^bb15(%c0 : index)
  ^bb15(%612: index):  // 2 preds: ^bb14, ^bb22
    %613 = cmpi "slt", %612, %c256 : index
    cond_br %613, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    br ^bb17(%c0 : index)
  ^bb17(%614: index):  // 2 preds: ^bb16, ^bb21
    %615 = cmpi "slt", %614, %c128 : index
    cond_br %615, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    br ^bb19(%c0 : index)
  ^bb19(%616: index):  // 2 preds: ^bb18, ^bb20
    %617 = cmpi "slt", %616, %c4 : index
    cond_br %617, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %618 = addi %0, %612 : index
    %619 = addi %614, %616 : index
    %620 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %621 = load %arg1[%619, %618] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %622 = mulf %620, %621 {RelaxedPrecision} : f32
    %623 = load %arg2[%c780, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %624 = addf %623, %622 {RelaxedPrecision} : f32
    store %624, %arg2[%c780, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %625 = load %arg2[%c780, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %625, %arg2[%c780, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %626 = addi %618, %c1 : index
    %627 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %628 = load %arg1[%619, %626] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %629 = mulf %627, %628 {RelaxedPrecision} : f32
    %630 = load %arg2[%c780, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %631 = addf %630, %629 {RelaxedPrecision} : f32
    store %631, %arg2[%c780, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %632 = load %arg2[%c780, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %632, %arg2[%c780, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %633 = addi %618, %c2 : index
    %634 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %635 = load %arg1[%619, %633] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %636 = mulf %634, %635 {RelaxedPrecision} : f32
    %637 = load %arg2[%c780, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %638 = addf %637, %636 {RelaxedPrecision} : f32
    store %638, %arg2[%c780, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %639 = load %arg2[%c780, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %639, %arg2[%c780, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %640 = addi %618, %c3 : index
    %641 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %642 = load %arg1[%619, %640] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %643 = mulf %641, %642 {RelaxedPrecision} : f32
    %644 = load %arg2[%c780, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %645 = addf %644, %643 {RelaxedPrecision} : f32
    store %645, %arg2[%c780, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %646 = load %arg2[%c780, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %646, %arg2[%c780, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %647 = addi %618, %c4 : index
    %648 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %649 = load %arg1[%619, %647] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %650 = mulf %648, %649 {RelaxedPrecision} : f32
    %651 = load %arg2[%c780, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %652 = addf %651, %650 {RelaxedPrecision} : f32
    store %652, %arg2[%c780, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %653 = load %arg2[%c780, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %653, %arg2[%c780, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %654 = addi %618, %c5 : index
    %655 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %656 = load %arg1[%619, %654] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %657 = mulf %655, %656 {RelaxedPrecision} : f32
    %658 = load %arg2[%c780, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %659 = addf %658, %657 {RelaxedPrecision} : f32
    store %659, %arg2[%c780, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %660 = load %arg2[%c780, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %660, %arg2[%c780, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %661 = addi %618, %c6 : index
    %662 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %663 = load %arg1[%619, %661] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %664 = mulf %662, %663 {RelaxedPrecision} : f32
    %665 = load %arg2[%c780, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %666 = addf %665, %664 {RelaxedPrecision} : f32
    store %666, %arg2[%c780, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %667 = load %arg2[%c780, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %667, %arg2[%c780, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %668 = addi %618, %c7 : index
    %669 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %670 = load %arg1[%619, %668] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %671 = mulf %669, %670 {RelaxedPrecision} : f32
    %672 = load %arg2[%c780, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %673 = addf %672, %671 {RelaxedPrecision} : f32
    store %673, %arg2[%c780, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %674 = load %arg2[%c780, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %674, %arg2[%c780, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %675 = addi %618, %c8 : index
    %676 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %677 = load %arg1[%619, %675] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %678 = mulf %676, %677 {RelaxedPrecision} : f32
    %679 = load %arg2[%c780, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %680 = addf %679, %678 {RelaxedPrecision} : f32
    store %680, %arg2[%c780, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %681 = load %arg2[%c780, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %681, %arg2[%c780, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %682 = addi %618, %c9 : index
    %683 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %684 = load %arg1[%619, %682] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %685 = mulf %683, %684 {RelaxedPrecision} : f32
    %686 = load %arg2[%c780, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %687 = addf %686, %685 {RelaxedPrecision} : f32
    store %687, %arg2[%c780, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %688 = load %arg2[%c780, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %688, %arg2[%c780, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %689 = addi %618, %c10 : index
    %690 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %691 = load %arg1[%619, %689] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %692 = mulf %690, %691 {RelaxedPrecision} : f32
    %693 = load %arg2[%c780, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %694 = addf %693, %692 {RelaxedPrecision} : f32
    store %694, %arg2[%c780, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %695 = load %arg2[%c780, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %695, %arg2[%c780, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %696 = addi %618, %c11 : index
    %697 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %698 = load %arg1[%619, %696] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %699 = mulf %697, %698 {RelaxedPrecision} : f32
    %700 = load %arg2[%c780, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %701 = addf %700, %699 {RelaxedPrecision} : f32
    store %701, %arg2[%c780, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %702 = load %arg2[%c780, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %702, %arg2[%c780, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %703 = addi %618, %c12 : index
    %704 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %705 = load %arg1[%619, %703] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %706 = mulf %704, %705 {RelaxedPrecision} : f32
    %707 = load %arg2[%c780, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %708 = addf %707, %706 {RelaxedPrecision} : f32
    store %708, %arg2[%c780, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %709 = load %arg2[%c780, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %709, %arg2[%c780, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %710 = addi %618, %c13 : index
    %711 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %712 = load %arg1[%619, %710] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %713 = mulf %711, %712 {RelaxedPrecision} : f32
    %714 = load %arg2[%c780, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %715 = addf %714, %713 {RelaxedPrecision} : f32
    store %715, %arg2[%c780, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %716 = load %arg2[%c780, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %716, %arg2[%c780, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %717 = addi %618, %c14 : index
    %718 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %719 = load %arg1[%619, %717] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %720 = mulf %718, %719 {RelaxedPrecision} : f32
    %721 = load %arg2[%c780, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %722 = addf %721, %720 {RelaxedPrecision} : f32
    store %722, %arg2[%c780, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %723 = load %arg2[%c780, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %723, %arg2[%c780, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %724 = addi %618, %c15 : index
    %725 = load %arg0[%c780, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %726 = load %arg1[%619, %724] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %727 = mulf %725, %726 {RelaxedPrecision} : f32
    %728 = load %arg2[%c780, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %729 = addf %728, %727 {RelaxedPrecision} : f32
    store %729, %arg2[%c780, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %730 = load %arg2[%c780, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %730, %arg2[%c780, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %731 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %732 = load %arg1[%619, %618] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %733 = mulf %731, %732 {RelaxedPrecision} : f32
    %734 = load %arg2[%c781, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %735 = addf %734, %733 {RelaxedPrecision} : f32
    store %735, %arg2[%c781, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %736 = load %arg2[%c781, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %736, %arg2[%c781, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %737 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %738 = load %arg1[%619, %626] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %739 = mulf %737, %738 {RelaxedPrecision} : f32
    %740 = load %arg2[%c781, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %741 = addf %740, %739 {RelaxedPrecision} : f32
    store %741, %arg2[%c781, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %742 = load %arg2[%c781, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %742, %arg2[%c781, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %743 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %744 = load %arg1[%619, %633] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %745 = mulf %743, %744 {RelaxedPrecision} : f32
    %746 = load %arg2[%c781, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %747 = addf %746, %745 {RelaxedPrecision} : f32
    store %747, %arg2[%c781, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %748 = load %arg2[%c781, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %748, %arg2[%c781, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %749 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %750 = load %arg1[%619, %640] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %751 = mulf %749, %750 {RelaxedPrecision} : f32
    %752 = load %arg2[%c781, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %753 = addf %752, %751 {RelaxedPrecision} : f32
    store %753, %arg2[%c781, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %754 = load %arg2[%c781, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %754, %arg2[%c781, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %755 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %756 = load %arg1[%619, %647] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %757 = mulf %755, %756 {RelaxedPrecision} : f32
    %758 = load %arg2[%c781, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %759 = addf %758, %757 {RelaxedPrecision} : f32
    store %759, %arg2[%c781, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %760 = load %arg2[%c781, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %760, %arg2[%c781, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %761 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %762 = load %arg1[%619, %654] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %763 = mulf %761, %762 {RelaxedPrecision} : f32
    %764 = load %arg2[%c781, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %765 = addf %764, %763 {RelaxedPrecision} : f32
    store %765, %arg2[%c781, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %766 = load %arg2[%c781, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %766, %arg2[%c781, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %767 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %768 = load %arg1[%619, %661] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %769 = mulf %767, %768 {RelaxedPrecision} : f32
    %770 = load %arg2[%c781, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %771 = addf %770, %769 {RelaxedPrecision} : f32
    store %771, %arg2[%c781, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %772 = load %arg2[%c781, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %772, %arg2[%c781, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %773 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %774 = load %arg1[%619, %668] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %775 = mulf %773, %774 {RelaxedPrecision} : f32
    %776 = load %arg2[%c781, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %777 = addf %776, %775 {RelaxedPrecision} : f32
    store %777, %arg2[%c781, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %778 = load %arg2[%c781, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %778, %arg2[%c781, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %779 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %780 = load %arg1[%619, %675] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %781 = mulf %779, %780 {RelaxedPrecision} : f32
    %782 = load %arg2[%c781, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %783 = addf %782, %781 {RelaxedPrecision} : f32
    store %783, %arg2[%c781, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %784 = load %arg2[%c781, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %784, %arg2[%c781, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %785 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %786 = load %arg1[%619, %682] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %787 = mulf %785, %786 {RelaxedPrecision} : f32
    %788 = load %arg2[%c781, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %789 = addf %788, %787 {RelaxedPrecision} : f32
    store %789, %arg2[%c781, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %790 = load %arg2[%c781, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %790, %arg2[%c781, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %791 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %792 = load %arg1[%619, %689] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %793 = mulf %791, %792 {RelaxedPrecision} : f32
    %794 = load %arg2[%c781, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %795 = addf %794, %793 {RelaxedPrecision} : f32
    store %795, %arg2[%c781, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %796 = load %arg2[%c781, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %796, %arg2[%c781, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %797 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %798 = load %arg1[%619, %696] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %799 = mulf %797, %798 {RelaxedPrecision} : f32
    %800 = load %arg2[%c781, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %801 = addf %800, %799 {RelaxedPrecision} : f32
    store %801, %arg2[%c781, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %802 = load %arg2[%c781, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %802, %arg2[%c781, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %803 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %804 = load %arg1[%619, %703] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %805 = mulf %803, %804 {RelaxedPrecision} : f32
    %806 = load %arg2[%c781, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %807 = addf %806, %805 {RelaxedPrecision} : f32
    store %807, %arg2[%c781, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %808 = load %arg2[%c781, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %808, %arg2[%c781, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %809 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %810 = load %arg1[%619, %710] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %811 = mulf %809, %810 {RelaxedPrecision} : f32
    %812 = load %arg2[%c781, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %813 = addf %812, %811 {RelaxedPrecision} : f32
    store %813, %arg2[%c781, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %814 = load %arg2[%c781, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %814, %arg2[%c781, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %815 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %816 = load %arg1[%619, %717] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %817 = mulf %815, %816 {RelaxedPrecision} : f32
    %818 = load %arg2[%c781, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %819 = addf %818, %817 {RelaxedPrecision} : f32
    store %819, %arg2[%c781, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %820 = load %arg2[%c781, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %820, %arg2[%c781, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %821 = load %arg0[%c781, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %822 = load %arg1[%619, %724] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %823 = mulf %821, %822 {RelaxedPrecision} : f32
    %824 = load %arg2[%c781, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %825 = addf %824, %823 {RelaxedPrecision} : f32
    store %825, %arg2[%c781, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %826 = load %arg2[%c781, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %826, %arg2[%c781, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %827 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %828 = load %arg1[%619, %618] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %829 = mulf %827, %828 {RelaxedPrecision} : f32
    %830 = load %arg2[%c782, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %831 = addf %830, %829 {RelaxedPrecision} : f32
    store %831, %arg2[%c782, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %832 = load %arg2[%c782, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %832, %arg2[%c782, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %833 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %834 = load %arg1[%619, %626] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %835 = mulf %833, %834 {RelaxedPrecision} : f32
    %836 = load %arg2[%c782, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %837 = addf %836, %835 {RelaxedPrecision} : f32
    store %837, %arg2[%c782, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %838 = load %arg2[%c782, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %838, %arg2[%c782, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %839 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %840 = load %arg1[%619, %633] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %841 = mulf %839, %840 {RelaxedPrecision} : f32
    %842 = load %arg2[%c782, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %843 = addf %842, %841 {RelaxedPrecision} : f32
    store %843, %arg2[%c782, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %844 = load %arg2[%c782, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %844, %arg2[%c782, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %845 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %846 = load %arg1[%619, %640] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %847 = mulf %845, %846 {RelaxedPrecision} : f32
    %848 = load %arg2[%c782, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %849 = addf %848, %847 {RelaxedPrecision} : f32
    store %849, %arg2[%c782, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %850 = load %arg2[%c782, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %850, %arg2[%c782, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %851 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %852 = load %arg1[%619, %647] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %853 = mulf %851, %852 {RelaxedPrecision} : f32
    %854 = load %arg2[%c782, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %855 = addf %854, %853 {RelaxedPrecision} : f32
    store %855, %arg2[%c782, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %856 = load %arg2[%c782, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %856, %arg2[%c782, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %857 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %858 = load %arg1[%619, %654] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %859 = mulf %857, %858 {RelaxedPrecision} : f32
    %860 = load %arg2[%c782, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %861 = addf %860, %859 {RelaxedPrecision} : f32
    store %861, %arg2[%c782, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %862 = load %arg2[%c782, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %862, %arg2[%c782, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %863 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %864 = load %arg1[%619, %661] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %865 = mulf %863, %864 {RelaxedPrecision} : f32
    %866 = load %arg2[%c782, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %867 = addf %866, %865 {RelaxedPrecision} : f32
    store %867, %arg2[%c782, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %868 = load %arg2[%c782, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %868, %arg2[%c782, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %869 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %870 = load %arg1[%619, %668] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %871 = mulf %869, %870 {RelaxedPrecision} : f32
    %872 = load %arg2[%c782, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %873 = addf %872, %871 {RelaxedPrecision} : f32
    store %873, %arg2[%c782, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %874 = load %arg2[%c782, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %874, %arg2[%c782, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %875 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %876 = load %arg1[%619, %675] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %877 = mulf %875, %876 {RelaxedPrecision} : f32
    %878 = load %arg2[%c782, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %879 = addf %878, %877 {RelaxedPrecision} : f32
    store %879, %arg2[%c782, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %880 = load %arg2[%c782, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %880, %arg2[%c782, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %881 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %882 = load %arg1[%619, %682] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %883 = mulf %881, %882 {RelaxedPrecision} : f32
    %884 = load %arg2[%c782, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %885 = addf %884, %883 {RelaxedPrecision} : f32
    store %885, %arg2[%c782, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %886 = load %arg2[%c782, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %886, %arg2[%c782, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %887 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %888 = load %arg1[%619, %689] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %889 = mulf %887, %888 {RelaxedPrecision} : f32
    %890 = load %arg2[%c782, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %891 = addf %890, %889 {RelaxedPrecision} : f32
    store %891, %arg2[%c782, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %892 = load %arg2[%c782, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %892, %arg2[%c782, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %893 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %894 = load %arg1[%619, %696] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %895 = mulf %893, %894 {RelaxedPrecision} : f32
    %896 = load %arg2[%c782, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %897 = addf %896, %895 {RelaxedPrecision} : f32
    store %897, %arg2[%c782, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %898 = load %arg2[%c782, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %898, %arg2[%c782, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %899 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %900 = load %arg1[%619, %703] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %901 = mulf %899, %900 {RelaxedPrecision} : f32
    %902 = load %arg2[%c782, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %903 = addf %902, %901 {RelaxedPrecision} : f32
    store %903, %arg2[%c782, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %904 = load %arg2[%c782, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %904, %arg2[%c782, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %905 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %906 = load %arg1[%619, %710] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %907 = mulf %905, %906 {RelaxedPrecision} : f32
    %908 = load %arg2[%c782, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %909 = addf %908, %907 {RelaxedPrecision} : f32
    store %909, %arg2[%c782, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %910 = load %arg2[%c782, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %910, %arg2[%c782, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %911 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %912 = load %arg1[%619, %717] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %913 = mulf %911, %912 {RelaxedPrecision} : f32
    %914 = load %arg2[%c782, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %915 = addf %914, %913 {RelaxedPrecision} : f32
    store %915, %arg2[%c782, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %916 = load %arg2[%c782, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %916, %arg2[%c782, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %917 = load %arg0[%c782, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %918 = load %arg1[%619, %724] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %919 = mulf %917, %918 {RelaxedPrecision} : f32
    %920 = load %arg2[%c782, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %921 = addf %920, %919 {RelaxedPrecision} : f32
    store %921, %arg2[%c782, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %922 = load %arg2[%c782, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %922, %arg2[%c782, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %923 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %924 = load %arg1[%619, %618] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %925 = mulf %923, %924 {RelaxedPrecision} : f32
    %926 = load %arg2[%c783, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %927 = addf %926, %925 {RelaxedPrecision} : f32
    store %927, %arg2[%c783, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %928 = load %arg2[%c783, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %928, %arg2[%c783, %618] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %929 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %930 = load %arg1[%619, %626] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %931 = mulf %929, %930 {RelaxedPrecision} : f32
    %932 = load %arg2[%c783, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %933 = addf %932, %931 {RelaxedPrecision} : f32
    store %933, %arg2[%c783, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %934 = load %arg2[%c783, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %934, %arg2[%c783, %626] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %935 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %936 = load %arg1[%619, %633] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %937 = mulf %935, %936 {RelaxedPrecision} : f32
    %938 = load %arg2[%c783, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %939 = addf %938, %937 {RelaxedPrecision} : f32
    store %939, %arg2[%c783, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %940 = load %arg2[%c783, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %940, %arg2[%c783, %633] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %941 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %942 = load %arg1[%619, %640] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %943 = mulf %941, %942 {RelaxedPrecision} : f32
    %944 = load %arg2[%c783, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %945 = addf %944, %943 {RelaxedPrecision} : f32
    store %945, %arg2[%c783, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %946 = load %arg2[%c783, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %946, %arg2[%c783, %640] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %947 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %948 = load %arg1[%619, %647] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %949 = mulf %947, %948 {RelaxedPrecision} : f32
    %950 = load %arg2[%c783, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %951 = addf %950, %949 {RelaxedPrecision} : f32
    store %951, %arg2[%c783, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %952 = load %arg2[%c783, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %952, %arg2[%c783, %647] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %953 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %954 = load %arg1[%619, %654] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %955 = mulf %953, %954 {RelaxedPrecision} : f32
    %956 = load %arg2[%c783, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %957 = addf %956, %955 {RelaxedPrecision} : f32
    store %957, %arg2[%c783, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %958 = load %arg2[%c783, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %958, %arg2[%c783, %654] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %959 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %960 = load %arg1[%619, %661] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %961 = mulf %959, %960 {RelaxedPrecision} : f32
    %962 = load %arg2[%c783, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %963 = addf %962, %961 {RelaxedPrecision} : f32
    store %963, %arg2[%c783, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %964 = load %arg2[%c783, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %964, %arg2[%c783, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %965 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %966 = load %arg1[%619, %668] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %967 = mulf %965, %966 {RelaxedPrecision} : f32
    %968 = load %arg2[%c783, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %969 = addf %968, %967 {RelaxedPrecision} : f32
    store %969, %arg2[%c783, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %970 = load %arg2[%c783, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %970, %arg2[%c783, %668] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %971 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %972 = load %arg1[%619, %675] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %973 = mulf %971, %972 {RelaxedPrecision} : f32
    %974 = load %arg2[%c783, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %975 = addf %974, %973 {RelaxedPrecision} : f32
    store %975, %arg2[%c783, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %976 = load %arg2[%c783, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %976, %arg2[%c783, %675] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %977 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %978 = load %arg1[%619, %682] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %979 = mulf %977, %978 {RelaxedPrecision} : f32
    %980 = load %arg2[%c783, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %981 = addf %980, %979 {RelaxedPrecision} : f32
    store %981, %arg2[%c783, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %982 = load %arg2[%c783, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %982, %arg2[%c783, %682] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %983 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %984 = load %arg1[%619, %689] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %985 = mulf %983, %984 {RelaxedPrecision} : f32
    %986 = load %arg2[%c783, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %987 = addf %986, %985 {RelaxedPrecision} : f32
    store %987, %arg2[%c783, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %988 = load %arg2[%c783, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %988, %arg2[%c783, %689] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %989 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %990 = load %arg1[%619, %696] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %991 = mulf %989, %990 {RelaxedPrecision} : f32
    %992 = load %arg2[%c783, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %993 = addf %992, %991 {RelaxedPrecision} : f32
    store %993, %arg2[%c783, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %994 = load %arg2[%c783, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %994, %arg2[%c783, %696] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %995 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %996 = load %arg1[%619, %703] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %997 = mulf %995, %996 {RelaxedPrecision} : f32
    %998 = load %arg2[%c783, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %999 = addf %998, %997 {RelaxedPrecision} : f32
    store %999, %arg2[%c783, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1000 = load %arg2[%c783, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %1000, %arg2[%c783, %703] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1001 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1002 = load %arg1[%619, %710] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1003 = mulf %1001, %1002 {RelaxedPrecision} : f32
    %1004 = load %arg2[%c783, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1005 = addf %1004, %1003 {RelaxedPrecision} : f32
    store %1005, %arg2[%c783, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1006 = load %arg2[%c783, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %1006, %arg2[%c783, %710] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1007 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1008 = load %arg1[%619, %717] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1009 = mulf %1007, %1008 {RelaxedPrecision} : f32
    %1010 = load %arg2[%c783, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1011 = addf %1010, %1009 {RelaxedPrecision} : f32
    store %1011, %arg2[%c783, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1012 = load %arg2[%c783, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %1012, %arg2[%c783, %717] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1013 = load %arg0[%c783, %619] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1014 = load %arg1[%619, %724] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1015 = mulf %1013, %1014 {RelaxedPrecision} : f32
    %1016 = load %arg2[%c783, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1017 = addf %1016, %1015 {RelaxedPrecision} : f32
    store %1017, %arg2[%c783, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1018 = load %arg2[%c783, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    store %1018, %arg2[%c783, %724] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1019 = addi %616, %c1 : index
    br ^bb19(%1019 : index)
  ^bb21:  // pred: ^bb19
    %1020 = addi %614, %c4 : index
    br ^bb17(%1020 : index)
  ^bb22:  // pred: ^bb17
    %1021 = addi %612, %c16 : index
    br ^bb15(%1021 : index)
  ^bb23:  // pred: ^bb15
    %1022 = addi %0, %c256 : index
    br ^bb1(%1022 : index)
  ^bb24:  // pred: ^bb1
    return
  }
  func @optimized_matmul_py(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, accv.emit_header_decl, accv.emit_raw_pointer_api} {
    call @optimized_matmul_py_impl_17630232307017152746(%arg0, %arg1, %arg2) : (memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) -> ()
    return
  }
}
