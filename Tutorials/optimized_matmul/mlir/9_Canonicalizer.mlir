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
              %8 = addi %arg3, %arg5 : index
              %9 = addi %8, %c1 : index
              %10 = addi %arg6, %arg7 : index
              %11 = load %arg0[%arg4, %10] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %12 = load %arg1[%10, %9] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %13 = mulf %11, %12 {RelaxedPrecision} : f32
              %14 = load %arg2[%arg4, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %15 = addf %14, %13 {RelaxedPrecision} : f32
              store %15, %arg2[%arg4, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %16 = load %arg2[%arg4, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %16, %arg2[%arg4, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %17 = addi %arg3, %arg5 : index
              %18 = addi %17, %c2 : index
              %19 = addi %arg6, %arg7 : index
              %20 = load %arg0[%arg4, %19] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %21 = load %arg1[%19, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %22 = mulf %20, %21 {RelaxedPrecision} : f32
              %23 = load %arg2[%arg4, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %24 = addf %23, %22 {RelaxedPrecision} : f32
              store %24, %arg2[%arg4, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %25 = load %arg2[%arg4, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %25, %arg2[%arg4, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %26 = addi %arg3, %arg5 : index
              %27 = addi %26, %c3 : index
              %28 = addi %arg6, %arg7 : index
              %29 = load %arg0[%arg4, %28] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %30 = load %arg1[%28, %27] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %31 = mulf %29, %30 {RelaxedPrecision} : f32
              %32 = load %arg2[%arg4, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %33 = addf %32, %31 {RelaxedPrecision} : f32
              store %33, %arg2[%arg4, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %34 = load %arg2[%arg4, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %34, %arg2[%arg4, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %35 = addi %arg3, %arg5 : index
              %36 = addi %35, %c4 : index
              %37 = addi %arg6, %arg7 : index
              %38 = load %arg0[%arg4, %37] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %39 = load %arg1[%37, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %40 = mulf %38, %39 {RelaxedPrecision} : f32
              %41 = load %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %42 = addf %41, %40 {RelaxedPrecision} : f32
              store %42, %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %43 = load %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %43, %arg2[%arg4, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %44 = addi %arg3, %arg5 : index
              %45 = addi %44, %c5 : index
              %46 = addi %arg6, %arg7 : index
              %47 = load %arg0[%arg4, %46] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %48 = load %arg1[%46, %45] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %49 = mulf %47, %48 {RelaxedPrecision} : f32
              %50 = load %arg2[%arg4, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %51 = addf %50, %49 {RelaxedPrecision} : f32
              store %51, %arg2[%arg4, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %52 = load %arg2[%arg4, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %52, %arg2[%arg4, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %53 = addi %arg3, %arg5 : index
              %54 = addi %53, %c6 : index
              %55 = addi %arg6, %arg7 : index
              %56 = load %arg0[%arg4, %55] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %57 = load %arg1[%55, %54] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %58 = mulf %56, %57 {RelaxedPrecision} : f32
              %59 = load %arg2[%arg4, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %60 = addf %59, %58 {RelaxedPrecision} : f32
              store %60, %arg2[%arg4, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %61 = load %arg2[%arg4, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %61, %arg2[%arg4, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %62 = addi %arg3, %arg5 : index
              %63 = addi %62, %c7 : index
              %64 = addi %arg6, %arg7 : index
              %65 = load %arg0[%arg4, %64] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %66 = load %arg1[%64, %63] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %67 = mulf %65, %66 {RelaxedPrecision} : f32
              %68 = load %arg2[%arg4, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %69 = addf %68, %67 {RelaxedPrecision} : f32
              store %69, %arg2[%arg4, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %70 = load %arg2[%arg4, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %70, %arg2[%arg4, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %71 = addi %arg3, %arg5 : index
              %72 = addi %71, %c8 : index
              %73 = addi %arg6, %arg7 : index
              %74 = load %arg0[%arg4, %73] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %75 = load %arg1[%73, %72] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %76 = mulf %74, %75 {RelaxedPrecision} : f32
              %77 = load %arg2[%arg4, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %78 = addf %77, %76 {RelaxedPrecision} : f32
              store %78, %arg2[%arg4, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %79 = load %arg2[%arg4, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %79, %arg2[%arg4, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %80 = addi %arg3, %arg5 : index
              %81 = addi %80, %c9 : index
              %82 = addi %arg6, %arg7 : index
              %83 = load %arg0[%arg4, %82] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %84 = load %arg1[%82, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %85 = mulf %83, %84 {RelaxedPrecision} : f32
              %86 = load %arg2[%arg4, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %87 = addf %86, %85 {RelaxedPrecision} : f32
              store %87, %arg2[%arg4, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %88 = load %arg2[%arg4, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %88, %arg2[%arg4, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %89 = addi %arg3, %arg5 : index
              %90 = addi %89, %c10 : index
              %91 = addi %arg6, %arg7 : index
              %92 = load %arg0[%arg4, %91] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %93 = load %arg1[%91, %90] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %94 = mulf %92, %93 {RelaxedPrecision} : f32
              %95 = load %arg2[%arg4, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %96 = addf %95, %94 {RelaxedPrecision} : f32
              store %96, %arg2[%arg4, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %97 = load %arg2[%arg4, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %97, %arg2[%arg4, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %98 = addi %arg3, %arg5 : index
              %99 = addi %98, %c11 : index
              %100 = addi %arg6, %arg7 : index
              %101 = load %arg0[%arg4, %100] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %102 = load %arg1[%100, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %103 = mulf %101, %102 {RelaxedPrecision} : f32
              %104 = load %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %105 = addf %104, %103 {RelaxedPrecision} : f32
              store %105, %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %106 = load %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %106, %arg2[%arg4, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %107 = addi %arg3, %arg5 : index
              %108 = addi %107, %c12 : index
              %109 = addi %arg6, %arg7 : index
              %110 = load %arg0[%arg4, %109] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %111 = load %arg1[%109, %108] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %112 = mulf %110, %111 {RelaxedPrecision} : f32
              %113 = load %arg2[%arg4, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %114 = addf %113, %112 {RelaxedPrecision} : f32
              store %114, %arg2[%arg4, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %115 = load %arg2[%arg4, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %115, %arg2[%arg4, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %116 = addi %arg3, %arg5 : index
              %117 = addi %116, %c13 : index
              %118 = addi %arg6, %arg7 : index
              %119 = load %arg0[%arg4, %118] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %120 = load %arg1[%118, %117] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %121 = mulf %119, %120 {RelaxedPrecision} : f32
              %122 = load %arg2[%arg4, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %123 = addf %122, %121 {RelaxedPrecision} : f32
              store %123, %arg2[%arg4, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %124 = load %arg2[%arg4, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %124, %arg2[%arg4, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %125 = addi %arg3, %arg5 : index
              %126 = addi %125, %c14 : index
              %127 = addi %arg6, %arg7 : index
              %128 = load %arg0[%arg4, %127] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %129 = load %arg1[%127, %126] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %130 = mulf %128, %129 {RelaxedPrecision} : f32
              %131 = load %arg2[%arg4, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %132 = addf %131, %130 {RelaxedPrecision} : f32
              store %132, %arg2[%arg4, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %133 = load %arg2[%arg4, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %133, %arg2[%arg4, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %134 = addi %arg3, %arg5 : index
              %135 = addi %134, %c15 : index
              %136 = addi %arg6, %arg7 : index
              %137 = load %arg0[%arg4, %136] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %138 = load %arg1[%136, %135] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %139 = mulf %137, %138 {RelaxedPrecision} : f32
              %140 = load %arg2[%arg4, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %141 = addf %140, %139 {RelaxedPrecision} : f32
              store %141, %arg2[%arg4, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %142 = load %arg2[%arg4, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %142, %arg2[%arg4, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %143 = addi %arg4, %c1 : index
              %144 = addi %arg3, %arg5 : index
              %145 = addi %arg6, %arg7 : index
              %146 = load %arg0[%143, %145] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %147 = load %arg1[%145, %144] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %148 = mulf %146, %147 {RelaxedPrecision} : f32
              %149 = load %arg2[%143, %144] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %150 = addf %149, %148 {RelaxedPrecision} : f32
              store %150, %arg2[%143, %144] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %151 = load %arg2[%143, %144] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %151, %arg2[%143, %144] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %152 = addi %arg4, %c1 : index
              %153 = addi %arg3, %arg5 : index
              %154 = addi %153, %c1 : index
              %155 = addi %arg6, %arg7 : index
              %156 = load %arg0[%152, %155] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %157 = load %arg1[%155, %154] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %158 = mulf %156, %157 {RelaxedPrecision} : f32
              %159 = load %arg2[%152, %154] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %160 = addf %159, %158 {RelaxedPrecision} : f32
              store %160, %arg2[%152, %154] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %161 = load %arg2[%152, %154] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %161, %arg2[%152, %154] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %162 = addi %arg4, %c1 : index
              %163 = addi %arg3, %arg5 : index
              %164 = addi %163, %c2 : index
              %165 = addi %arg6, %arg7 : index
              %166 = load %arg0[%162, %165] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %167 = load %arg1[%165, %164] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %168 = mulf %166, %167 {RelaxedPrecision} : f32
              %169 = load %arg2[%162, %164] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %170 = addf %169, %168 {RelaxedPrecision} : f32
              store %170, %arg2[%162, %164] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %171 = load %arg2[%162, %164] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %171, %arg2[%162, %164] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %172 = addi %arg4, %c1 : index
              %173 = addi %arg3, %arg5 : index
              %174 = addi %173, %c3 : index
              %175 = addi %arg6, %arg7 : index
              %176 = load %arg0[%172, %175] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %177 = load %arg1[%175, %174] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %178 = mulf %176, %177 {RelaxedPrecision} : f32
              %179 = load %arg2[%172, %174] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %180 = addf %179, %178 {RelaxedPrecision} : f32
              store %180, %arg2[%172, %174] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %181 = load %arg2[%172, %174] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %181, %arg2[%172, %174] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %182 = addi %arg4, %c1 : index
              %183 = addi %arg3, %arg5 : index
              %184 = addi %183, %c4 : index
              %185 = addi %arg6, %arg7 : index
              %186 = load %arg0[%182, %185] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %187 = load %arg1[%185, %184] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %188 = mulf %186, %187 {RelaxedPrecision} : f32
              %189 = load %arg2[%182, %184] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %190 = addf %189, %188 {RelaxedPrecision} : f32
              store %190, %arg2[%182, %184] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %191 = load %arg2[%182, %184] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %191, %arg2[%182, %184] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %192 = addi %arg4, %c1 : index
              %193 = addi %arg3, %arg5 : index
              %194 = addi %193, %c5 : index
              %195 = addi %arg6, %arg7 : index
              %196 = load %arg0[%192, %195] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %197 = load %arg1[%195, %194] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %198 = mulf %196, %197 {RelaxedPrecision} : f32
              %199 = load %arg2[%192, %194] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %200 = addf %199, %198 {RelaxedPrecision} : f32
              store %200, %arg2[%192, %194] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %201 = load %arg2[%192, %194] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %201, %arg2[%192, %194] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %202 = addi %arg4, %c1 : index
              %203 = addi %arg3, %arg5 : index
              %204 = addi %203, %c6 : index
              %205 = addi %arg6, %arg7 : index
              %206 = load %arg0[%202, %205] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %207 = load %arg1[%205, %204] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %208 = mulf %206, %207 {RelaxedPrecision} : f32
              %209 = load %arg2[%202, %204] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %210 = addf %209, %208 {RelaxedPrecision} : f32
              store %210, %arg2[%202, %204] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %211 = load %arg2[%202, %204] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %211, %arg2[%202, %204] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %212 = addi %arg4, %c1 : index
              %213 = addi %arg3, %arg5 : index
              %214 = addi %213, %c7 : index
              %215 = addi %arg6, %arg7 : index
              %216 = load %arg0[%212, %215] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %217 = load %arg1[%215, %214] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %218 = mulf %216, %217 {RelaxedPrecision} : f32
              %219 = load %arg2[%212, %214] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %220 = addf %219, %218 {RelaxedPrecision} : f32
              store %220, %arg2[%212, %214] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %221 = load %arg2[%212, %214] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %221, %arg2[%212, %214] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %222 = addi %arg4, %c1 : index
              %223 = addi %arg3, %arg5 : index
              %224 = addi %223, %c8 : index
              %225 = addi %arg6, %arg7 : index
              %226 = load %arg0[%222, %225] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %227 = load %arg1[%225, %224] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %228 = mulf %226, %227 {RelaxedPrecision} : f32
              %229 = load %arg2[%222, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %230 = addf %229, %228 {RelaxedPrecision} : f32
              store %230, %arg2[%222, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %231 = load %arg2[%222, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %231, %arg2[%222, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %232 = addi %arg4, %c1 : index
              %233 = addi %arg3, %arg5 : index
              %234 = addi %233, %c9 : index
              %235 = addi %arg6, %arg7 : index
              %236 = load %arg0[%232, %235] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %237 = load %arg1[%235, %234] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %238 = mulf %236, %237 {RelaxedPrecision} : f32
              %239 = load %arg2[%232, %234] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %240 = addf %239, %238 {RelaxedPrecision} : f32
              store %240, %arg2[%232, %234] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %241 = load %arg2[%232, %234] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %241, %arg2[%232, %234] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %242 = addi %arg4, %c1 : index
              %243 = addi %arg3, %arg5 : index
              %244 = addi %243, %c10 : index
              %245 = addi %arg6, %arg7 : index
              %246 = load %arg0[%242, %245] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %247 = load %arg1[%245, %244] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %248 = mulf %246, %247 {RelaxedPrecision} : f32
              %249 = load %arg2[%242, %244] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %250 = addf %249, %248 {RelaxedPrecision} : f32
              store %250, %arg2[%242, %244] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %251 = load %arg2[%242, %244] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %251, %arg2[%242, %244] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %252 = addi %arg4, %c1 : index
              %253 = addi %arg3, %arg5 : index
              %254 = addi %253, %c11 : index
              %255 = addi %arg6, %arg7 : index
              %256 = load %arg0[%252, %255] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %257 = load %arg1[%255, %254] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %258 = mulf %256, %257 {RelaxedPrecision} : f32
              %259 = load %arg2[%252, %254] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %260 = addf %259, %258 {RelaxedPrecision} : f32
              store %260, %arg2[%252, %254] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %261 = load %arg2[%252, %254] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %261, %arg2[%252, %254] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %262 = addi %arg4, %c1 : index
              %263 = addi %arg3, %arg5 : index
              %264 = addi %263, %c12 : index
              %265 = addi %arg6, %arg7 : index
              %266 = load %arg0[%262, %265] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %267 = load %arg1[%265, %264] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %268 = mulf %266, %267 {RelaxedPrecision} : f32
              %269 = load %arg2[%262, %264] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %270 = addf %269, %268 {RelaxedPrecision} : f32
              store %270, %arg2[%262, %264] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %271 = load %arg2[%262, %264] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %271, %arg2[%262, %264] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %272 = addi %arg4, %c1 : index
              %273 = addi %arg3, %arg5 : index
              %274 = addi %273, %c13 : index
              %275 = addi %arg6, %arg7 : index
              %276 = load %arg0[%272, %275] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %277 = load %arg1[%275, %274] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %278 = mulf %276, %277 {RelaxedPrecision} : f32
              %279 = load %arg2[%272, %274] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %280 = addf %279, %278 {RelaxedPrecision} : f32
              store %280, %arg2[%272, %274] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %281 = load %arg2[%272, %274] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %281, %arg2[%272, %274] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %282 = addi %arg4, %c1 : index
              %283 = addi %arg3, %arg5 : index
              %284 = addi %283, %c14 : index
              %285 = addi %arg6, %arg7 : index
              %286 = load %arg0[%282, %285] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %287 = load %arg1[%285, %284] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %288 = mulf %286, %287 {RelaxedPrecision} : f32
              %289 = load %arg2[%282, %284] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %290 = addf %289, %288 {RelaxedPrecision} : f32
              store %290, %arg2[%282, %284] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %291 = load %arg2[%282, %284] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %291, %arg2[%282, %284] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %292 = addi %arg4, %c1 : index
              %293 = addi %arg3, %arg5 : index
              %294 = addi %293, %c15 : index
              %295 = addi %arg6, %arg7 : index
              %296 = load %arg0[%292, %295] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %297 = load %arg1[%295, %294] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %298 = mulf %296, %297 {RelaxedPrecision} : f32
              %299 = load %arg2[%292, %294] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %300 = addf %299, %298 {RelaxedPrecision} : f32
              store %300, %arg2[%292, %294] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %301 = load %arg2[%292, %294] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %301, %arg2[%292, %294] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %302 = addi %arg4, %c2 : index
              %303 = addi %arg3, %arg5 : index
              %304 = addi %arg6, %arg7 : index
              %305 = load %arg0[%302, %304] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %306 = load %arg1[%304, %303] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %307 = mulf %305, %306 {RelaxedPrecision} : f32
              %308 = load %arg2[%302, %303] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %309 = addf %308, %307 {RelaxedPrecision} : f32
              store %309, %arg2[%302, %303] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %310 = load %arg2[%302, %303] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %310, %arg2[%302, %303] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %311 = addi %arg4, %c2 : index
              %312 = addi %arg3, %arg5 : index
              %313 = addi %312, %c1 : index
              %314 = addi %arg6, %arg7 : index
              %315 = load %arg0[%311, %314] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %316 = load %arg1[%314, %313] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %317 = mulf %315, %316 {RelaxedPrecision} : f32
              %318 = load %arg2[%311, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %319 = addf %318, %317 {RelaxedPrecision} : f32
              store %319, %arg2[%311, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %320 = load %arg2[%311, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %320, %arg2[%311, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %321 = addi %arg4, %c2 : index
              %322 = addi %arg3, %arg5 : index
              %323 = addi %322, %c2 : index
              %324 = addi %arg6, %arg7 : index
              %325 = load %arg0[%321, %324] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %326 = load %arg1[%324, %323] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %327 = mulf %325, %326 {RelaxedPrecision} : f32
              %328 = load %arg2[%321, %323] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %329 = addf %328, %327 {RelaxedPrecision} : f32
              store %329, %arg2[%321, %323] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %330 = load %arg2[%321, %323] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %330, %arg2[%321, %323] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %331 = addi %arg4, %c2 : index
              %332 = addi %arg3, %arg5 : index
              %333 = addi %332, %c3 : index
              %334 = addi %arg6, %arg7 : index
              %335 = load %arg0[%331, %334] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %336 = load %arg1[%334, %333] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %337 = mulf %335, %336 {RelaxedPrecision} : f32
              %338 = load %arg2[%331, %333] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %339 = addf %338, %337 {RelaxedPrecision} : f32
              store %339, %arg2[%331, %333] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %340 = load %arg2[%331, %333] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %340, %arg2[%331, %333] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %341 = addi %arg4, %c2 : index
              %342 = addi %arg3, %arg5 : index
              %343 = addi %342, %c4 : index
              %344 = addi %arg6, %arg7 : index
              %345 = load %arg0[%341, %344] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %346 = load %arg1[%344, %343] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %347 = mulf %345, %346 {RelaxedPrecision} : f32
              %348 = load %arg2[%341, %343] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %349 = addf %348, %347 {RelaxedPrecision} : f32
              store %349, %arg2[%341, %343] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %350 = load %arg2[%341, %343] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %350, %arg2[%341, %343] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %351 = addi %arg4, %c2 : index
              %352 = addi %arg3, %arg5 : index
              %353 = addi %352, %c5 : index
              %354 = addi %arg6, %arg7 : index
              %355 = load %arg0[%351, %354] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %356 = load %arg1[%354, %353] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %357 = mulf %355, %356 {RelaxedPrecision} : f32
              %358 = load %arg2[%351, %353] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %359 = addf %358, %357 {RelaxedPrecision} : f32
              store %359, %arg2[%351, %353] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %360 = load %arg2[%351, %353] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %360, %arg2[%351, %353] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %361 = addi %arg4, %c2 : index
              %362 = addi %arg3, %arg5 : index
              %363 = addi %362, %c6 : index
              %364 = addi %arg6, %arg7 : index
              %365 = load %arg0[%361, %364] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %366 = load %arg1[%364, %363] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %367 = mulf %365, %366 {RelaxedPrecision} : f32
              %368 = load %arg2[%361, %363] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %369 = addf %368, %367 {RelaxedPrecision} : f32
              store %369, %arg2[%361, %363] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %370 = load %arg2[%361, %363] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %370, %arg2[%361, %363] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %371 = addi %arg4, %c2 : index
              %372 = addi %arg3, %arg5 : index
              %373 = addi %372, %c7 : index
              %374 = addi %arg6, %arg7 : index
              %375 = load %arg0[%371, %374] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %376 = load %arg1[%374, %373] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %377 = mulf %375, %376 {RelaxedPrecision} : f32
              %378 = load %arg2[%371, %373] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %379 = addf %378, %377 {RelaxedPrecision} : f32
              store %379, %arg2[%371, %373] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %380 = load %arg2[%371, %373] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %380, %arg2[%371, %373] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %381 = addi %arg4, %c2 : index
              %382 = addi %arg3, %arg5 : index
              %383 = addi %382, %c8 : index
              %384 = addi %arg6, %arg7 : index
              %385 = load %arg0[%381, %384] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %386 = load %arg1[%384, %383] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %387 = mulf %385, %386 {RelaxedPrecision} : f32
              %388 = load %arg2[%381, %383] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %389 = addf %388, %387 {RelaxedPrecision} : f32
              store %389, %arg2[%381, %383] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %390 = load %arg2[%381, %383] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %390, %arg2[%381, %383] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %391 = addi %arg4, %c2 : index
              %392 = addi %arg3, %arg5 : index
              %393 = addi %392, %c9 : index
              %394 = addi %arg6, %arg7 : index
              %395 = load %arg0[%391, %394] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %396 = load %arg1[%394, %393] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %397 = mulf %395, %396 {RelaxedPrecision} : f32
              %398 = load %arg2[%391, %393] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %399 = addf %398, %397 {RelaxedPrecision} : f32
              store %399, %arg2[%391, %393] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %400 = load %arg2[%391, %393] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %400, %arg2[%391, %393] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %401 = addi %arg4, %c2 : index
              %402 = addi %arg3, %arg5 : index
              %403 = addi %402, %c10 : index
              %404 = addi %arg6, %arg7 : index
              %405 = load %arg0[%401, %404] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %406 = load %arg1[%404, %403] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %407 = mulf %405, %406 {RelaxedPrecision} : f32
              %408 = load %arg2[%401, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %409 = addf %408, %407 {RelaxedPrecision} : f32
              store %409, %arg2[%401, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %410 = load %arg2[%401, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %410, %arg2[%401, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %411 = addi %arg4, %c2 : index
              %412 = addi %arg3, %arg5 : index
              %413 = addi %412, %c11 : index
              %414 = addi %arg6, %arg7 : index
              %415 = load %arg0[%411, %414] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %416 = load %arg1[%414, %413] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %417 = mulf %415, %416 {RelaxedPrecision} : f32
              %418 = load %arg2[%411, %413] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %419 = addf %418, %417 {RelaxedPrecision} : f32
              store %419, %arg2[%411, %413] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %420 = load %arg2[%411, %413] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %420, %arg2[%411, %413] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %421 = addi %arg4, %c2 : index
              %422 = addi %arg3, %arg5 : index
              %423 = addi %422, %c12 : index
              %424 = addi %arg6, %arg7 : index
              %425 = load %arg0[%421, %424] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %426 = load %arg1[%424, %423] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %427 = mulf %425, %426 {RelaxedPrecision} : f32
              %428 = load %arg2[%421, %423] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %429 = addf %428, %427 {RelaxedPrecision} : f32
              store %429, %arg2[%421, %423] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %430 = load %arg2[%421, %423] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %430, %arg2[%421, %423] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %431 = addi %arg4, %c2 : index
              %432 = addi %arg3, %arg5 : index
              %433 = addi %432, %c13 : index
              %434 = addi %arg6, %arg7 : index
              %435 = load %arg0[%431, %434] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %436 = load %arg1[%434, %433] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %437 = mulf %435, %436 {RelaxedPrecision} : f32
              %438 = load %arg2[%431, %433] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %439 = addf %438, %437 {RelaxedPrecision} : f32
              store %439, %arg2[%431, %433] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %440 = load %arg2[%431, %433] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %440, %arg2[%431, %433] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %441 = addi %arg4, %c2 : index
              %442 = addi %arg3, %arg5 : index
              %443 = addi %442, %c14 : index
              %444 = addi %arg6, %arg7 : index
              %445 = load %arg0[%441, %444] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %446 = load %arg1[%444, %443] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %447 = mulf %445, %446 {RelaxedPrecision} : f32
              %448 = load %arg2[%441, %443] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %449 = addf %448, %447 {RelaxedPrecision} : f32
              store %449, %arg2[%441, %443] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %450 = load %arg2[%441, %443] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %450, %arg2[%441, %443] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %451 = addi %arg4, %c2 : index
              %452 = addi %arg3, %arg5 : index
              %453 = addi %452, %c15 : index
              %454 = addi %arg6, %arg7 : index
              %455 = load %arg0[%451, %454] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %456 = load %arg1[%454, %453] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %457 = mulf %455, %456 {RelaxedPrecision} : f32
              %458 = load %arg2[%451, %453] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %459 = addf %458, %457 {RelaxedPrecision} : f32
              store %459, %arg2[%451, %453] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %460 = load %arg2[%451, %453] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %460, %arg2[%451, %453] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %461 = addi %arg4, %c3 : index
              %462 = addi %arg3, %arg5 : index
              %463 = addi %arg6, %arg7 : index
              %464 = load %arg0[%461, %463] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %465 = load %arg1[%463, %462] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %466 = mulf %464, %465 {RelaxedPrecision} : f32
              %467 = load %arg2[%461, %462] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %468 = addf %467, %466 {RelaxedPrecision} : f32
              store %468, %arg2[%461, %462] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %469 = load %arg2[%461, %462] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %469, %arg2[%461, %462] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %470 = addi %arg4, %c3 : index
              %471 = addi %arg3, %arg5 : index
              %472 = addi %471, %c1 : index
              %473 = addi %arg6, %arg7 : index
              %474 = load %arg0[%470, %473] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %475 = load %arg1[%473, %472] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %476 = mulf %474, %475 {RelaxedPrecision} : f32
              %477 = load %arg2[%470, %472] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %478 = addf %477, %476 {RelaxedPrecision} : f32
              store %478, %arg2[%470, %472] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %479 = load %arg2[%470, %472] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %479, %arg2[%470, %472] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %480 = addi %arg4, %c3 : index
              %481 = addi %arg3, %arg5 : index
              %482 = addi %481, %c2 : index
              %483 = addi %arg6, %arg7 : index
              %484 = load %arg0[%480, %483] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %485 = load %arg1[%483, %482] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %486 = mulf %484, %485 {RelaxedPrecision} : f32
              %487 = load %arg2[%480, %482] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %488 = addf %487, %486 {RelaxedPrecision} : f32
              store %488, %arg2[%480, %482] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %489 = load %arg2[%480, %482] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %489, %arg2[%480, %482] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %490 = addi %arg4, %c3 : index
              %491 = addi %arg3, %arg5 : index
              %492 = addi %491, %c3 : index
              %493 = addi %arg6, %arg7 : index
              %494 = load %arg0[%490, %493] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %495 = load %arg1[%493, %492] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %496 = mulf %494, %495 {RelaxedPrecision} : f32
              %497 = load %arg2[%490, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %498 = addf %497, %496 {RelaxedPrecision} : f32
              store %498, %arg2[%490, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %499 = load %arg2[%490, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %499, %arg2[%490, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %500 = addi %arg4, %c3 : index
              %501 = addi %arg3, %arg5 : index
              %502 = addi %501, %c4 : index
              %503 = addi %arg6, %arg7 : index
              %504 = load %arg0[%500, %503] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %505 = load %arg1[%503, %502] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %506 = mulf %504, %505 {RelaxedPrecision} : f32
              %507 = load %arg2[%500, %502] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %508 = addf %507, %506 {RelaxedPrecision} : f32
              store %508, %arg2[%500, %502] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %509 = load %arg2[%500, %502] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %509, %arg2[%500, %502] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %510 = addi %arg4, %c3 : index
              %511 = addi %arg3, %arg5 : index
              %512 = addi %511, %c5 : index
              %513 = addi %arg6, %arg7 : index
              %514 = load %arg0[%510, %513] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %515 = load %arg1[%513, %512] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %516 = mulf %514, %515 {RelaxedPrecision} : f32
              %517 = load %arg2[%510, %512] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %518 = addf %517, %516 {RelaxedPrecision} : f32
              store %518, %arg2[%510, %512] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %519 = load %arg2[%510, %512] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %519, %arg2[%510, %512] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %520 = addi %arg4, %c3 : index
              %521 = addi %arg3, %arg5 : index
              %522 = addi %521, %c6 : index
              %523 = addi %arg6, %arg7 : index
              %524 = load %arg0[%520, %523] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %525 = load %arg1[%523, %522] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %526 = mulf %524, %525 {RelaxedPrecision} : f32
              %527 = load %arg2[%520, %522] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %528 = addf %527, %526 {RelaxedPrecision} : f32
              store %528, %arg2[%520, %522] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %529 = load %arg2[%520, %522] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %529, %arg2[%520, %522] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %530 = addi %arg4, %c3 : index
              %531 = addi %arg3, %arg5 : index
              %532 = addi %531, %c7 : index
              %533 = addi %arg6, %arg7 : index
              %534 = load %arg0[%530, %533] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %535 = load %arg1[%533, %532] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %536 = mulf %534, %535 {RelaxedPrecision} : f32
              %537 = load %arg2[%530, %532] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %538 = addf %537, %536 {RelaxedPrecision} : f32
              store %538, %arg2[%530, %532] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %539 = load %arg2[%530, %532] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %539, %arg2[%530, %532] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %540 = addi %arg4, %c3 : index
              %541 = addi %arg3, %arg5 : index
              %542 = addi %541, %c8 : index
              %543 = addi %arg6, %arg7 : index
              %544 = load %arg0[%540, %543] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %545 = load %arg1[%543, %542] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %546 = mulf %544, %545 {RelaxedPrecision} : f32
              %547 = load %arg2[%540, %542] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %548 = addf %547, %546 {RelaxedPrecision} : f32
              store %548, %arg2[%540, %542] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %549 = load %arg2[%540, %542] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %549, %arg2[%540, %542] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %550 = addi %arg4, %c3 : index
              %551 = addi %arg3, %arg5 : index
              %552 = addi %551, %c9 : index
              %553 = addi %arg6, %arg7 : index
              %554 = load %arg0[%550, %553] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %555 = load %arg1[%553, %552] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %556 = mulf %554, %555 {RelaxedPrecision} : f32
              %557 = load %arg2[%550, %552] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %558 = addf %557, %556 {RelaxedPrecision} : f32
              store %558, %arg2[%550, %552] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %559 = load %arg2[%550, %552] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %559, %arg2[%550, %552] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %560 = addi %arg4, %c3 : index
              %561 = addi %arg3, %arg5 : index
              %562 = addi %561, %c10 : index
              %563 = addi %arg6, %arg7 : index
              %564 = load %arg0[%560, %563] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %565 = load %arg1[%563, %562] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %566 = mulf %564, %565 {RelaxedPrecision} : f32
              %567 = load %arg2[%560, %562] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %568 = addf %567, %566 {RelaxedPrecision} : f32
              store %568, %arg2[%560, %562] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %569 = load %arg2[%560, %562] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %569, %arg2[%560, %562] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %570 = addi %arg4, %c3 : index
              %571 = addi %arg3, %arg5 : index
              %572 = addi %571, %c11 : index
              %573 = addi %arg6, %arg7 : index
              %574 = load %arg0[%570, %573] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %575 = load %arg1[%573, %572] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %576 = mulf %574, %575 {RelaxedPrecision} : f32
              %577 = load %arg2[%570, %572] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %578 = addf %577, %576 {RelaxedPrecision} : f32
              store %578, %arg2[%570, %572] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %579 = load %arg2[%570, %572] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %579, %arg2[%570, %572] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %580 = addi %arg4, %c3 : index
              %581 = addi %arg3, %arg5 : index
              %582 = addi %581, %c12 : index
              %583 = addi %arg6, %arg7 : index
              %584 = load %arg0[%580, %583] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %585 = load %arg1[%583, %582] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %586 = mulf %584, %585 {RelaxedPrecision} : f32
              %587 = load %arg2[%580, %582] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %588 = addf %587, %586 {RelaxedPrecision} : f32
              store %588, %arg2[%580, %582] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %589 = load %arg2[%580, %582] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %589, %arg2[%580, %582] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %590 = addi %arg4, %c3 : index
              %591 = addi %arg3, %arg5 : index
              %592 = addi %591, %c13 : index
              %593 = addi %arg6, %arg7 : index
              %594 = load %arg0[%590, %593] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %595 = load %arg1[%593, %592] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %596 = mulf %594, %595 {RelaxedPrecision} : f32
              %597 = load %arg2[%590, %592] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %598 = addf %597, %596 {RelaxedPrecision} : f32
              store %598, %arg2[%590, %592] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %599 = load %arg2[%590, %592] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %599, %arg2[%590, %592] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %600 = addi %arg4, %c3 : index
              %601 = addi %arg3, %arg5 : index
              %602 = addi %601, %c14 : index
              %603 = addi %arg6, %arg7 : index
              %604 = load %arg0[%600, %603] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %605 = load %arg1[%603, %602] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %606 = mulf %604, %605 {RelaxedPrecision} : f32
              %607 = load %arg2[%600, %602] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %608 = addf %607, %606 {RelaxedPrecision} : f32
              store %608, %arg2[%600, %602] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %609 = load %arg2[%600, %602] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %609, %arg2[%600, %602] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %610 = addi %arg4, %c3 : index
              %611 = addi %arg3, %arg5 : index
              %612 = addi %611, %c15 : index
              %613 = addi %arg6, %arg7 : index
              %614 = load %arg0[%610, %613] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %615 = load %arg1[%613, %612] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %616 = mulf %614, %615 {RelaxedPrecision} : f32
              %617 = load %arg2[%610, %612] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %618 = addf %617, %616 {RelaxedPrecision} : f32
              store %618, %arg2[%610, %612] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %619 = load %arg2[%610, %612] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %619, %arg2[%610, %612] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %620 = addi %arg4, %c4 : index
              %621 = addi %arg3, %arg5 : index
              %622 = addi %arg6, %arg7 : index
              %623 = load %arg0[%620, %622] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %624 = load %arg1[%622, %621] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %625 = mulf %623, %624 {RelaxedPrecision} : f32
              %626 = load %arg2[%620, %621] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %627 = addf %626, %625 {RelaxedPrecision} : f32
              store %627, %arg2[%620, %621] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %628 = load %arg2[%620, %621] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %628, %arg2[%620, %621] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %629 = addi %arg4, %c4 : index
              %630 = addi %arg3, %arg5 : index
              %631 = addi %630, %c1 : index
              %632 = addi %arg6, %arg7 : index
              %633 = load %arg0[%629, %632] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %634 = load %arg1[%632, %631] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %635 = mulf %633, %634 {RelaxedPrecision} : f32
              %636 = load %arg2[%629, %631] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %637 = addf %636, %635 {RelaxedPrecision} : f32
              store %637, %arg2[%629, %631] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %638 = load %arg2[%629, %631] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %638, %arg2[%629, %631] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %639 = addi %arg4, %c4 : index
              %640 = addi %arg3, %arg5 : index
              %641 = addi %640, %c2 : index
              %642 = addi %arg6, %arg7 : index
              %643 = load %arg0[%639, %642] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %644 = load %arg1[%642, %641] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %645 = mulf %643, %644 {RelaxedPrecision} : f32
              %646 = load %arg2[%639, %641] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %647 = addf %646, %645 {RelaxedPrecision} : f32
              store %647, %arg2[%639, %641] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %648 = load %arg2[%639, %641] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %648, %arg2[%639, %641] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %649 = addi %arg4, %c4 : index
              %650 = addi %arg3, %arg5 : index
              %651 = addi %650, %c3 : index
              %652 = addi %arg6, %arg7 : index
              %653 = load %arg0[%649, %652] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %654 = load %arg1[%652, %651] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %655 = mulf %653, %654 {RelaxedPrecision} : f32
              %656 = load %arg2[%649, %651] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %657 = addf %656, %655 {RelaxedPrecision} : f32
              store %657, %arg2[%649, %651] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %658 = load %arg2[%649, %651] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %658, %arg2[%649, %651] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %659 = addi %arg4, %c4 : index
              %660 = addi %arg3, %arg5 : index
              %661 = addi %660, %c4 : index
              %662 = addi %arg6, %arg7 : index
              %663 = load %arg0[%659, %662] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %664 = load %arg1[%662, %661] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %665 = mulf %663, %664 {RelaxedPrecision} : f32
              %666 = load %arg2[%659, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %667 = addf %666, %665 {RelaxedPrecision} : f32
              store %667, %arg2[%659, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %668 = load %arg2[%659, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %668, %arg2[%659, %661] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %669 = addi %arg4, %c4 : index
              %670 = addi %arg3, %arg5 : index
              %671 = addi %670, %c5 : index
              %672 = addi %arg6, %arg7 : index
              %673 = load %arg0[%669, %672] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %674 = load %arg1[%672, %671] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %675 = mulf %673, %674 {RelaxedPrecision} : f32
              %676 = load %arg2[%669, %671] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %677 = addf %676, %675 {RelaxedPrecision} : f32
              store %677, %arg2[%669, %671] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %678 = load %arg2[%669, %671] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %678, %arg2[%669, %671] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %679 = addi %arg4, %c4 : index
              %680 = addi %arg3, %arg5 : index
              %681 = addi %680, %c6 : index
              %682 = addi %arg6, %arg7 : index
              %683 = load %arg0[%679, %682] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %684 = load %arg1[%682, %681] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %685 = mulf %683, %684 {RelaxedPrecision} : f32
              %686 = load %arg2[%679, %681] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %687 = addf %686, %685 {RelaxedPrecision} : f32
              store %687, %arg2[%679, %681] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %688 = load %arg2[%679, %681] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %688, %arg2[%679, %681] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %689 = addi %arg4, %c4 : index
              %690 = addi %arg3, %arg5 : index
              %691 = addi %690, %c7 : index
              %692 = addi %arg6, %arg7 : index
              %693 = load %arg0[%689, %692] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %694 = load %arg1[%692, %691] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %695 = mulf %693, %694 {RelaxedPrecision} : f32
              %696 = load %arg2[%689, %691] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %697 = addf %696, %695 {RelaxedPrecision} : f32
              store %697, %arg2[%689, %691] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %698 = load %arg2[%689, %691] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %698, %arg2[%689, %691] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %699 = addi %arg4, %c4 : index
              %700 = addi %arg3, %arg5 : index
              %701 = addi %700, %c8 : index
              %702 = addi %arg6, %arg7 : index
              %703 = load %arg0[%699, %702] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %704 = load %arg1[%702, %701] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %705 = mulf %703, %704 {RelaxedPrecision} : f32
              %706 = load %arg2[%699, %701] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %707 = addf %706, %705 {RelaxedPrecision} : f32
              store %707, %arg2[%699, %701] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %708 = load %arg2[%699, %701] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %708, %arg2[%699, %701] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %709 = addi %arg4, %c4 : index
              %710 = addi %arg3, %arg5 : index
              %711 = addi %710, %c9 : index
              %712 = addi %arg6, %arg7 : index
              %713 = load %arg0[%709, %712] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %714 = load %arg1[%712, %711] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %715 = mulf %713, %714 {RelaxedPrecision} : f32
              %716 = load %arg2[%709, %711] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %717 = addf %716, %715 {RelaxedPrecision} : f32
              store %717, %arg2[%709, %711] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %718 = load %arg2[%709, %711] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %718, %arg2[%709, %711] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %719 = addi %arg4, %c4 : index
              %720 = addi %arg3, %arg5 : index
              %721 = addi %720, %c10 : index
              %722 = addi %arg6, %arg7 : index
              %723 = load %arg0[%719, %722] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %724 = load %arg1[%722, %721] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %725 = mulf %723, %724 {RelaxedPrecision} : f32
              %726 = load %arg2[%719, %721] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %727 = addf %726, %725 {RelaxedPrecision} : f32
              store %727, %arg2[%719, %721] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %728 = load %arg2[%719, %721] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %728, %arg2[%719, %721] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %729 = addi %arg4, %c4 : index
              %730 = addi %arg3, %arg5 : index
              %731 = addi %730, %c11 : index
              %732 = addi %arg6, %arg7 : index
              %733 = load %arg0[%729, %732] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %734 = load %arg1[%732, %731] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %735 = mulf %733, %734 {RelaxedPrecision} : f32
              %736 = load %arg2[%729, %731] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %737 = addf %736, %735 {RelaxedPrecision} : f32
              store %737, %arg2[%729, %731] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %738 = load %arg2[%729, %731] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %738, %arg2[%729, %731] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %739 = addi %arg4, %c4 : index
              %740 = addi %arg3, %arg5 : index
              %741 = addi %740, %c12 : index
              %742 = addi %arg6, %arg7 : index
              %743 = load %arg0[%739, %742] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %744 = load %arg1[%742, %741] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %745 = mulf %743, %744 {RelaxedPrecision} : f32
              %746 = load %arg2[%739, %741] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %747 = addf %746, %745 {RelaxedPrecision} : f32
              store %747, %arg2[%739, %741] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %748 = load %arg2[%739, %741] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %748, %arg2[%739, %741] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %749 = addi %arg4, %c4 : index
              %750 = addi %arg3, %arg5 : index
              %751 = addi %750, %c13 : index
              %752 = addi %arg6, %arg7 : index
              %753 = load %arg0[%749, %752] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %754 = load %arg1[%752, %751] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %755 = mulf %753, %754 {RelaxedPrecision} : f32
              %756 = load %arg2[%749, %751] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %757 = addf %756, %755 {RelaxedPrecision} : f32
              store %757, %arg2[%749, %751] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %758 = load %arg2[%749, %751] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %758, %arg2[%749, %751] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %759 = addi %arg4, %c4 : index
              %760 = addi %arg3, %arg5 : index
              %761 = addi %760, %c14 : index
              %762 = addi %arg6, %arg7 : index
              %763 = load %arg0[%759, %762] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %764 = load %arg1[%762, %761] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %765 = mulf %763, %764 {RelaxedPrecision} : f32
              %766 = load %arg2[%759, %761] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %767 = addf %766, %765 {RelaxedPrecision} : f32
              store %767, %arg2[%759, %761] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %768 = load %arg2[%759, %761] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %768, %arg2[%759, %761] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %769 = addi %arg4, %c4 : index
              %770 = addi %arg3, %arg5 : index
              %771 = addi %770, %c15 : index
              %772 = addi %arg6, %arg7 : index
              %773 = load %arg0[%769, %772] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %774 = load %arg1[%772, %771] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %775 = mulf %773, %774 {RelaxedPrecision} : f32
              %776 = load %arg2[%769, %771] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %777 = addf %776, %775 {RelaxedPrecision} : f32
              store %777, %arg2[%769, %771] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %778 = load %arg2[%769, %771] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %778, %arg2[%769, %771] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %779 = addi %arg4, %c5 : index
              %780 = addi %arg3, %arg5 : index
              %781 = addi %arg6, %arg7 : index
              %782 = load %arg0[%779, %781] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %783 = load %arg1[%781, %780] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %784 = mulf %782, %783 {RelaxedPrecision} : f32
              %785 = load %arg2[%779, %780] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %786 = addf %785, %784 {RelaxedPrecision} : f32
              store %786, %arg2[%779, %780] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %787 = load %arg2[%779, %780] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %787, %arg2[%779, %780] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %788 = addi %arg4, %c5 : index
              %789 = addi %arg3, %arg5 : index
              %790 = addi %789, %c1 : index
              %791 = addi %arg6, %arg7 : index
              %792 = load %arg0[%788, %791] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %793 = load %arg1[%791, %790] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %794 = mulf %792, %793 {RelaxedPrecision} : f32
              %795 = load %arg2[%788, %790] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %796 = addf %795, %794 {RelaxedPrecision} : f32
              store %796, %arg2[%788, %790] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %797 = load %arg2[%788, %790] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %797, %arg2[%788, %790] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %798 = addi %arg4, %c5 : index
              %799 = addi %arg3, %arg5 : index
              %800 = addi %799, %c2 : index
              %801 = addi %arg6, %arg7 : index
              %802 = load %arg0[%798, %801] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %803 = load %arg1[%801, %800] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %804 = mulf %802, %803 {RelaxedPrecision} : f32
              %805 = load %arg2[%798, %800] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %806 = addf %805, %804 {RelaxedPrecision} : f32
              store %806, %arg2[%798, %800] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %807 = load %arg2[%798, %800] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %807, %arg2[%798, %800] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %808 = addi %arg4, %c5 : index
              %809 = addi %arg3, %arg5 : index
              %810 = addi %809, %c3 : index
              %811 = addi %arg6, %arg7 : index
              %812 = load %arg0[%808, %811] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %813 = load %arg1[%811, %810] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %814 = mulf %812, %813 {RelaxedPrecision} : f32
              %815 = load %arg2[%808, %810] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %816 = addf %815, %814 {RelaxedPrecision} : f32
              store %816, %arg2[%808, %810] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %817 = load %arg2[%808, %810] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %817, %arg2[%808, %810] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %818 = addi %arg4, %c5 : index
              %819 = addi %arg3, %arg5 : index
              %820 = addi %819, %c4 : index
              %821 = addi %arg6, %arg7 : index
              %822 = load %arg0[%818, %821] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %823 = load %arg1[%821, %820] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %824 = mulf %822, %823 {RelaxedPrecision} : f32
              %825 = load %arg2[%818, %820] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %826 = addf %825, %824 {RelaxedPrecision} : f32
              store %826, %arg2[%818, %820] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %827 = load %arg2[%818, %820] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %827, %arg2[%818, %820] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %828 = addi %arg4, %c5 : index
              %829 = addi %arg3, %arg5 : index
              %830 = addi %829, %c5 : index
              %831 = addi %arg6, %arg7 : index
              %832 = load %arg0[%828, %831] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %833 = load %arg1[%831, %830] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %834 = mulf %832, %833 {RelaxedPrecision} : f32
              %835 = load %arg2[%828, %830] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %836 = addf %835, %834 {RelaxedPrecision} : f32
              store %836, %arg2[%828, %830] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %837 = load %arg2[%828, %830] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %837, %arg2[%828, %830] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %838 = addi %arg4, %c5 : index
              %839 = addi %arg3, %arg5 : index
              %840 = addi %839, %c6 : index
              %841 = addi %arg6, %arg7 : index
              %842 = load %arg0[%838, %841] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %843 = load %arg1[%841, %840] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %844 = mulf %842, %843 {RelaxedPrecision} : f32
              %845 = load %arg2[%838, %840] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %846 = addf %845, %844 {RelaxedPrecision} : f32
              store %846, %arg2[%838, %840] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %847 = load %arg2[%838, %840] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %847, %arg2[%838, %840] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %848 = addi %arg4, %c5 : index
              %849 = addi %arg3, %arg5 : index
              %850 = addi %849, %c7 : index
              %851 = addi %arg6, %arg7 : index
              %852 = load %arg0[%848, %851] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %853 = load %arg1[%851, %850] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %854 = mulf %852, %853 {RelaxedPrecision} : f32
              %855 = load %arg2[%848, %850] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %856 = addf %855, %854 {RelaxedPrecision} : f32
              store %856, %arg2[%848, %850] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %857 = load %arg2[%848, %850] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %857, %arg2[%848, %850] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %858 = addi %arg4, %c5 : index
              %859 = addi %arg3, %arg5 : index
              %860 = addi %859, %c8 : index
              %861 = addi %arg6, %arg7 : index
              %862 = load %arg0[%858, %861] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %863 = load %arg1[%861, %860] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %864 = mulf %862, %863 {RelaxedPrecision} : f32
              %865 = load %arg2[%858, %860] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %866 = addf %865, %864 {RelaxedPrecision} : f32
              store %866, %arg2[%858, %860] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %867 = load %arg2[%858, %860] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %867, %arg2[%858, %860] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %868 = addi %arg4, %c5 : index
              %869 = addi %arg3, %arg5 : index
              %870 = addi %869, %c9 : index
              %871 = addi %arg6, %arg7 : index
              %872 = load %arg0[%868, %871] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %873 = load %arg1[%871, %870] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %874 = mulf %872, %873 {RelaxedPrecision} : f32
              %875 = load %arg2[%868, %870] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %876 = addf %875, %874 {RelaxedPrecision} : f32
              store %876, %arg2[%868, %870] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %877 = load %arg2[%868, %870] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %877, %arg2[%868, %870] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %878 = addi %arg4, %c5 : index
              %879 = addi %arg3, %arg5 : index
              %880 = addi %879, %c10 : index
              %881 = addi %arg6, %arg7 : index
              %882 = load %arg0[%878, %881] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %883 = load %arg1[%881, %880] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %884 = mulf %882, %883 {RelaxedPrecision} : f32
              %885 = load %arg2[%878, %880] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %886 = addf %885, %884 {RelaxedPrecision} : f32
              store %886, %arg2[%878, %880] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %887 = load %arg2[%878, %880] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %887, %arg2[%878, %880] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %888 = addi %arg4, %c5 : index
              %889 = addi %arg3, %arg5 : index
              %890 = addi %889, %c11 : index
              %891 = addi %arg6, %arg7 : index
              %892 = load %arg0[%888, %891] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %893 = load %arg1[%891, %890] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %894 = mulf %892, %893 {RelaxedPrecision} : f32
              %895 = load %arg2[%888, %890] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %896 = addf %895, %894 {RelaxedPrecision} : f32
              store %896, %arg2[%888, %890] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %897 = load %arg2[%888, %890] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %897, %arg2[%888, %890] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %898 = addi %arg4, %c5 : index
              %899 = addi %arg3, %arg5 : index
              %900 = addi %899, %c12 : index
              %901 = addi %arg6, %arg7 : index
              %902 = load %arg0[%898, %901] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %903 = load %arg1[%901, %900] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %904 = mulf %902, %903 {RelaxedPrecision} : f32
              %905 = load %arg2[%898, %900] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %906 = addf %905, %904 {RelaxedPrecision} : f32
              store %906, %arg2[%898, %900] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %907 = load %arg2[%898, %900] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %907, %arg2[%898, %900] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %908 = addi %arg4, %c5 : index
              %909 = addi %arg3, %arg5 : index
              %910 = addi %909, %c13 : index
              %911 = addi %arg6, %arg7 : index
              %912 = load %arg0[%908, %911] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %913 = load %arg1[%911, %910] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %914 = mulf %912, %913 {RelaxedPrecision} : f32
              %915 = load %arg2[%908, %910] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %916 = addf %915, %914 {RelaxedPrecision} : f32
              store %916, %arg2[%908, %910] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %917 = load %arg2[%908, %910] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %917, %arg2[%908, %910] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %918 = addi %arg4, %c5 : index
              %919 = addi %arg3, %arg5 : index
              %920 = addi %919, %c14 : index
              %921 = addi %arg6, %arg7 : index
              %922 = load %arg0[%918, %921] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %923 = load %arg1[%921, %920] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %924 = mulf %922, %923 {RelaxedPrecision} : f32
              %925 = load %arg2[%918, %920] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %926 = addf %925, %924 {RelaxedPrecision} : f32
              store %926, %arg2[%918, %920] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %927 = load %arg2[%918, %920] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %927, %arg2[%918, %920] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %928 = addi %arg4, %c5 : index
              %929 = addi %arg3, %arg5 : index
              %930 = addi %929, %c15 : index
              %931 = addi %arg6, %arg7 : index
              %932 = load %arg0[%928, %931] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %933 = load %arg1[%931, %930] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %934 = mulf %932, %933 {RelaxedPrecision} : f32
              %935 = load %arg2[%928, %930] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %936 = addf %935, %934 {RelaxedPrecision} : f32
              store %936, %arg2[%928, %930] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              %937 = load %arg2[%928, %930] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
              store %937, %arg2[%928, %930] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
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
            %8 = addi %arg3, %arg4 : index
            %9 = addi %8, %c1 : index
            %10 = addi %arg5, %arg6 : index
            %11 = load %arg0[%c780, %10] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %12 = load %arg1[%10, %9] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %13 = mulf %11, %12 {RelaxedPrecision} : f32
            %14 = load %arg2[%c780, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %15 = addf %14, %13 {RelaxedPrecision} : f32
            store %15, %arg2[%c780, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %16 = load %arg2[%c780, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %16, %arg2[%c780, %9] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %17 = addi %arg3, %arg4 : index
            %18 = addi %17, %c2 : index
            %19 = addi %arg5, %arg6 : index
            %20 = load %arg0[%c780, %19] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %21 = load %arg1[%19, %18] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %22 = mulf %20, %21 {RelaxedPrecision} : f32
            %23 = load %arg2[%c780, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %24 = addf %23, %22 {RelaxedPrecision} : f32
            store %24, %arg2[%c780, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %25 = load %arg2[%c780, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %25, %arg2[%c780, %18] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %26 = addi %arg3, %arg4 : index
            %27 = addi %26, %c3 : index
            %28 = addi %arg5, %arg6 : index
            %29 = load %arg0[%c780, %28] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %30 = load %arg1[%28, %27] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %31 = mulf %29, %30 {RelaxedPrecision} : f32
            %32 = load %arg2[%c780, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %33 = addf %32, %31 {RelaxedPrecision} : f32
            store %33, %arg2[%c780, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %34 = load %arg2[%c780, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %34, %arg2[%c780, %27] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %35 = addi %arg3, %arg4 : index
            %36 = addi %35, %c4 : index
            %37 = addi %arg5, %arg6 : index
            %38 = load %arg0[%c780, %37] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %39 = load %arg1[%37, %36] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %40 = mulf %38, %39 {RelaxedPrecision} : f32
            %41 = load %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %42 = addf %41, %40 {RelaxedPrecision} : f32
            store %42, %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %43 = load %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %43, %arg2[%c780, %36] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %44 = addi %arg3, %arg4 : index
            %45 = addi %44, %c5 : index
            %46 = addi %arg5, %arg6 : index
            %47 = load %arg0[%c780, %46] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %48 = load %arg1[%46, %45] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %49 = mulf %47, %48 {RelaxedPrecision} : f32
            %50 = load %arg2[%c780, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %51 = addf %50, %49 {RelaxedPrecision} : f32
            store %51, %arg2[%c780, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %52 = load %arg2[%c780, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %52, %arg2[%c780, %45] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %53 = addi %arg3, %arg4 : index
            %54 = addi %53, %c6 : index
            %55 = addi %arg5, %arg6 : index
            %56 = load %arg0[%c780, %55] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %57 = load %arg1[%55, %54] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %58 = mulf %56, %57 {RelaxedPrecision} : f32
            %59 = load %arg2[%c780, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %60 = addf %59, %58 {RelaxedPrecision} : f32
            store %60, %arg2[%c780, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %61 = load %arg2[%c780, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %61, %arg2[%c780, %54] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %62 = addi %arg3, %arg4 : index
            %63 = addi %62, %c7 : index
            %64 = addi %arg5, %arg6 : index
            %65 = load %arg0[%c780, %64] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %66 = load %arg1[%64, %63] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %67 = mulf %65, %66 {RelaxedPrecision} : f32
            %68 = load %arg2[%c780, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %69 = addf %68, %67 {RelaxedPrecision} : f32
            store %69, %arg2[%c780, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %70 = load %arg2[%c780, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %70, %arg2[%c780, %63] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %71 = addi %arg3, %arg4 : index
            %72 = addi %71, %c8 : index
            %73 = addi %arg5, %arg6 : index
            %74 = load %arg0[%c780, %73] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %75 = load %arg1[%73, %72] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %76 = mulf %74, %75 {RelaxedPrecision} : f32
            %77 = load %arg2[%c780, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %78 = addf %77, %76 {RelaxedPrecision} : f32
            store %78, %arg2[%c780, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %79 = load %arg2[%c780, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %79, %arg2[%c780, %72] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %80 = addi %arg3, %arg4 : index
            %81 = addi %80, %c9 : index
            %82 = addi %arg5, %arg6 : index
            %83 = load %arg0[%c780, %82] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %84 = load %arg1[%82, %81] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %85 = mulf %83, %84 {RelaxedPrecision} : f32
            %86 = load %arg2[%c780, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %87 = addf %86, %85 {RelaxedPrecision} : f32
            store %87, %arg2[%c780, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %88 = load %arg2[%c780, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %88, %arg2[%c780, %81] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %89 = addi %arg3, %arg4 : index
            %90 = addi %89, %c10 : index
            %91 = addi %arg5, %arg6 : index
            %92 = load %arg0[%c780, %91] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %93 = load %arg1[%91, %90] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %94 = mulf %92, %93 {RelaxedPrecision} : f32
            %95 = load %arg2[%c780, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %96 = addf %95, %94 {RelaxedPrecision} : f32
            store %96, %arg2[%c780, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %97 = load %arg2[%c780, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %97, %arg2[%c780, %90] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %98 = addi %arg3, %arg4 : index
            %99 = addi %98, %c11 : index
            %100 = addi %arg5, %arg6 : index
            %101 = load %arg0[%c780, %100] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %102 = load %arg1[%100, %99] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %103 = mulf %101, %102 {RelaxedPrecision} : f32
            %104 = load %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %105 = addf %104, %103 {RelaxedPrecision} : f32
            store %105, %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %106 = load %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %106, %arg2[%c780, %99] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %107 = addi %arg3, %arg4 : index
            %108 = addi %107, %c12 : index
            %109 = addi %arg5, %arg6 : index
            %110 = load %arg0[%c780, %109] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %111 = load %arg1[%109, %108] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %112 = mulf %110, %111 {RelaxedPrecision} : f32
            %113 = load %arg2[%c780, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %114 = addf %113, %112 {RelaxedPrecision} : f32
            store %114, %arg2[%c780, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %115 = load %arg2[%c780, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %115, %arg2[%c780, %108] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %116 = addi %arg3, %arg4 : index
            %117 = addi %116, %c13 : index
            %118 = addi %arg5, %arg6 : index
            %119 = load %arg0[%c780, %118] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %120 = load %arg1[%118, %117] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %121 = mulf %119, %120 {RelaxedPrecision} : f32
            %122 = load %arg2[%c780, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %123 = addf %122, %121 {RelaxedPrecision} : f32
            store %123, %arg2[%c780, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %124 = load %arg2[%c780, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %124, %arg2[%c780, %117] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %125 = addi %arg3, %arg4 : index
            %126 = addi %125, %c14 : index
            %127 = addi %arg5, %arg6 : index
            %128 = load %arg0[%c780, %127] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %129 = load %arg1[%127, %126] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %130 = mulf %128, %129 {RelaxedPrecision} : f32
            %131 = load %arg2[%c780, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %132 = addf %131, %130 {RelaxedPrecision} : f32
            store %132, %arg2[%c780, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %133 = load %arg2[%c780, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %133, %arg2[%c780, %126] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %134 = addi %arg3, %arg4 : index
            %135 = addi %134, %c15 : index
            %136 = addi %arg5, %arg6 : index
            %137 = load %arg0[%c780, %136] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %138 = load %arg1[%136, %135] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %139 = mulf %137, %138 {RelaxedPrecision} : f32
            %140 = load %arg2[%c780, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %141 = addf %140, %139 {RelaxedPrecision} : f32
            store %141, %arg2[%c780, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %142 = load %arg2[%c780, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %142, %arg2[%c780, %135] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %143 = addi %arg3, %arg4 : index
            %144 = addi %arg5, %arg6 : index
            %145 = load %arg0[%c781, %144] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %146 = load %arg1[%144, %143] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %147 = mulf %145, %146 {RelaxedPrecision} : f32
            %148 = load %arg2[%c781, %143] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %149 = addf %148, %147 {RelaxedPrecision} : f32
            store %149, %arg2[%c781, %143] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %150 = load %arg2[%c781, %143] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %150, %arg2[%c781, %143] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %151 = addi %arg3, %arg4 : index
            %152 = addi %151, %c1 : index
            %153 = addi %arg5, %arg6 : index
            %154 = load %arg0[%c781, %153] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %155 = load %arg1[%153, %152] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %156 = mulf %154, %155 {RelaxedPrecision} : f32
            %157 = load %arg2[%c781, %152] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %158 = addf %157, %156 {RelaxedPrecision} : f32
            store %158, %arg2[%c781, %152] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %159 = load %arg2[%c781, %152] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %159, %arg2[%c781, %152] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %160 = addi %arg3, %arg4 : index
            %161 = addi %160, %c2 : index
            %162 = addi %arg5, %arg6 : index
            %163 = load %arg0[%c781, %162] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %164 = load %arg1[%162, %161] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %165 = mulf %163, %164 {RelaxedPrecision} : f32
            %166 = load %arg2[%c781, %161] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %167 = addf %166, %165 {RelaxedPrecision} : f32
            store %167, %arg2[%c781, %161] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %168 = load %arg2[%c781, %161] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %168, %arg2[%c781, %161] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %169 = addi %arg3, %arg4 : index
            %170 = addi %169, %c3 : index
            %171 = addi %arg5, %arg6 : index
            %172 = load %arg0[%c781, %171] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %173 = load %arg1[%171, %170] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %174 = mulf %172, %173 {RelaxedPrecision} : f32
            %175 = load %arg2[%c781, %170] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %176 = addf %175, %174 {RelaxedPrecision} : f32
            store %176, %arg2[%c781, %170] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %177 = load %arg2[%c781, %170] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %177, %arg2[%c781, %170] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %178 = addi %arg3, %arg4 : index
            %179 = addi %178, %c4 : index
            %180 = addi %arg5, %arg6 : index
            %181 = load %arg0[%c781, %180] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %182 = load %arg1[%180, %179] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %183 = mulf %181, %182 {RelaxedPrecision} : f32
            %184 = load %arg2[%c781, %179] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %185 = addf %184, %183 {RelaxedPrecision} : f32
            store %185, %arg2[%c781, %179] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %186 = load %arg2[%c781, %179] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %186, %arg2[%c781, %179] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %187 = addi %arg3, %arg4 : index
            %188 = addi %187, %c5 : index
            %189 = addi %arg5, %arg6 : index
            %190 = load %arg0[%c781, %189] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %191 = load %arg1[%189, %188] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %192 = mulf %190, %191 {RelaxedPrecision} : f32
            %193 = load %arg2[%c781, %188] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %194 = addf %193, %192 {RelaxedPrecision} : f32
            store %194, %arg2[%c781, %188] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %195 = load %arg2[%c781, %188] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %195, %arg2[%c781, %188] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %196 = addi %arg3, %arg4 : index
            %197 = addi %196, %c6 : index
            %198 = addi %arg5, %arg6 : index
            %199 = load %arg0[%c781, %198] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %200 = load %arg1[%198, %197] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %201 = mulf %199, %200 {RelaxedPrecision} : f32
            %202 = load %arg2[%c781, %197] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %203 = addf %202, %201 {RelaxedPrecision} : f32
            store %203, %arg2[%c781, %197] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %204 = load %arg2[%c781, %197] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %204, %arg2[%c781, %197] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %205 = addi %arg3, %arg4 : index
            %206 = addi %205, %c7 : index
            %207 = addi %arg5, %arg6 : index
            %208 = load %arg0[%c781, %207] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %209 = load %arg1[%207, %206] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %210 = mulf %208, %209 {RelaxedPrecision} : f32
            %211 = load %arg2[%c781, %206] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %212 = addf %211, %210 {RelaxedPrecision} : f32
            store %212, %arg2[%c781, %206] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %213 = load %arg2[%c781, %206] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %213, %arg2[%c781, %206] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %214 = addi %arg3, %arg4 : index
            %215 = addi %214, %c8 : index
            %216 = addi %arg5, %arg6 : index
            %217 = load %arg0[%c781, %216] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %218 = load %arg1[%216, %215] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %219 = mulf %217, %218 {RelaxedPrecision} : f32
            %220 = load %arg2[%c781, %215] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %221 = addf %220, %219 {RelaxedPrecision} : f32
            store %221, %arg2[%c781, %215] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %222 = load %arg2[%c781, %215] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %222, %arg2[%c781, %215] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %223 = addi %arg3, %arg4 : index
            %224 = addi %223, %c9 : index
            %225 = addi %arg5, %arg6 : index
            %226 = load %arg0[%c781, %225] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %227 = load %arg1[%225, %224] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %228 = mulf %226, %227 {RelaxedPrecision} : f32
            %229 = load %arg2[%c781, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %230 = addf %229, %228 {RelaxedPrecision} : f32
            store %230, %arg2[%c781, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %231 = load %arg2[%c781, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %231, %arg2[%c781, %224] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %232 = addi %arg3, %arg4 : index
            %233 = addi %232, %c10 : index
            %234 = addi %arg5, %arg6 : index
            %235 = load %arg0[%c781, %234] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %236 = load %arg1[%234, %233] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %237 = mulf %235, %236 {RelaxedPrecision} : f32
            %238 = load %arg2[%c781, %233] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %239 = addf %238, %237 {RelaxedPrecision} : f32
            store %239, %arg2[%c781, %233] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %240 = load %arg2[%c781, %233] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %240, %arg2[%c781, %233] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %241 = addi %arg3, %arg4 : index
            %242 = addi %241, %c11 : index
            %243 = addi %arg5, %arg6 : index
            %244 = load %arg0[%c781, %243] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %245 = load %arg1[%243, %242] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %246 = mulf %244, %245 {RelaxedPrecision} : f32
            %247 = load %arg2[%c781, %242] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %248 = addf %247, %246 {RelaxedPrecision} : f32
            store %248, %arg2[%c781, %242] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %249 = load %arg2[%c781, %242] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %249, %arg2[%c781, %242] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %250 = addi %arg3, %arg4 : index
            %251 = addi %250, %c12 : index
            %252 = addi %arg5, %arg6 : index
            %253 = load %arg0[%c781, %252] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %254 = load %arg1[%252, %251] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %255 = mulf %253, %254 {RelaxedPrecision} : f32
            %256 = load %arg2[%c781, %251] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %257 = addf %256, %255 {RelaxedPrecision} : f32
            store %257, %arg2[%c781, %251] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %258 = load %arg2[%c781, %251] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %258, %arg2[%c781, %251] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %259 = addi %arg3, %arg4 : index
            %260 = addi %259, %c13 : index
            %261 = addi %arg5, %arg6 : index
            %262 = load %arg0[%c781, %261] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %263 = load %arg1[%261, %260] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %264 = mulf %262, %263 {RelaxedPrecision} : f32
            %265 = load %arg2[%c781, %260] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %266 = addf %265, %264 {RelaxedPrecision} : f32
            store %266, %arg2[%c781, %260] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %267 = load %arg2[%c781, %260] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %267, %arg2[%c781, %260] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %268 = addi %arg3, %arg4 : index
            %269 = addi %268, %c14 : index
            %270 = addi %arg5, %arg6 : index
            %271 = load %arg0[%c781, %270] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %272 = load %arg1[%270, %269] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %273 = mulf %271, %272 {RelaxedPrecision} : f32
            %274 = load %arg2[%c781, %269] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %275 = addf %274, %273 {RelaxedPrecision} : f32
            store %275, %arg2[%c781, %269] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %276 = load %arg2[%c781, %269] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %276, %arg2[%c781, %269] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %277 = addi %arg3, %arg4 : index
            %278 = addi %277, %c15 : index
            %279 = addi %arg5, %arg6 : index
            %280 = load %arg0[%c781, %279] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %281 = load %arg1[%279, %278] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %282 = mulf %280, %281 {RelaxedPrecision} : f32
            %283 = load %arg2[%c781, %278] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %284 = addf %283, %282 {RelaxedPrecision} : f32
            store %284, %arg2[%c781, %278] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %285 = load %arg2[%c781, %278] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %285, %arg2[%c781, %278] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %286 = addi %arg3, %arg4 : index
            %287 = addi %arg5, %arg6 : index
            %288 = load %arg0[%c782, %287] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %289 = load %arg1[%287, %286] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %290 = mulf %288, %289 {RelaxedPrecision} : f32
            %291 = load %arg2[%c782, %286] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %292 = addf %291, %290 {RelaxedPrecision} : f32
            store %292, %arg2[%c782, %286] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %293 = load %arg2[%c782, %286] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %293, %arg2[%c782, %286] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %294 = addi %arg3, %arg4 : index
            %295 = addi %294, %c1 : index
            %296 = addi %arg5, %arg6 : index
            %297 = load %arg0[%c782, %296] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %298 = load %arg1[%296, %295] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %299 = mulf %297, %298 {RelaxedPrecision} : f32
            %300 = load %arg2[%c782, %295] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %301 = addf %300, %299 {RelaxedPrecision} : f32
            store %301, %arg2[%c782, %295] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %302 = load %arg2[%c782, %295] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %302, %arg2[%c782, %295] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %303 = addi %arg3, %arg4 : index
            %304 = addi %303, %c2 : index
            %305 = addi %arg5, %arg6 : index
            %306 = load %arg0[%c782, %305] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %307 = load %arg1[%305, %304] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %308 = mulf %306, %307 {RelaxedPrecision} : f32
            %309 = load %arg2[%c782, %304] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %310 = addf %309, %308 {RelaxedPrecision} : f32
            store %310, %arg2[%c782, %304] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %311 = load %arg2[%c782, %304] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %311, %arg2[%c782, %304] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %312 = addi %arg3, %arg4 : index
            %313 = addi %312, %c3 : index
            %314 = addi %arg5, %arg6 : index
            %315 = load %arg0[%c782, %314] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %316 = load %arg1[%314, %313] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %317 = mulf %315, %316 {RelaxedPrecision} : f32
            %318 = load %arg2[%c782, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %319 = addf %318, %317 {RelaxedPrecision} : f32
            store %319, %arg2[%c782, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %320 = load %arg2[%c782, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %320, %arg2[%c782, %313] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %321 = addi %arg3, %arg4 : index
            %322 = addi %321, %c4 : index
            %323 = addi %arg5, %arg6 : index
            %324 = load %arg0[%c782, %323] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %325 = load %arg1[%323, %322] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %326 = mulf %324, %325 {RelaxedPrecision} : f32
            %327 = load %arg2[%c782, %322] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %328 = addf %327, %326 {RelaxedPrecision} : f32
            store %328, %arg2[%c782, %322] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %329 = load %arg2[%c782, %322] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %329, %arg2[%c782, %322] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %330 = addi %arg3, %arg4 : index
            %331 = addi %330, %c5 : index
            %332 = addi %arg5, %arg6 : index
            %333 = load %arg0[%c782, %332] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %334 = load %arg1[%332, %331] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %335 = mulf %333, %334 {RelaxedPrecision} : f32
            %336 = load %arg2[%c782, %331] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %337 = addf %336, %335 {RelaxedPrecision} : f32
            store %337, %arg2[%c782, %331] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %338 = load %arg2[%c782, %331] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %338, %arg2[%c782, %331] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %339 = addi %arg3, %arg4 : index
            %340 = addi %339, %c6 : index
            %341 = addi %arg5, %arg6 : index
            %342 = load %arg0[%c782, %341] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %343 = load %arg1[%341, %340] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %344 = mulf %342, %343 {RelaxedPrecision} : f32
            %345 = load %arg2[%c782, %340] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %346 = addf %345, %344 {RelaxedPrecision} : f32
            store %346, %arg2[%c782, %340] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %347 = load %arg2[%c782, %340] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %347, %arg2[%c782, %340] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %348 = addi %arg3, %arg4 : index
            %349 = addi %348, %c7 : index
            %350 = addi %arg5, %arg6 : index
            %351 = load %arg0[%c782, %350] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %352 = load %arg1[%350, %349] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %353 = mulf %351, %352 {RelaxedPrecision} : f32
            %354 = load %arg2[%c782, %349] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %355 = addf %354, %353 {RelaxedPrecision} : f32
            store %355, %arg2[%c782, %349] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %356 = load %arg2[%c782, %349] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %356, %arg2[%c782, %349] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %357 = addi %arg3, %arg4 : index
            %358 = addi %357, %c8 : index
            %359 = addi %arg5, %arg6 : index
            %360 = load %arg0[%c782, %359] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %361 = load %arg1[%359, %358] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %362 = mulf %360, %361 {RelaxedPrecision} : f32
            %363 = load %arg2[%c782, %358] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %364 = addf %363, %362 {RelaxedPrecision} : f32
            store %364, %arg2[%c782, %358] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %365 = load %arg2[%c782, %358] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %365, %arg2[%c782, %358] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %366 = addi %arg3, %arg4 : index
            %367 = addi %366, %c9 : index
            %368 = addi %arg5, %arg6 : index
            %369 = load %arg0[%c782, %368] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %370 = load %arg1[%368, %367] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %371 = mulf %369, %370 {RelaxedPrecision} : f32
            %372 = load %arg2[%c782, %367] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %373 = addf %372, %371 {RelaxedPrecision} : f32
            store %373, %arg2[%c782, %367] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %374 = load %arg2[%c782, %367] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %374, %arg2[%c782, %367] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %375 = addi %arg3, %arg4 : index
            %376 = addi %375, %c10 : index
            %377 = addi %arg5, %arg6 : index
            %378 = load %arg0[%c782, %377] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %379 = load %arg1[%377, %376] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %380 = mulf %378, %379 {RelaxedPrecision} : f32
            %381 = load %arg2[%c782, %376] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %382 = addf %381, %380 {RelaxedPrecision} : f32
            store %382, %arg2[%c782, %376] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %383 = load %arg2[%c782, %376] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %383, %arg2[%c782, %376] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %384 = addi %arg3, %arg4 : index
            %385 = addi %384, %c11 : index
            %386 = addi %arg5, %arg6 : index
            %387 = load %arg0[%c782, %386] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %388 = load %arg1[%386, %385] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %389 = mulf %387, %388 {RelaxedPrecision} : f32
            %390 = load %arg2[%c782, %385] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %391 = addf %390, %389 {RelaxedPrecision} : f32
            store %391, %arg2[%c782, %385] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %392 = load %arg2[%c782, %385] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %392, %arg2[%c782, %385] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %393 = addi %arg3, %arg4 : index
            %394 = addi %393, %c12 : index
            %395 = addi %arg5, %arg6 : index
            %396 = load %arg0[%c782, %395] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %397 = load %arg1[%395, %394] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %398 = mulf %396, %397 {RelaxedPrecision} : f32
            %399 = load %arg2[%c782, %394] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %400 = addf %399, %398 {RelaxedPrecision} : f32
            store %400, %arg2[%c782, %394] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %401 = load %arg2[%c782, %394] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %401, %arg2[%c782, %394] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %402 = addi %arg3, %arg4 : index
            %403 = addi %402, %c13 : index
            %404 = addi %arg5, %arg6 : index
            %405 = load %arg0[%c782, %404] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %406 = load %arg1[%404, %403] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %407 = mulf %405, %406 {RelaxedPrecision} : f32
            %408 = load %arg2[%c782, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %409 = addf %408, %407 {RelaxedPrecision} : f32
            store %409, %arg2[%c782, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %410 = load %arg2[%c782, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %410, %arg2[%c782, %403] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %411 = addi %arg3, %arg4 : index
            %412 = addi %411, %c14 : index
            %413 = addi %arg5, %arg6 : index
            %414 = load %arg0[%c782, %413] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %415 = load %arg1[%413, %412] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %416 = mulf %414, %415 {RelaxedPrecision} : f32
            %417 = load %arg2[%c782, %412] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %418 = addf %417, %416 {RelaxedPrecision} : f32
            store %418, %arg2[%c782, %412] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %419 = load %arg2[%c782, %412] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %419, %arg2[%c782, %412] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %420 = addi %arg3, %arg4 : index
            %421 = addi %420, %c15 : index
            %422 = addi %arg5, %arg6 : index
            %423 = load %arg0[%c782, %422] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %424 = load %arg1[%422, %421] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %425 = mulf %423, %424 {RelaxedPrecision} : f32
            %426 = load %arg2[%c782, %421] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %427 = addf %426, %425 {RelaxedPrecision} : f32
            store %427, %arg2[%c782, %421] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %428 = load %arg2[%c782, %421] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %428, %arg2[%c782, %421] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %429 = addi %arg3, %arg4 : index
            %430 = addi %arg5, %arg6 : index
            %431 = load %arg0[%c783, %430] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %432 = load %arg1[%430, %429] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %433 = mulf %431, %432 {RelaxedPrecision} : f32
            %434 = load %arg2[%c783, %429] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %435 = addf %434, %433 {RelaxedPrecision} : f32
            store %435, %arg2[%c783, %429] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %436 = load %arg2[%c783, %429] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %436, %arg2[%c783, %429] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %437 = addi %arg3, %arg4 : index
            %438 = addi %437, %c1 : index
            %439 = addi %arg5, %arg6 : index
            %440 = load %arg0[%c783, %439] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %441 = load %arg1[%439, %438] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %442 = mulf %440, %441 {RelaxedPrecision} : f32
            %443 = load %arg2[%c783, %438] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %444 = addf %443, %442 {RelaxedPrecision} : f32
            store %444, %arg2[%c783, %438] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %445 = load %arg2[%c783, %438] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %445, %arg2[%c783, %438] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %446 = addi %arg3, %arg4 : index
            %447 = addi %446, %c2 : index
            %448 = addi %arg5, %arg6 : index
            %449 = load %arg0[%c783, %448] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %450 = load %arg1[%448, %447] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %451 = mulf %449, %450 {RelaxedPrecision} : f32
            %452 = load %arg2[%c783, %447] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %453 = addf %452, %451 {RelaxedPrecision} : f32
            store %453, %arg2[%c783, %447] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %454 = load %arg2[%c783, %447] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %454, %arg2[%c783, %447] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %455 = addi %arg3, %arg4 : index
            %456 = addi %455, %c3 : index
            %457 = addi %arg5, %arg6 : index
            %458 = load %arg0[%c783, %457] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %459 = load %arg1[%457, %456] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %460 = mulf %458, %459 {RelaxedPrecision} : f32
            %461 = load %arg2[%c783, %456] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %462 = addf %461, %460 {RelaxedPrecision} : f32
            store %462, %arg2[%c783, %456] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %463 = load %arg2[%c783, %456] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %463, %arg2[%c783, %456] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %464 = addi %arg3, %arg4 : index
            %465 = addi %464, %c4 : index
            %466 = addi %arg5, %arg6 : index
            %467 = load %arg0[%c783, %466] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %468 = load %arg1[%466, %465] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %469 = mulf %467, %468 {RelaxedPrecision} : f32
            %470 = load %arg2[%c783, %465] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %471 = addf %470, %469 {RelaxedPrecision} : f32
            store %471, %arg2[%c783, %465] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %472 = load %arg2[%c783, %465] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %472, %arg2[%c783, %465] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %473 = addi %arg3, %arg4 : index
            %474 = addi %473, %c5 : index
            %475 = addi %arg5, %arg6 : index
            %476 = load %arg0[%c783, %475] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %477 = load %arg1[%475, %474] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %478 = mulf %476, %477 {RelaxedPrecision} : f32
            %479 = load %arg2[%c783, %474] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %480 = addf %479, %478 {RelaxedPrecision} : f32
            store %480, %arg2[%c783, %474] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %481 = load %arg2[%c783, %474] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %481, %arg2[%c783, %474] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %482 = addi %arg3, %arg4 : index
            %483 = addi %482, %c6 : index
            %484 = addi %arg5, %arg6 : index
            %485 = load %arg0[%c783, %484] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %486 = load %arg1[%484, %483] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %487 = mulf %485, %486 {RelaxedPrecision} : f32
            %488 = load %arg2[%c783, %483] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %489 = addf %488, %487 {RelaxedPrecision} : f32
            store %489, %arg2[%c783, %483] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %490 = load %arg2[%c783, %483] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %490, %arg2[%c783, %483] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %491 = addi %arg3, %arg4 : index
            %492 = addi %491, %c7 : index
            %493 = addi %arg5, %arg6 : index
            %494 = load %arg0[%c783, %493] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %495 = load %arg1[%493, %492] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %496 = mulf %494, %495 {RelaxedPrecision} : f32
            %497 = load %arg2[%c783, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %498 = addf %497, %496 {RelaxedPrecision} : f32
            store %498, %arg2[%c783, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %499 = load %arg2[%c783, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %499, %arg2[%c783, %492] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %500 = addi %arg3, %arg4 : index
            %501 = addi %500, %c8 : index
            %502 = addi %arg5, %arg6 : index
            %503 = load %arg0[%c783, %502] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %504 = load %arg1[%502, %501] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %505 = mulf %503, %504 {RelaxedPrecision} : f32
            %506 = load %arg2[%c783, %501] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %507 = addf %506, %505 {RelaxedPrecision} : f32
            store %507, %arg2[%c783, %501] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %508 = load %arg2[%c783, %501] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %508, %arg2[%c783, %501] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %509 = addi %arg3, %arg4 : index
            %510 = addi %509, %c9 : index
            %511 = addi %arg5, %arg6 : index
            %512 = load %arg0[%c783, %511] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %513 = load %arg1[%511, %510] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %514 = mulf %512, %513 {RelaxedPrecision} : f32
            %515 = load %arg2[%c783, %510] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %516 = addf %515, %514 {RelaxedPrecision} : f32
            store %516, %arg2[%c783, %510] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %517 = load %arg2[%c783, %510] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %517, %arg2[%c783, %510] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %518 = addi %arg3, %arg4 : index
            %519 = addi %518, %c10 : index
            %520 = addi %arg5, %arg6 : index
            %521 = load %arg0[%c783, %520] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %522 = load %arg1[%520, %519] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %523 = mulf %521, %522 {RelaxedPrecision} : f32
            %524 = load %arg2[%c783, %519] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %525 = addf %524, %523 {RelaxedPrecision} : f32
            store %525, %arg2[%c783, %519] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %526 = load %arg2[%c783, %519] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %526, %arg2[%c783, %519] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %527 = addi %arg3, %arg4 : index
            %528 = addi %527, %c11 : index
            %529 = addi %arg5, %arg6 : index
            %530 = load %arg0[%c783, %529] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %531 = load %arg1[%529, %528] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %532 = mulf %530, %531 {RelaxedPrecision} : f32
            %533 = load %arg2[%c783, %528] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %534 = addf %533, %532 {RelaxedPrecision} : f32
            store %534, %arg2[%c783, %528] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %535 = load %arg2[%c783, %528] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %535, %arg2[%c783, %528] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %536 = addi %arg3, %arg4 : index
            %537 = addi %536, %c12 : index
            %538 = addi %arg5, %arg6 : index
            %539 = load %arg0[%c783, %538] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %540 = load %arg1[%538, %537] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %541 = mulf %539, %540 {RelaxedPrecision} : f32
            %542 = load %arg2[%c783, %537] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %543 = addf %542, %541 {RelaxedPrecision} : f32
            store %543, %arg2[%c783, %537] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %544 = load %arg2[%c783, %537] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %544, %arg2[%c783, %537] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %545 = addi %arg3, %arg4 : index
            %546 = addi %545, %c13 : index
            %547 = addi %arg5, %arg6 : index
            %548 = load %arg0[%c783, %547] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %549 = load %arg1[%547, %546] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %550 = mulf %548, %549 {RelaxedPrecision} : f32
            %551 = load %arg2[%c783, %546] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %552 = addf %551, %550 {RelaxedPrecision} : f32
            store %552, %arg2[%c783, %546] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %553 = load %arg2[%c783, %546] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %553, %arg2[%c783, %546] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %554 = addi %arg3, %arg4 : index
            %555 = addi %554, %c14 : index
            %556 = addi %arg5, %arg6 : index
            %557 = load %arg0[%c783, %556] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %558 = load %arg1[%556, %555] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %559 = mulf %557, %558 {RelaxedPrecision} : f32
            %560 = load %arg2[%c783, %555] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %561 = addf %560, %559 {RelaxedPrecision} : f32
            store %561, %arg2[%c783, %555] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %562 = load %arg2[%c783, %555] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %562, %arg2[%c783, %555] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %563 = addi %arg3, %arg4 : index
            %564 = addi %563, %c15 : index
            %565 = addi %arg5, %arg6 : index
            %566 = load %arg0[%c783, %565] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
            %567 = load %arg1[%565, %564] : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %568 = mulf %566, %567 {RelaxedPrecision} : f32
            %569 = load %arg2[%c783, %564] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %570 = addf %569, %568 {RelaxedPrecision} : f32
            store %570, %arg2[%c783, %564] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            %571 = load %arg2[%c783, %564] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            store %571, %arg2[%c783, %564] : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
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
