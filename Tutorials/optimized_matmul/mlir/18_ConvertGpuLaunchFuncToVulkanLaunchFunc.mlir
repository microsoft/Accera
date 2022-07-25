module @optimized_matmul attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"} {
  "accv.global"() {sym_name = "cache_17", type = memref<16x128x2xvector<8xf32>>} : () -> ()
  "accv.global"() {sym_name = "cache_16", type = memref<16x6x2xvector<8xf32>>} : () -> ()
  func @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, sym_visibility = "nested"} {
    %cst = constant 0.000000e+00 : f32
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %c2_i64 = constant 2 : i64
    %c3_i64 = constant 3 : i64
    %c4_i64 = constant 4 : i64
    %c5_i64 = constant 5 : i64
    %c6_i64 = constant 6 : i64
    %c7_i64 = constant 7 : i64
    %cst_0 = constant dense<0.000000e+00> : vector<8xf32>
    %c10 = constant 10 : index
    %c12 = constant 12 : index
    %c14 = constant 14 : index
    %c512 = constant 512 : index
    %c784 = constant 784 : index
    %c256 = constant 256 : index
    %c128 = constant 128 : index
    %true = constant true
    %c24 = constant 24 : index
    %c32 = constant 32 : index
    %c40 = constant 40 : index
    %c48 = constant 48 : index
    %c3 = constant 3 : index
    %c56 = constant 56 : index
    %c64 = constant 64 : index
    %c4 = constant 4 : index
    %c72 = constant 72 : index
    %c9 = constant 9 : index
    %c80 = constant 80 : index
    %c5 = constant 5 : index
    %c88 = constant 88 : index
    %c11 = constant 11 : index
    %c96 = constant 96 : index
    %c6 = constant 6 : index
    %c104 = constant 104 : index
    %c13 = constant 13 : index
    %c112 = constant 112 : index
    %c-16 = constant -16 : index
    %c7 = constant 7 : index
    %c120 = constant 120 : index
    %c2 = constant 2 : index
    %c-1 = constant -1 : index
    %c-2 = constant -2 : index
    %c15 = constant 15 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c1 = constant 1 : index
    %c8 = constant 8 : index
    %0 = alloca() {alignment = 32 : i64} : memref<1x16xvector<8xf32>>
    %1 = alloca() {alignment = 32 : i64} : memref<1x16xvector<8xf32>>
    %2 = "accv.ref_global"() {global_name = @cache_16} : () -> memref<16x6x2xvector<8xf32>>
    %3 = "accv.ref_global"() {global_name = @cache_17} : () -> memref<16x128x2xvector<8xf32>>
    scf.for %arg3 = %c0 to %c512 step %c256 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        scf.for %arg5 = %c0 to %c256 step %c128 {
          scf.if %true {
            %4 = addi %arg3, %arg5 : index
            %5 = vector.transfer_read %arg1[%arg4, %4], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %5, %0[%c0, %c0] : memref<1x16xvector<8xf32>>
            %6 = addi %4, %c8 : index
            %7 = vector.transfer_read %arg1[%arg4, %6], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %7, %0[%c0, %c1] : memref<1x16xvector<8xf32>>
            %8 = addi %4, %c16 : index
            %9 = vector.transfer_read %arg1[%arg4, %8], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %9, %0[%c0, %c2] : memref<1x16xvector<8xf32>>
            %10 = addi %4, %c24 : index
            %11 = vector.transfer_read %arg1[%arg4, %10], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %11, %0[%c0, %c3] : memref<1x16xvector<8xf32>>
            %12 = addi %4, %c32 : index
            %13 = vector.transfer_read %arg1[%arg4, %12], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %13, %0[%c0, %c4] : memref<1x16xvector<8xf32>>
            %14 = addi %4, %c40 : index
            %15 = vector.transfer_read %arg1[%arg4, %14], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %15, %0[%c0, %c5] : memref<1x16xvector<8xf32>>
            %16 = addi %4, %c48 : index
            %17 = vector.transfer_read %arg1[%arg4, %16], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %17, %0[%c0, %c6] : memref<1x16xvector<8xf32>>
            %18 = addi %4, %c56 : index
            %19 = vector.transfer_read %arg1[%arg4, %18], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %19, %0[%c0, %c7] : memref<1x16xvector<8xf32>>
            %20 = addi %4, %c64 : index
            %21 = vector.transfer_read %arg1[%arg4, %20], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %21, %0[%c0, %c8] : memref<1x16xvector<8xf32>>
            %22 = addi %4, %c72 : index
            %23 = vector.transfer_read %arg1[%arg4, %22], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %23, %0[%c0, %c9] : memref<1x16xvector<8xf32>>
            %24 = addi %4, %c80 : index
            %25 = vector.transfer_read %arg1[%arg4, %24], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %25, %0[%c0, %c10] : memref<1x16xvector<8xf32>>
            %26 = addi %4, %c88 : index
            %27 = vector.transfer_read %arg1[%arg4, %26], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %27, %0[%c0, %c11] : memref<1x16xvector<8xf32>>
            %28 = addi %4, %c96 : index
            %29 = vector.transfer_read %arg1[%arg4, %28], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %29, %0[%c0, %c12] : memref<1x16xvector<8xf32>>
            %30 = addi %4, %c104 : index
            %31 = vector.transfer_read %arg1[%arg4, %30], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %31, %0[%c0, %c13] : memref<1x16xvector<8xf32>>
            %32 = addi %4, %c112 : index
            %33 = vector.transfer_read %arg1[%arg4, %32], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %33, %0[%c0, %c14] : memref<1x16xvector<8xf32>>
            %34 = addi %4, %c120 : index
            %35 = vector.transfer_read %arg1[%arg4, %34], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %35, %0[%c0, %c15] : memref<1x16xvector<8xf32>>
            %36 = load %0[%c0, %c0] : memref<1x16xvector<8xf32>>
            %37 = cmpi "slt", %arg5, %c0 : index
            %38 = subi %c-1, %arg5 : index
            %39 = select %37, %38, %arg5 : index
            %40 = divi_signed %39, %c16 : index
            %41 = subi %c-1, %40 : index
            %42 = select %37, %41, %40 : index
            %43 = remi_signed %42, %c16 : index
            %44 = cmpi "slt", %43, %c0 : index
            %45 = addi %43, %c16 : index
            %46 = select %44, %45, %43 : index
            %47 = remi_signed %arg4, %c128 : index
            %48 = cmpi "slt", %47, %c0 : index
            %49 = addi %47, %c128 : index
            %50 = select %48, %49, %47 : index
            %51 = remi_signed %arg5, %c16 : index
            %52 = cmpi "slt", %51, %c0 : index
            %53 = addi %51, %c16 : index
            %54 = select %52, %53, %51 : index
            %55 = cmpi "slt", %54, %c0 : index
            %56 = subi %c-1, %54 : index
            %57 = select %55, %56, %54 : index
            %58 = divi_signed %57, %c8 : index
            %59 = subi %c-1, %58 : index
            %60 = select %55, %59, %58 : index
            %61 = remi_signed %60, %c2 : index
            %62 = cmpi "slt", %61, %c0 : index
            %63 = addi %61, %c2 : index
            %64 = select %62, %63, %61 : index
            store %36, %3[%46, %50, %64] : memref<16x128x2xvector<8xf32>>
            %65 = load %0[%c0, %c1] : memref<1x16xvector<8xf32>>
            %66 = addi %arg5, %c8 : index
            %67 = cmpi "slt", %66, %c0 : index
            %68 = subi %c-1, %66 : index
            %69 = select %67, %68, %66 : index
            %70 = divi_signed %69, %c16 : index
            %71 = subi %c-1, %70 : index
            %72 = select %67, %71, %70 : index
            %73 = remi_signed %72, %c16 : index
            %74 = cmpi "slt", %73, %c0 : index
            %75 = addi %73, %c16 : index
            %76 = select %74, %75, %73 : index
            %77 = divi_signed %39, %c8 : index
            %78 = subi %c-1, %77 : index
            %79 = select %37, %78, %77 : index
            %80 = muli %72, %c-2 : index
            %81 = addi %79, %80 : index
            %82 = addi %81, %c1 : index
            %83 = cmpi "slt", %82, %c0 : index
            %84 = subi %c-1, %82 : index
            %85 = select %83, %84, %82 : index
            %86 = divi_signed %85, %c2 : index
            %87 = subi %c-1, %86 : index
            %88 = select %83, %87, %86 : index
            %89 = muli %88, %c-2 : index
            %90 = addi %81, %89 : index
            %91 = addi %90, %c1 : index
            store %65, %3[%76, %50, %91] : memref<16x128x2xvector<8xf32>>
            %92 = load %0[%c0, %c2] : memref<1x16xvector<8xf32>>
            %93 = addi %42, %c1 : index
            %94 = cmpi "slt", %93, %c0 : index
            %95 = subi %c-1, %93 : index
            %96 = select %94, %95, %93 : index
            %97 = divi_signed %96, %c16 : index
            %98 = subi %c-1, %97 : index
            %99 = select %94, %98, %97 : index
            %100 = muli %99, %c-16 : index
            %101 = addi %42, %100 : index
            %102 = addi %101, %c1 : index
            store %92, %3[%102, %50, %64] : memref<16x128x2xvector<8xf32>>
            %103 = load %0[%c0, %c3] : memref<1x16xvector<8xf32>>
            %104 = addi %arg5, %c24 : index
            %105 = cmpi "slt", %104, %c0 : index
            %106 = subi %c-1, %104 : index
            %107 = select %105, %106, %104 : index
            %108 = divi_signed %107, %c16 : index
            %109 = subi %c-1, %108 : index
            %110 = select %105, %109, %108 : index
            %111 = remi_signed %110, %c16 : index
            %112 = cmpi "slt", %111, %c0 : index
            %113 = addi %111, %c16 : index
            %114 = select %112, %113, %111 : index
            %115 = muli %110, %c-2 : index
            %116 = addi %79, %115 : index
            %117 = addi %116, %c3 : index
            %118 = cmpi "slt", %117, %c0 : index
            %119 = subi %c-1, %117 : index
            %120 = select %118, %119, %117 : index
            %121 = divi_signed %120, %c2 : index
            %122 = subi %c-1, %121 : index
            %123 = select %118, %122, %121 : index
            %124 = muli %123, %c-2 : index
            %125 = addi %116, %124 : index
            %126 = addi %125, %c3 : index
            store %103, %3[%114, %50, %126] : memref<16x128x2xvector<8xf32>>
            %127 = load %0[%c0, %c4] : memref<1x16xvector<8xf32>>
            %128 = addi %42, %c2 : index
            %129 = cmpi "slt", %128, %c0 : index
            %130 = subi %c-1, %128 : index
            %131 = select %129, %130, %128 : index
            %132 = divi_signed %131, %c16 : index
            %133 = subi %c-1, %132 : index
            %134 = select %129, %133, %132 : index
            %135 = muli %134, %c-16 : index
            %136 = addi %42, %135 : index
            %137 = addi %136, %c2 : index
            store %127, %3[%137, %50, %64] : memref<16x128x2xvector<8xf32>>
            %138 = load %0[%c0, %c5] : memref<1x16xvector<8xf32>>
            %139 = addi %arg5, %c40 : index
            %140 = cmpi "slt", %139, %c0 : index
            %141 = subi %c-1, %139 : index
            %142 = select %140, %141, %139 : index
            %143 = divi_signed %142, %c16 : index
            %144 = subi %c-1, %143 : index
            %145 = select %140, %144, %143 : index
            %146 = remi_signed %145, %c16 : index
            %147 = cmpi "slt", %146, %c0 : index
            %148 = addi %146, %c16 : index
            %149 = select %147, %148, %146 : index
            %150 = muli %145, %c-2 : index
            %151 = addi %79, %150 : index
            %152 = addi %151, %c5 : index
            %153 = cmpi "slt", %152, %c0 : index
            %154 = subi %c-1, %152 : index
            %155 = select %153, %154, %152 : index
            %156 = divi_signed %155, %c2 : index
            %157 = subi %c-1, %156 : index
            %158 = select %153, %157, %156 : index
            %159 = muli %158, %c-2 : index
            %160 = addi %151, %159 : index
            %161 = addi %160, %c5 : index
            store %138, %3[%149, %50, %161] : memref<16x128x2xvector<8xf32>>
            %162 = load %0[%c0, %c6] : memref<1x16xvector<8xf32>>
            %163 = addi %42, %c3 : index
            %164 = cmpi "slt", %163, %c0 : index
            %165 = subi %c-1, %163 : index
            %166 = select %164, %165, %163 : index
            %167 = divi_signed %166, %c16 : index
            %168 = subi %c-1, %167 : index
            %169 = select %164, %168, %167 : index
            %170 = muli %169, %c-16 : index
            %171 = addi %42, %170 : index
            %172 = addi %171, %c3 : index
            store %162, %3[%172, %50, %64] : memref<16x128x2xvector<8xf32>>
            %173 = load %0[%c0, %c7] : memref<1x16xvector<8xf32>>
            %174 = addi %arg5, %c56 : index
            %175 = cmpi "slt", %174, %c0 : index
            %176 = subi %c-1, %174 : index
            %177 = select %175, %176, %174 : index
            %178 = divi_signed %177, %c16 : index
            %179 = subi %c-1, %178 : index
            %180 = select %175, %179, %178 : index
            %181 = remi_signed %180, %c16 : index
            %182 = cmpi "slt", %181, %c0 : index
            %183 = addi %181, %c16 : index
            %184 = select %182, %183, %181 : index
            %185 = muli %180, %c-2 : index
            %186 = addi %79, %185 : index
            %187 = addi %186, %c7 : index
            %188 = cmpi "slt", %187, %c0 : index
            %189 = subi %c-1, %187 : index
            %190 = select %188, %189, %187 : index
            %191 = divi_signed %190, %c2 : index
            %192 = subi %c-1, %191 : index
            %193 = select %188, %192, %191 : index
            %194 = muli %193, %c-2 : index
            %195 = addi %186, %194 : index
            %196 = addi %195, %c7 : index
            store %173, %3[%184, %50, %196] : memref<16x128x2xvector<8xf32>>
            %197 = load %0[%c0, %c8] : memref<1x16xvector<8xf32>>
            %198 = addi %42, %c4 : index
            %199 = cmpi "slt", %198, %c0 : index
            %200 = subi %c-1, %198 : index
            %201 = select %199, %200, %198 : index
            %202 = divi_signed %201, %c16 : index
            %203 = subi %c-1, %202 : index
            %204 = select %199, %203, %202 : index
            %205 = muli %204, %c-16 : index
            %206 = addi %42, %205 : index
            %207 = addi %206, %c4 : index
            store %197, %3[%207, %50, %64] : memref<16x128x2xvector<8xf32>>
            %208 = load %0[%c0, %c9] : memref<1x16xvector<8xf32>>
            %209 = addi %arg5, %c72 : index
            %210 = cmpi "slt", %209, %c0 : index
            %211 = subi %c-1, %209 : index
            %212 = select %210, %211, %209 : index
            %213 = divi_signed %212, %c16 : index
            %214 = subi %c-1, %213 : index
            %215 = select %210, %214, %213 : index
            %216 = remi_signed %215, %c16 : index
            %217 = cmpi "slt", %216, %c0 : index
            %218 = addi %216, %c16 : index
            %219 = select %217, %218, %216 : index
            %220 = muli %215, %c-2 : index
            %221 = addi %79, %220 : index
            %222 = addi %221, %c9 : index
            %223 = cmpi "slt", %222, %c0 : index
            %224 = subi %c-1, %222 : index
            %225 = select %223, %224, %222 : index
            %226 = divi_signed %225, %c2 : index
            %227 = subi %c-1, %226 : index
            %228 = select %223, %227, %226 : index
            %229 = muli %228, %c-2 : index
            %230 = addi %221, %229 : index
            %231 = addi %230, %c9 : index
            store %208, %3[%219, %50, %231] : memref<16x128x2xvector<8xf32>>
            %232 = load %0[%c0, %c10] : memref<1x16xvector<8xf32>>
            %233 = addi %42, %c5 : index
            %234 = cmpi "slt", %233, %c0 : index
            %235 = subi %c-1, %233 : index
            %236 = select %234, %235, %233 : index
            %237 = divi_signed %236, %c16 : index
            %238 = subi %c-1, %237 : index
            %239 = select %234, %238, %237 : index
            %240 = muli %239, %c-16 : index
            %241 = addi %42, %240 : index
            %242 = addi %241, %c5 : index
            store %232, %3[%242, %50, %64] : memref<16x128x2xvector<8xf32>>
            %243 = load %0[%c0, %c11] : memref<1x16xvector<8xf32>>
            %244 = addi %arg5, %c88 : index
            %245 = cmpi "slt", %244, %c0 : index
            %246 = subi %c-1, %244 : index
            %247 = select %245, %246, %244 : index
            %248 = divi_signed %247, %c16 : index
            %249 = subi %c-1, %248 : index
            %250 = select %245, %249, %248 : index
            %251 = remi_signed %250, %c16 : index
            %252 = cmpi "slt", %251, %c0 : index
            %253 = addi %251, %c16 : index
            %254 = select %252, %253, %251 : index
            %255 = muli %250, %c-2 : index
            %256 = addi %79, %255 : index
            %257 = addi %256, %c11 : index
            %258 = cmpi "slt", %257, %c0 : index
            %259 = subi %c-1, %257 : index
            %260 = select %258, %259, %257 : index
            %261 = divi_signed %260, %c2 : index
            %262 = subi %c-1, %261 : index
            %263 = select %258, %262, %261 : index
            %264 = muli %263, %c-2 : index
            %265 = addi %256, %264 : index
            %266 = addi %265, %c11 : index
            store %243, %3[%254, %50, %266] : memref<16x128x2xvector<8xf32>>
            %267 = load %0[%c0, %c12] : memref<1x16xvector<8xf32>>
            %268 = addi %42, %c6 : index
            %269 = cmpi "slt", %268, %c0 : index
            %270 = subi %c-1, %268 : index
            %271 = select %269, %270, %268 : index
            %272 = divi_signed %271, %c16 : index
            %273 = subi %c-1, %272 : index
            %274 = select %269, %273, %272 : index
            %275 = muli %274, %c-16 : index
            %276 = addi %42, %275 : index
            %277 = addi %276, %c6 : index
            store %267, %3[%277, %50, %64] : memref<16x128x2xvector<8xf32>>
            %278 = load %0[%c0, %c13] : memref<1x16xvector<8xf32>>
            %279 = addi %arg5, %c104 : index
            %280 = cmpi "slt", %279, %c0 : index
            %281 = subi %c-1, %279 : index
            %282 = select %280, %281, %279 : index
            %283 = divi_signed %282, %c16 : index
            %284 = subi %c-1, %283 : index
            %285 = select %280, %284, %283 : index
            %286 = remi_signed %285, %c16 : index
            %287 = cmpi "slt", %286, %c0 : index
            %288 = addi %286, %c16 : index
            %289 = select %287, %288, %286 : index
            %290 = muli %285, %c-2 : index
            %291 = addi %79, %290 : index
            %292 = addi %291, %c13 : index
            %293 = cmpi "slt", %292, %c0 : index
            %294 = subi %c-1, %292 : index
            %295 = select %293, %294, %292 : index
            %296 = divi_signed %295, %c2 : index
            %297 = subi %c-1, %296 : index
            %298 = select %293, %297, %296 : index
            %299 = muli %298, %c-2 : index
            %300 = addi %291, %299 : index
            %301 = addi %300, %c13 : index
            store %278, %3[%289, %50, %301] : memref<16x128x2xvector<8xf32>>
            %302 = load %0[%c0, %c14] : memref<1x16xvector<8xf32>>
            %303 = addi %42, %c7 : index
            %304 = cmpi "slt", %303, %c0 : index
            %305 = subi %c-1, %303 : index
            %306 = select %304, %305, %303 : index
            %307 = divi_signed %306, %c16 : index
            %308 = subi %c-1, %307 : index
            %309 = select %304, %308, %307 : index
            %310 = muli %309, %c-16 : index
            %311 = addi %42, %310 : index
            %312 = addi %311, %c7 : index
            store %302, %3[%312, %50, %64] : memref<16x128x2xvector<8xf32>>
            %313 = load %0[%c0, %c15] : memref<1x16xvector<8xf32>>
            %314 = addi %arg5, %c120 : index
            %315 = cmpi "slt", %314, %c0 : index
            %316 = subi %c-1, %314 : index
            %317 = select %315, %316, %314 : index
            %318 = divi_signed %317, %c16 : index
            %319 = subi %c-1, %318 : index
            %320 = select %315, %319, %318 : index
            %321 = remi_signed %320, %c16 : index
            %322 = cmpi "slt", %321, %c0 : index
            %323 = addi %321, %c16 : index
            %324 = select %322, %323, %321 : index
            %325 = muli %320, %c-2 : index
            %326 = addi %79, %325 : index
            %327 = addi %326, %c15 : index
            %328 = cmpi "slt", %327, %c0 : index
            %329 = subi %c-1, %327 : index
            %330 = select %328, %329, %327 : index
            %331 = divi_signed %330, %c2 : index
            %332 = subi %c-1, %331 : index
            %333 = select %328, %332, %331 : index
            %334 = muli %333, %c-2 : index
            %335 = addi %326, %334 : index
            %336 = addi %335, %c15 : index
            store %313, %3[%324, %50, %336] : memref<16x128x2xvector<8xf32>>
          } else {
            %4 = addi %arg3, %arg5 : index
            %5 = vector.transfer_read %arg1[%arg4, %4], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %5, %0[%c0, %c0] : memref<1x16xvector<8xf32>>
            %6 = addi %4, %c8 : index
            %7 = vector.transfer_read %arg1[%arg4, %6], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %7, %0[%c0, %c1] : memref<1x16xvector<8xf32>>
            %8 = addi %4, %c16 : index
            %9 = vector.transfer_read %arg1[%arg4, %8], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %9, %0[%c0, %c2] : memref<1x16xvector<8xf32>>
            %10 = addi %4, %c24 : index
            %11 = vector.transfer_read %arg1[%arg4, %10], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %11, %0[%c0, %c3] : memref<1x16xvector<8xf32>>
            %12 = addi %4, %c32 : index
            %13 = vector.transfer_read %arg1[%arg4, %12], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %13, %0[%c0, %c4] : memref<1x16xvector<8xf32>>
            %14 = addi %4, %c40 : index
            %15 = vector.transfer_read %arg1[%arg4, %14], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %15, %0[%c0, %c5] : memref<1x16xvector<8xf32>>
            %16 = addi %4, %c48 : index
            %17 = vector.transfer_read %arg1[%arg4, %16], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %17, %0[%c0, %c6] : memref<1x16xvector<8xf32>>
            %18 = addi %4, %c56 : index
            %19 = vector.transfer_read %arg1[%arg4, %18], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %19, %0[%c0, %c7] : memref<1x16xvector<8xf32>>
            %20 = addi %4, %c64 : index
            %21 = vector.transfer_read %arg1[%arg4, %20], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %21, %0[%c0, %c8] : memref<1x16xvector<8xf32>>
            %22 = addi %4, %c72 : index
            %23 = vector.transfer_read %arg1[%arg4, %22], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %23, %0[%c0, %c9] : memref<1x16xvector<8xf32>>
            %24 = addi %4, %c80 : index
            %25 = vector.transfer_read %arg1[%arg4, %24], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %25, %0[%c0, %c10] : memref<1x16xvector<8xf32>>
            %26 = addi %4, %c88 : index
            %27 = vector.transfer_read %arg1[%arg4, %26], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %27, %0[%c0, %c11] : memref<1x16xvector<8xf32>>
            %28 = addi %4, %c96 : index
            %29 = vector.transfer_read %arg1[%arg4, %28], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %29, %0[%c0, %c12] : memref<1x16xvector<8xf32>>
            %30 = addi %4, %c104 : index
            %31 = vector.transfer_read %arg1[%arg4, %30], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %31, %0[%c0, %c13] : memref<1x16xvector<8xf32>>
            %32 = addi %4, %c112 : index
            %33 = vector.transfer_read %arg1[%arg4, %32], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %33, %0[%c0, %c14] : memref<1x16xvector<8xf32>>
            %34 = addi %4, %c120 : index
            %35 = vector.transfer_read %arg1[%arg4, %34], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            store %35, %0[%c0, %c15] : memref<1x16xvector<8xf32>>
            %36 = load %0[%c0, %c0] : memref<1x16xvector<8xf32>>
            %37 = cmpi "slt", %arg5, %c0 : index
            %38 = subi %c-1, %arg5 : index
            %39 = select %37, %38, %arg5 : index
            %40 = divi_signed %39, %c16 : index
            %41 = subi %c-1, %40 : index
            %42 = select %37, %41, %40 : index
            %43 = remi_signed %42, %c16 : index
            %44 = cmpi "slt", %43, %c0 : index
            %45 = addi %43, %c16 : index
            %46 = select %44, %45, %43 : index
            %47 = remi_signed %arg4, %c128 : index
            %48 = cmpi "slt", %47, %c0 : index
            %49 = addi %47, %c128 : index
            %50 = select %48, %49, %47 : index
            %51 = remi_signed %arg5, %c16 : index
            %52 = cmpi "slt", %51, %c0 : index
            %53 = addi %51, %c16 : index
            %54 = select %52, %53, %51 : index
            %55 = cmpi "slt", %54, %c0 : index
            %56 = subi %c-1, %54 : index
            %57 = select %55, %56, %54 : index
            %58 = divi_signed %57, %c8 : index
            %59 = subi %c-1, %58 : index
            %60 = select %55, %59, %58 : index
            %61 = remi_signed %60, %c2 : index
            %62 = cmpi "slt", %61, %c0 : index
            %63 = addi %61, %c2 : index
            %64 = select %62, %63, %61 : index
            store %36, %3[%46, %50, %64] : memref<16x128x2xvector<8xf32>>
            %65 = load %0[%c0, %c1] : memref<1x16xvector<8xf32>>
            %66 = addi %arg5, %c8 : index
            %67 = cmpi "slt", %66, %c0 : index
            %68 = subi %c-1, %66 : index
            %69 = select %67, %68, %66 : index
            %70 = divi_signed %69, %c16 : index
            %71 = subi %c-1, %70 : index
            %72 = select %67, %71, %70 : index
            %73 = remi_signed %72, %c16 : index
            %74 = cmpi "slt", %73, %c0 : index
            %75 = addi %73, %c16 : index
            %76 = select %74, %75, %73 : index
            %77 = divi_signed %39, %c8 : index
            %78 = subi %c-1, %77 : index
            %79 = select %37, %78, %77 : index
            %80 = muli %72, %c-2 : index
            %81 = addi %79, %80 : index
            %82 = addi %81, %c1 : index
            %83 = cmpi "slt", %82, %c0 : index
            %84 = subi %c-1, %82 : index
            %85 = select %83, %84, %82 : index
            %86 = divi_signed %85, %c2 : index
            %87 = subi %c-1, %86 : index
            %88 = select %83, %87, %86 : index
            %89 = muli %88, %c-2 : index
            %90 = addi %81, %89 : index
            %91 = addi %90, %c1 : index
            store %65, %3[%76, %50, %91] : memref<16x128x2xvector<8xf32>>
            %92 = load %0[%c0, %c2] : memref<1x16xvector<8xf32>>
            %93 = addi %42, %c1 : index
            %94 = cmpi "slt", %93, %c0 : index
            %95 = subi %c-1, %93 : index
            %96 = select %94, %95, %93 : index
            %97 = divi_signed %96, %c16 : index
            %98 = subi %c-1, %97 : index
            %99 = select %94, %98, %97 : index
            %100 = muli %99, %c-16 : index
            %101 = addi %42, %100 : index
            %102 = addi %101, %c1 : index
            store %92, %3[%102, %50, %64] : memref<16x128x2xvector<8xf32>>
            %103 = load %0[%c0, %c3] : memref<1x16xvector<8xf32>>
            %104 = addi %arg5, %c24 : index
            %105 = cmpi "slt", %104, %c0 : index
            %106 = subi %c-1, %104 : index
            %107 = select %105, %106, %104 : index
            %108 = divi_signed %107, %c16 : index
            %109 = subi %c-1, %108 : index
            %110 = select %105, %109, %108 : index
            %111 = remi_signed %110, %c16 : index
            %112 = cmpi "slt", %111, %c0 : index
            %113 = addi %111, %c16 : index
            %114 = select %112, %113, %111 : index
            %115 = muli %110, %c-2 : index
            %116 = addi %79, %115 : index
            %117 = addi %116, %c3 : index
            %118 = cmpi "slt", %117, %c0 : index
            %119 = subi %c-1, %117 : index
            %120 = select %118, %119, %117 : index
            %121 = divi_signed %120, %c2 : index
            %122 = subi %c-1, %121 : index
            %123 = select %118, %122, %121 : index
            %124 = muli %123, %c-2 : index
            %125 = addi %116, %124 : index
            %126 = addi %125, %c3 : index
            store %103, %3[%114, %50, %126] : memref<16x128x2xvector<8xf32>>
            %127 = load %0[%c0, %c4] : memref<1x16xvector<8xf32>>
            %128 = addi %42, %c2 : index
            %129 = cmpi "slt", %128, %c0 : index
            %130 = subi %c-1, %128 : index
            %131 = select %129, %130, %128 : index
            %132 = divi_signed %131, %c16 : index
            %133 = subi %c-1, %132 : index
            %134 = select %129, %133, %132 : index
            %135 = muli %134, %c-16 : index
            %136 = addi %42, %135 : index
            %137 = addi %136, %c2 : index
            store %127, %3[%137, %50, %64] : memref<16x128x2xvector<8xf32>>
            %138 = load %0[%c0, %c5] : memref<1x16xvector<8xf32>>
            %139 = addi %arg5, %c40 : index
            %140 = cmpi "slt", %139, %c0 : index
            %141 = subi %c-1, %139 : index
            %142 = select %140, %141, %139 : index
            %143 = divi_signed %142, %c16 : index
            %144 = subi %c-1, %143 : index
            %145 = select %140, %144, %143 : index
            %146 = remi_signed %145, %c16 : index
            %147 = cmpi "slt", %146, %c0 : index
            %148 = addi %146, %c16 : index
            %149 = select %147, %148, %146 : index
            %150 = muli %145, %c-2 : index
            %151 = addi %79, %150 : index
            %152 = addi %151, %c5 : index
            %153 = cmpi "slt", %152, %c0 : index
            %154 = subi %c-1, %152 : index
            %155 = select %153, %154, %152 : index
            %156 = divi_signed %155, %c2 : index
            %157 = subi %c-1, %156 : index
            %158 = select %153, %157, %156 : index
            %159 = muli %158, %c-2 : index
            %160 = addi %151, %159 : index
            %161 = addi %160, %c5 : index
            store %138, %3[%149, %50, %161] : memref<16x128x2xvector<8xf32>>
            %162 = load %0[%c0, %c6] : memref<1x16xvector<8xf32>>
            %163 = addi %42, %c3 : index
            %164 = cmpi "slt", %163, %c0 : index
            %165 = subi %c-1, %163 : index
            %166 = select %164, %165, %163 : index
            %167 = divi_signed %166, %c16 : index
            %168 = subi %c-1, %167 : index
            %169 = select %164, %168, %167 : index
            %170 = muli %169, %c-16 : index
            %171 = addi %42, %170 : index
            %172 = addi %171, %c3 : index
            store %162, %3[%172, %50, %64] : memref<16x128x2xvector<8xf32>>
            %173 = load %0[%c0, %c7] : memref<1x16xvector<8xf32>>
            %174 = addi %arg5, %c56 : index
            %175 = cmpi "slt", %174, %c0 : index
            %176 = subi %c-1, %174 : index
            %177 = select %175, %176, %174 : index
            %178 = divi_signed %177, %c16 : index
            %179 = subi %c-1, %178 : index
            %180 = select %175, %179, %178 : index
            %181 = remi_signed %180, %c16 : index
            %182 = cmpi "slt", %181, %c0 : index
            %183 = addi %181, %c16 : index
            %184 = select %182, %183, %181 : index
            %185 = muli %180, %c-2 : index
            %186 = addi %79, %185 : index
            %187 = addi %186, %c7 : index
            %188 = cmpi "slt", %187, %c0 : index
            %189 = subi %c-1, %187 : index
            %190 = select %188, %189, %187 : index
            %191 = divi_signed %190, %c2 : index
            %192 = subi %c-1, %191 : index
            %193 = select %188, %192, %191 : index
            %194 = muli %193, %c-2 : index
            %195 = addi %186, %194 : index
            %196 = addi %195, %c7 : index
            store %173, %3[%184, %50, %196] : memref<16x128x2xvector<8xf32>>
            %197 = load %0[%c0, %c8] : memref<1x16xvector<8xf32>>
            %198 = addi %42, %c4 : index
            %199 = cmpi "slt", %198, %c0 : index
            %200 = subi %c-1, %198 : index
            %201 = select %199, %200, %198 : index
            %202 = divi_signed %201, %c16 : index
            %203 = subi %c-1, %202 : index
            %204 = select %199, %203, %202 : index
            %205 = muli %204, %c-16 : index
            %206 = addi %42, %205 : index
            %207 = addi %206, %c4 : index
            store %197, %3[%207, %50, %64] : memref<16x128x2xvector<8xf32>>
            %208 = load %0[%c0, %c9] : memref<1x16xvector<8xf32>>
            %209 = addi %arg5, %c72 : index
            %210 = cmpi "slt", %209, %c0 : index
            %211 = subi %c-1, %209 : index
            %212 = select %210, %211, %209 : index
            %213 = divi_signed %212, %c16 : index
            %214 = subi %c-1, %213 : index
            %215 = select %210, %214, %213 : index
            %216 = remi_signed %215, %c16 : index
            %217 = cmpi "slt", %216, %c0 : index
            %218 = addi %216, %c16 : index
            %219 = select %217, %218, %216 : index
            %220 = muli %215, %c-2 : index
            %221 = addi %79, %220 : index
            %222 = addi %221, %c9 : index
            %223 = cmpi "slt", %222, %c0 : index
            %224 = subi %c-1, %222 : index
            %225 = select %223, %224, %222 : index
            %226 = divi_signed %225, %c2 : index
            %227 = subi %c-1, %226 : index
            %228 = select %223, %227, %226 : index
            %229 = muli %228, %c-2 : index
            %230 = addi %221, %229 : index
            %231 = addi %230, %c9 : index
            store %208, %3[%219, %50, %231] : memref<16x128x2xvector<8xf32>>
            %232 = load %0[%c0, %c10] : memref<1x16xvector<8xf32>>
            %233 = addi %42, %c5 : index
            %234 = cmpi "slt", %233, %c0 : index
            %235 = subi %c-1, %233 : index
            %236 = select %234, %235, %233 : index
            %237 = divi_signed %236, %c16 : index
            %238 = subi %c-1, %237 : index
            %239 = select %234, %238, %237 : index
            %240 = muli %239, %c-16 : index
            %241 = addi %42, %240 : index
            %242 = addi %241, %c5 : index
            store %232, %3[%242, %50, %64] : memref<16x128x2xvector<8xf32>>
            %243 = load %0[%c0, %c11] : memref<1x16xvector<8xf32>>
            %244 = addi %arg5, %c88 : index
            %245 = cmpi "slt", %244, %c0 : index
            %246 = subi %c-1, %244 : index
            %247 = select %245, %246, %244 : index
            %248 = divi_signed %247, %c16 : index
            %249 = subi %c-1, %248 : index
            %250 = select %245, %249, %248 : index
            %251 = remi_signed %250, %c16 : index
            %252 = cmpi "slt", %251, %c0 : index
            %253 = addi %251, %c16 : index
            %254 = select %252, %253, %251 : index
            %255 = muli %250, %c-2 : index
            %256 = addi %79, %255 : index
            %257 = addi %256, %c11 : index
            %258 = cmpi "slt", %257, %c0 : index
            %259 = subi %c-1, %257 : index
            %260 = select %258, %259, %257 : index
            %261 = divi_signed %260, %c2 : index
            %262 = subi %c-1, %261 : index
            %263 = select %258, %262, %261 : index
            %264 = muli %263, %c-2 : index
            %265 = addi %256, %264 : index
            %266 = addi %265, %c11 : index
            store %243, %3[%254, %50, %266] : memref<16x128x2xvector<8xf32>>
            %267 = load %0[%c0, %c12] : memref<1x16xvector<8xf32>>
            %268 = addi %42, %c6 : index
            %269 = cmpi "slt", %268, %c0 : index
            %270 = subi %c-1, %268 : index
            %271 = select %269, %270, %268 : index
            %272 = divi_signed %271, %c16 : index
            %273 = subi %c-1, %272 : index
            %274 = select %269, %273, %272 : index
            %275 = muli %274, %c-16 : index
            %276 = addi %42, %275 : index
            %277 = addi %276, %c6 : index
            store %267, %3[%277, %50, %64] : memref<16x128x2xvector<8xf32>>
            %278 = load %0[%c0, %c13] : memref<1x16xvector<8xf32>>
            %279 = addi %arg5, %c104 : index
            %280 = cmpi "slt", %279, %c0 : index
            %281 = subi %c-1, %279 : index
            %282 = select %280, %281, %279 : index
            %283 = divi_signed %282, %c16 : index
            %284 = subi %c-1, %283 : index
            %285 = select %280, %284, %283 : index
            %286 = remi_signed %285, %c16 : index
            %287 = cmpi "slt", %286, %c0 : index
            %288 = addi %286, %c16 : index
            %289 = select %287, %288, %286 : index
            %290 = muli %285, %c-2 : index
            %291 = addi %79, %290 : index
            %292 = addi %291, %c13 : index
            %293 = cmpi "slt", %292, %c0 : index
            %294 = subi %c-1, %292 : index
            %295 = select %293, %294, %292 : index
            %296 = divi_signed %295, %c2 : index
            %297 = subi %c-1, %296 : index
            %298 = select %293, %297, %296 : index
            %299 = muli %298, %c-2 : index
            %300 = addi %291, %299 : index
            %301 = addi %300, %c13 : index
            store %278, %3[%289, %50, %301] : memref<16x128x2xvector<8xf32>>
            %302 = load %0[%c0, %c14] : memref<1x16xvector<8xf32>>
            %303 = addi %42, %c7 : index
            %304 = cmpi "slt", %303, %c0 : index
            %305 = subi %c-1, %303 : index
            %306 = select %304, %305, %303 : index
            %307 = divi_signed %306, %c16 : index
            %308 = subi %c-1, %307 : index
            %309 = select %304, %308, %307 : index
            %310 = muli %309, %c-16 : index
            %311 = addi %42, %310 : index
            %312 = addi %311, %c7 : index
            store %302, %3[%312, %50, %64] : memref<16x128x2xvector<8xf32>>
            %313 = load %0[%c0, %c15] : memref<1x16xvector<8xf32>>
            %314 = addi %arg5, %c120 : index
            %315 = cmpi "slt", %314, %c0 : index
            %316 = subi %c-1, %314 : index
            %317 = select %315, %316, %314 : index
            %318 = divi_signed %317, %c16 : index
            %319 = subi %c-1, %318 : index
            %320 = select %315, %319, %318 : index
            %321 = remi_signed %320, %c16 : index
            %322 = cmpi "slt", %321, %c0 : index
            %323 = addi %321, %c16 : index
            %324 = select %322, %323, %321 : index
            %325 = muli %320, %c-2 : index
            %326 = addi %79, %325 : index
            %327 = addi %326, %c15 : index
            %328 = cmpi "slt", %327, %c0 : index
            %329 = subi %c-1, %327 : index
            %330 = select %328, %329, %327 : index
            %331 = divi_signed %330, %c2 : index
            %332 = subi %c-1, %331 : index
            %333 = select %328, %332, %331 : index
            %334 = muli %333, %c-2 : index
            %335 = addi %326, %334 : index
            %336 = addi %335, %c15 : index
            store %313, %3[%324, %50, %336] : memref<16x128x2xvector<8xf32>>
          }
        }
      }
      scf.for %arg4 = %c0 to %c784 step %c1 {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c6 step %c1 {
            scf.for %arg7 = %c0 to %c2 step %c1 {
              store %cst_0, %2[%arg5, %arg6, %arg7] : memref<16x6x2xvector<8xf32>>
            }
          }
        }
        scf.for %arg5 = %c0 to %c256 step %c16 {
          scf.for %arg6 = %c0 to %c128 step %c4 {
            scf.for %arg7 = %c0 to %c0 step %c6 {
              scf.for %arg8 = %c0 to %c4 step %c1 {
                scf.for %arg9 = %c0 to %c0 step %c1 {
                  %4 = addi %arg4, %arg7 : index
                  %5 = addi %4, %arg9 : index
                  %6 = addi %arg6, %arg8 : index
                  %7 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %8 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %9 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %10 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %11 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %12 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %13 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %14 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %15 = cmpi "slt", %arg5, %c0 : index
                  %16 = subi %c-1, %arg5 : index
                  %17 = select %15, %16, %arg5 : index
                  %18 = divi_signed %17, %c16 : index
                  %19 = subi %c-1, %18 : index
                  %20 = select %15, %19, %18 : index
                  %21 = remi_signed %20, %c16 : index
                  %22 = cmpi "slt", %21, %c0 : index
                  %23 = addi %21, %c16 : index
                  %24 = select %22, %23, %21 : index
                  %25 = remi_signed %6, %c128 : index
                  %26 = cmpi "slt", %25, %c0 : index
                  %27 = addi %25, %c128 : index
                  %28 = select %26, %27, %25 : index
                  %29 = remi_signed %arg5, %c16 : index
                  %30 = cmpi "slt", %29, %c0 : index
                  %31 = addi %29, %c16 : index
                  %32 = select %30, %31, %29 : index
                  %33 = cmpi "slt", %32, %c0 : index
                  %34 = subi %c-1, %32 : index
                  %35 = select %33, %34, %32 : index
                  %36 = divi_signed %35, %c8 : index
                  %37 = subi %c-1, %36 : index
                  %38 = select %33, %37, %36 : index
                  %39 = remi_signed %38, %c2 : index
                  %40 = cmpi "slt", %39, %c0 : index
                  %41 = addi %39, %c2 : index
                  %42 = select %40, %41, %39 : index
                  %43 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %44 = vector.extractelement %43[%c0_i64 : i64] : vector<8xf32>
                  %45 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %46 = vector.extractelement %45[%c1_i64 : i64] : vector<8xf32>
                  %47 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %48 = vector.extractelement %47[%c2_i64 : i64] : vector<8xf32>
                  %49 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %50 = vector.extractelement %49[%c3_i64 : i64] : vector<8xf32>
                  %51 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %52 = vector.extractelement %51[%c4_i64 : i64] : vector<8xf32>
                  %53 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %54 = vector.extractelement %53[%c5_i64 : i64] : vector<8xf32>
                  %55 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %56 = vector.extractelement %55[%c6_i64 : i64] : vector<8xf32>
                  %57 = load %3[%24, %28, %42] : memref<16x128x2xvector<8xf32>>
                  %58 = vector.extractelement %57[%c7_i64 : i64] : vector<8xf32>
                  %59 = mulf %7, %44 {RelaxedPrecision} : f32
                  %60 = mulf %8, %46 {RelaxedPrecision} : f32
                  %61 = mulf %9, %48 {RelaxedPrecision} : f32
                  %62 = mulf %10, %50 {RelaxedPrecision} : f32
                  %63 = mulf %11, %52 {RelaxedPrecision} : f32
                  %64 = mulf %12, %54 {RelaxedPrecision} : f32
                  %65 = mulf %13, %56 {RelaxedPrecision} : f32
                  %66 = mulf %14, %58 {RelaxedPrecision} : f32
                  %67 = addi %arg7, %arg9 : index
                  %68 = remi_signed %67, %c6 : index
                  %69 = cmpi "slt", %68, %c0 : index
                  %70 = addi %68, %c6 : index
                  %71 = select %69, %70, %68 : index
                  %72 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %73 = vector.extractelement %72[%c0_i64 : i64] : vector<8xf32>
                  %74 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %75 = vector.extractelement %74[%c1_i64 : i64] : vector<8xf32>
                  %76 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %77 = vector.extractelement %76[%c2_i64 : i64] : vector<8xf32>
                  %78 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %79 = vector.extractelement %78[%c3_i64 : i64] : vector<8xf32>
                  %80 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %81 = vector.extractelement %80[%c4_i64 : i64] : vector<8xf32>
                  %82 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %83 = vector.extractelement %82[%c5_i64 : i64] : vector<8xf32>
                  %84 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %85 = vector.extractelement %84[%c6_i64 : i64] : vector<8xf32>
                  %86 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %87 = vector.extractelement %86[%c7_i64 : i64] : vector<8xf32>
                  %88 = addf %73, %59 {RelaxedPrecision} : f32
                  %89 = addf %75, %60 {RelaxedPrecision} : f32
                  %90 = addf %77, %61 {RelaxedPrecision} : f32
                  %91 = addf %79, %62 {RelaxedPrecision} : f32
                  %92 = addf %81, %63 {RelaxedPrecision} : f32
                  %93 = addf %83, %64 {RelaxedPrecision} : f32
                  %94 = addf %85, %65 {RelaxedPrecision} : f32
                  %95 = addf %87, %66 {RelaxedPrecision} : f32
                  %96 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %97 = vector.insertelement %88, %96[%c0_i64 : i64] : vector<8xf32>
                  store %97, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %98 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %99 = vector.insertelement %89, %98[%c1_i64 : i64] : vector<8xf32>
                  store %99, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %100 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %101 = vector.insertelement %90, %100[%c2_i64 : i64] : vector<8xf32>
                  store %101, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %102 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %103 = vector.insertelement %91, %102[%c3_i64 : i64] : vector<8xf32>
                  store %103, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %104 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %105 = vector.insertelement %92, %104[%c4_i64 : i64] : vector<8xf32>
                  store %105, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %106 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %107 = vector.insertelement %93, %106[%c5_i64 : i64] : vector<8xf32>
                  store %107, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %108 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %109 = vector.insertelement %94, %108[%c6_i64 : i64] : vector<8xf32>
                  store %109, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %110 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %111 = vector.insertelement %95, %110[%c7_i64 : i64] : vector<8xf32>
                  store %111, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %112 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %113 = vector.insertelement %88, %112[%c0_i64 : i64] : vector<8xf32>
                  store %113, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %114 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %115 = vector.insertelement %89, %114[%c1_i64 : i64] : vector<8xf32>
                  store %115, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %116 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %117 = vector.insertelement %90, %116[%c2_i64 : i64] : vector<8xf32>
                  store %117, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %118 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %119 = vector.insertelement %91, %118[%c3_i64 : i64] : vector<8xf32>
                  store %119, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %120 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %121 = vector.insertelement %92, %120[%c4_i64 : i64] : vector<8xf32>
                  store %121, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %122 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %123 = vector.insertelement %93, %122[%c5_i64 : i64] : vector<8xf32>
                  store %123, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %124 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %125 = vector.insertelement %94, %124[%c6_i64 : i64] : vector<8xf32>
                  store %125, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %126 = load %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %127 = vector.insertelement %95, %126[%c7_i64 : i64] : vector<8xf32>
                  store %127, %2[%24, %71, %42] : memref<16x6x2xvector<8xf32>>
                  %128 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %129 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %130 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %131 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %132 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %133 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %134 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %135 = load %arg0[%5, %6] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
                  %136 = addi %arg5, %c8 : index
                  %137 = cmpi "slt", %136, %c0 : index
                  %138 = subi %c-1, %136 : index
                  %139 = select %137, %138, %136 : index
                  %140 = divi_signed %139, %c16 : index
                  %141 = subi %c-1, %140 : index
                  %142 = select %137, %141, %140 : index
                  %143 = remi_signed %142, %c16 : index
                  %144 = cmpi "slt", %143, %c0 : index
                  %145 = addi %143, %c16 : index
                  %146 = select %144, %145, %143 : index
                  %147 = divi_signed %17, %c8 : index
                  %148 = subi %c-1, %147 : index
                  %149 = select %15, %148, %147 : index
                  %150 = muli %142, %c-2 : index
                  %151 = addi %149, %150 : index
                  %152 = addi %151, %c1 : index
                  %153 = cmpi "slt", %152, %c0 : index
                  %154 = subi %c-1, %152 : index
                  %155 = select %153, %154, %152 : index
                  %156 = divi_signed %155, %c2 : index
                  %157 = subi %c-1, %156 : index
                  %158 = select %153, %157, %156 : index
                  %159 = muli %158, %c-2 : index
                  %160 = addi %151, %159 : index
                  %161 = addi %160, %c1 : index
                  %162 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %163 = vector.extractelement %162[%c0_i64 : i64] : vector<8xf32>
                  %164 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %165 = vector.extractelement %164[%c1_i64 : i64] : vector<8xf32>
                  %166 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %167 = vector.extractelement %166[%c2_i64 : i64] : vector<8xf32>
                  %168 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %169 = vector.extractelement %168[%c3_i64 : i64] : vector<8xf32>
                  %170 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %171 = vector.extractelement %170[%c4_i64 : i64] : vector<8xf32>
                  %172 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %173 = vector.extractelement %172[%c5_i64 : i64] : vector<8xf32>
                  %174 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %175 = vector.extractelement %174[%c6_i64 : i64] : vector<8xf32>
                  %176 = load %3[%146, %28, %161] : memref<16x128x2xvector<8xf32>>
                  %177 = vector.extractelement %176[%c7_i64 : i64] : vector<8xf32>
                  %178 = mulf %128, %163 {RelaxedPrecision} : f32
                  %179 = mulf %129, %165 {RelaxedPrecision} : f32
                  %180 = mulf %130, %167 {RelaxedPrecision} : f32
                  %181 = mulf %131, %169 {RelaxedPrecision} : f32
                  %182 = mulf %132, %171 {RelaxedPrecision} : f32
                  %183 = mulf %133, %173 {RelaxedPrecision} : f32
                  %184 = mulf %134, %175 {RelaxedPrecision} : f32
                  %185 = mulf %135, %177 {RelaxedPrecision} : f32
                  %186 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %187 = vector.extractelement %186[%c0_i64 : i64] : vector<8xf32>
                  %188 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %189 = vector.extractelement %188[%c1_i64 : i64] : vector<8xf32>
                  %190 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %191 = vector.extractelement %190[%c2_i64 : i64] : vector<8xf32>
                  %192 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %193 = vector.extractelement %192[%c3_i64 : i64] : vector<8xf32>
                  %194 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %195 = vector.extractelement %194[%c4_i64 : i64] : vector<8xf32>
                  %196 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %197 = vector.extractelement %196[%c5_i64 : i64] : vector<8xf32>
                  %198 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %199 = vector.extractelement %198[%c6_i64 : i64] : vector<8xf32>
                  %200 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %201 = vector.extractelement %200[%c7_i64 : i64] : vector<8xf32>
                  %202 = addf %187, %178 {RelaxedPrecision} : f32
                  %203 = addf %189, %179 {RelaxedPrecision} : f32
                  %204 = addf %191, %180 {RelaxedPrecision} : f32
                  %205 = addf %193, %181 {RelaxedPrecision} : f32
                  %206 = addf %195, %182 {RelaxedPrecision} : f32
                  %207 = addf %197, %183 {RelaxedPrecision} : f32
                  %208 = addf %199, %184 {RelaxedPrecision} : f32
                  %209 = addf %201, %185 {RelaxedPrecision} : f32
                  %210 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %211 = vector.insertelement %202, %210[%c0_i64 : i64] : vector<8xf32>
                  store %211, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %212 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %213 = vector.insertelement %203, %212[%c1_i64 : i64] : vector<8xf32>
                  store %213, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %214 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %215 = vector.insertelement %204, %214[%c2_i64 : i64] : vector<8xf32>
                  store %215, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %216 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %217 = vector.insertelement %205, %216[%c3_i64 : i64] : vector<8xf32>
                  store %217, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %218 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %219 = vector.insertelement %206, %218[%c4_i64 : i64] : vector<8xf32>
                  store %219, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %220 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %221 = vector.insertelement %207, %220[%c5_i64 : i64] : vector<8xf32>
                  store %221, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %222 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %223 = vector.insertelement %208, %222[%c6_i64 : i64] : vector<8xf32>
                  store %223, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %224 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %225 = vector.insertelement %209, %224[%c7_i64 : i64] : vector<8xf32>
                  store %225, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %226 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %227 = vector.insertelement %202, %226[%c0_i64 : i64] : vector<8xf32>
                  store %227, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %228 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %229 = vector.insertelement %203, %228[%c1_i64 : i64] : vector<8xf32>
                  store %229, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %230 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %231 = vector.insertelement %204, %230[%c2_i64 : i64] : vector<8xf32>
                  store %231, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %232 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %233 = vector.insertelement %205, %232[%c3_i64 : i64] : vector<8xf32>
                  store %233, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %234 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %235 = vector.insertelement %206, %234[%c4_i64 : i64] : vector<8xf32>
                  store %235, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %236 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %237 = vector.insertelement %207, %236[%c5_i64 : i64] : vector<8xf32>
                  store %237, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %238 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %239 = vector.insertelement %208, %238[%c6_i64 : i64] : vector<8xf32>
                  store %239, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %240 = load %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                  %241 = vector.insertelement %209, %240[%c7_i64 : i64] : vector<8xf32>
                  store %241, %2[%146, %71, %161] : memref<16x6x2xvector<8xf32>>
                }
              }
            }
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %4 = addi %arg6, %arg7 : index
              %5 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %6 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %7 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %8 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %9 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %10 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %11 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %12 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %13 = cmpi "slt", %arg5, %c0 : index
              %14 = subi %c-1, %arg5 : index
              %15 = select %13, %14, %arg5 : index
              %16 = divi_signed %15, %c16 : index
              %17 = subi %c-1, %16 : index
              %18 = select %13, %17, %16 : index
              %19 = remi_signed %18, %c16 : index
              %20 = cmpi "slt", %19, %c0 : index
              %21 = addi %19, %c16 : index
              %22 = select %20, %21, %19 : index
              %23 = remi_signed %4, %c128 : index
              %24 = cmpi "slt", %23, %c0 : index
              %25 = addi %23, %c128 : index
              %26 = select %24, %25, %23 : index
              %27 = remi_signed %arg5, %c16 : index
              %28 = cmpi "slt", %27, %c0 : index
              %29 = addi %27, %c16 : index
              %30 = select %28, %29, %27 : index
              %31 = cmpi "slt", %30, %c0 : index
              %32 = subi %c-1, %30 : index
              %33 = select %31, %32, %30 : index
              %34 = divi_signed %33, %c8 : index
              %35 = subi %c-1, %34 : index
              %36 = select %31, %35, %34 : index
              %37 = remi_signed %36, %c2 : index
              %38 = cmpi "slt", %37, %c0 : index
              %39 = addi %37, %c2 : index
              %40 = select %38, %39, %37 : index
              %41 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %42 = vector.extractelement %41[%c0_i64 : i64] : vector<8xf32>
              %43 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %44 = vector.extractelement %43[%c1_i64 : i64] : vector<8xf32>
              %45 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %46 = vector.extractelement %45[%c2_i64 : i64] : vector<8xf32>
              %47 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %48 = vector.extractelement %47[%c3_i64 : i64] : vector<8xf32>
              %49 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %50 = vector.extractelement %49[%c4_i64 : i64] : vector<8xf32>
              %51 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %52 = vector.extractelement %51[%c5_i64 : i64] : vector<8xf32>
              %53 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %54 = vector.extractelement %53[%c6_i64 : i64] : vector<8xf32>
              %55 = load %3[%22, %26, %40] : memref<16x128x2xvector<8xf32>>
              %56 = vector.extractelement %55[%c7_i64 : i64] : vector<8xf32>
              %57 = mulf %5, %42 {RelaxedPrecision} : f32
              %58 = mulf %6, %44 {RelaxedPrecision} : f32
              %59 = mulf %7, %46 {RelaxedPrecision} : f32
              %60 = mulf %8, %48 {RelaxedPrecision} : f32
              %61 = mulf %9, %50 {RelaxedPrecision} : f32
              %62 = mulf %10, %52 {RelaxedPrecision} : f32
              %63 = mulf %11, %54 {RelaxedPrecision} : f32
              %64 = mulf %12, %56 {RelaxedPrecision} : f32
              %65 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %66 = vector.extractelement %65[%c0_i64 : i64] : vector<8xf32>
              %67 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %68 = vector.extractelement %67[%c1_i64 : i64] : vector<8xf32>
              %69 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %70 = vector.extractelement %69[%c2_i64 : i64] : vector<8xf32>
              %71 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %72 = vector.extractelement %71[%c3_i64 : i64] : vector<8xf32>
              %73 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %74 = vector.extractelement %73[%c4_i64 : i64] : vector<8xf32>
              %75 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %76 = vector.extractelement %75[%c5_i64 : i64] : vector<8xf32>
              %77 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %78 = vector.extractelement %77[%c6_i64 : i64] : vector<8xf32>
              %79 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %80 = vector.extractelement %79[%c7_i64 : i64] : vector<8xf32>
              %81 = addf %66, %57 {RelaxedPrecision} : f32
              %82 = addf %68, %58 {RelaxedPrecision} : f32
              %83 = addf %70, %59 {RelaxedPrecision} : f32
              %84 = addf %72, %60 {RelaxedPrecision} : f32
              %85 = addf %74, %61 {RelaxedPrecision} : f32
              %86 = addf %76, %62 {RelaxedPrecision} : f32
              %87 = addf %78, %63 {RelaxedPrecision} : f32
              %88 = addf %80, %64 {RelaxedPrecision} : f32
              %89 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %90 = vector.insertelement %81, %89[%c0_i64 : i64] : vector<8xf32>
              store %90, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %91 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %92 = vector.insertelement %82, %91[%c1_i64 : i64] : vector<8xf32>
              store %92, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %93 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %94 = vector.insertelement %83, %93[%c2_i64 : i64] : vector<8xf32>
              store %94, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %95 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %96 = vector.insertelement %84, %95[%c3_i64 : i64] : vector<8xf32>
              store %96, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %97 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %98 = vector.insertelement %85, %97[%c4_i64 : i64] : vector<8xf32>
              store %98, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %99 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %100 = vector.insertelement %86, %99[%c5_i64 : i64] : vector<8xf32>
              store %100, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %101 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %102 = vector.insertelement %87, %101[%c6_i64 : i64] : vector<8xf32>
              store %102, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %103 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %104 = vector.insertelement %88, %103[%c7_i64 : i64] : vector<8xf32>
              store %104, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %105 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %106 = vector.insertelement %81, %105[%c0_i64 : i64] : vector<8xf32>
              store %106, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %107 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %108 = vector.insertelement %82, %107[%c1_i64 : i64] : vector<8xf32>
              store %108, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %109 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %110 = vector.insertelement %83, %109[%c2_i64 : i64] : vector<8xf32>
              store %110, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %111 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %112 = vector.insertelement %84, %111[%c3_i64 : i64] : vector<8xf32>
              store %112, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %113 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %114 = vector.insertelement %85, %113[%c4_i64 : i64] : vector<8xf32>
              store %114, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %115 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %116 = vector.insertelement %86, %115[%c5_i64 : i64] : vector<8xf32>
              store %116, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %117 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %118 = vector.insertelement %87, %117[%c6_i64 : i64] : vector<8xf32>
              store %118, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %119 = load %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %120 = vector.insertelement %88, %119[%c7_i64 : i64] : vector<8xf32>
              store %120, %2[%22, %c0, %40] : memref<16x6x2xvector<8xf32>>
              %121 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %122 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %123 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %124 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %125 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %126 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %127 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %128 = load %arg0[%arg4, %4] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
              %129 = addi %arg5, %c8 : index
              %130 = cmpi "slt", %129, %c0 : index
              %131 = subi %c-1, %129 : index
              %132 = select %130, %131, %129 : index
              %133 = divi_signed %132, %c16 : index
              %134 = subi %c-1, %133 : index
              %135 = select %130, %134, %133 : index
              %136 = remi_signed %135, %c16 : index
              %137 = cmpi "slt", %136, %c0 : index
              %138 = addi %136, %c16 : index
              %139 = select %137, %138, %136 : index
              %140 = divi_signed %15, %c8 : index
              %141 = subi %c-1, %140 : index
              %142 = select %13, %141, %140 : index
              %143 = muli %135, %c-2 : index
              %144 = addi %142, %143 : index
              %145 = addi %144, %c1 : index
              %146 = cmpi "slt", %145, %c0 : index
              %147 = subi %c-1, %145 : index
              %148 = select %146, %147, %145 : index
              %149 = divi_signed %148, %c2 : index
              %150 = subi %c-1, %149 : index
              %151 = select %146, %150, %149 : index
              %152 = muli %151, %c-2 : index
              %153 = addi %144, %152 : index
              %154 = addi %153, %c1 : index
              %155 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %156 = vector.extractelement %155[%c0_i64 : i64] : vector<8xf32>
              %157 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %158 = vector.extractelement %157[%c1_i64 : i64] : vector<8xf32>
              %159 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %160 = vector.extractelement %159[%c2_i64 : i64] : vector<8xf32>
              %161 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %162 = vector.extractelement %161[%c3_i64 : i64] : vector<8xf32>
              %163 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %164 = vector.extractelement %163[%c4_i64 : i64] : vector<8xf32>
              %165 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %166 = vector.extractelement %165[%c5_i64 : i64] : vector<8xf32>
              %167 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %168 = vector.extractelement %167[%c6_i64 : i64] : vector<8xf32>
              %169 = load %3[%139, %26, %154] : memref<16x128x2xvector<8xf32>>
              %170 = vector.extractelement %169[%c7_i64 : i64] : vector<8xf32>
              %171 = mulf %121, %156 {RelaxedPrecision} : f32
              %172 = mulf %122, %158 {RelaxedPrecision} : f32
              %173 = mulf %123, %160 {RelaxedPrecision} : f32
              %174 = mulf %124, %162 {RelaxedPrecision} : f32
              %175 = mulf %125, %164 {RelaxedPrecision} : f32
              %176 = mulf %126, %166 {RelaxedPrecision} : f32
              %177 = mulf %127, %168 {RelaxedPrecision} : f32
              %178 = mulf %128, %170 {RelaxedPrecision} : f32
              %179 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %180 = vector.extractelement %179[%c0_i64 : i64] : vector<8xf32>
              %181 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %182 = vector.extractelement %181[%c1_i64 : i64] : vector<8xf32>
              %183 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %184 = vector.extractelement %183[%c2_i64 : i64] : vector<8xf32>
              %185 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %186 = vector.extractelement %185[%c3_i64 : i64] : vector<8xf32>
              %187 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %188 = vector.extractelement %187[%c4_i64 : i64] : vector<8xf32>
              %189 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %190 = vector.extractelement %189[%c5_i64 : i64] : vector<8xf32>
              %191 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %192 = vector.extractelement %191[%c6_i64 : i64] : vector<8xf32>
              %193 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %194 = vector.extractelement %193[%c7_i64 : i64] : vector<8xf32>
              %195 = addf %180, %171 {RelaxedPrecision} : f32
              %196 = addf %182, %172 {RelaxedPrecision} : f32
              %197 = addf %184, %173 {RelaxedPrecision} : f32
              %198 = addf %186, %174 {RelaxedPrecision} : f32
              %199 = addf %188, %175 {RelaxedPrecision} : f32
              %200 = addf %190, %176 {RelaxedPrecision} : f32
              %201 = addf %192, %177 {RelaxedPrecision} : f32
              %202 = addf %194, %178 {RelaxedPrecision} : f32
              %203 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %204 = vector.insertelement %195, %203[%c0_i64 : i64] : vector<8xf32>
              store %204, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %205 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %206 = vector.insertelement %196, %205[%c1_i64 : i64] : vector<8xf32>
              store %206, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %207 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %208 = vector.insertelement %197, %207[%c2_i64 : i64] : vector<8xf32>
              store %208, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %209 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %210 = vector.insertelement %198, %209[%c3_i64 : i64] : vector<8xf32>
              store %210, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %211 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %212 = vector.insertelement %199, %211[%c4_i64 : i64] : vector<8xf32>
              store %212, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %213 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %214 = vector.insertelement %200, %213[%c5_i64 : i64] : vector<8xf32>
              store %214, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %215 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %216 = vector.insertelement %201, %215[%c6_i64 : i64] : vector<8xf32>
              store %216, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %217 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %218 = vector.insertelement %202, %217[%c7_i64 : i64] : vector<8xf32>
              store %218, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %219 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %220 = vector.insertelement %195, %219[%c0_i64 : i64] : vector<8xf32>
              store %220, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %221 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %222 = vector.insertelement %196, %221[%c1_i64 : i64] : vector<8xf32>
              store %222, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %223 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %224 = vector.insertelement %197, %223[%c2_i64 : i64] : vector<8xf32>
              store %224, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %225 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %226 = vector.insertelement %198, %225[%c3_i64 : i64] : vector<8xf32>
              store %226, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %227 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %228 = vector.insertelement %199, %227[%c4_i64 : i64] : vector<8xf32>
              store %228, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %229 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %230 = vector.insertelement %200, %229[%c5_i64 : i64] : vector<8xf32>
              store %230, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %231 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %232 = vector.insertelement %201, %231[%c6_i64 : i64] : vector<8xf32>
              store %232, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %233 = load %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
              %234 = vector.insertelement %202, %233[%c7_i64 : i64] : vector<8xf32>
              store %234, %2[%139, %c0, %154] : memref<16x6x2xvector<8xf32>>
            }
          }
        }
        scf.for %arg5 = %c0 to %c256 step %c128 {
          scf.if %true {
            %4 = addi %arg3, %arg5 : index
            %5 = vector.transfer_read %arg2[%arg4, %4], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %6 = cmpi "slt", %arg5, %c0 : index
            %7 = subi %c-1, %arg5 : index
            %8 = select %6, %7, %arg5 : index
            %9 = divi_signed %8, %c16 : index
            %10 = subi %c-1, %9 : index
            %11 = select %6, %10, %9 : index
            %12 = remi_signed %11, %c16 : index
            %13 = cmpi "slt", %12, %c0 : index
            %14 = addi %12, %c16 : index
            %15 = select %13, %14, %12 : index
            %16 = remi_signed %arg5, %c16 : index
            %17 = cmpi "slt", %16, %c0 : index
            %18 = addi %16, %c16 : index
            %19 = select %17, %18, %16 : index
            %20 = cmpi "slt", %19, %c0 : index
            %21 = subi %c-1, %19 : index
            %22 = select %20, %21, %19 : index
            %23 = divi_signed %22, %c8 : index
            %24 = subi %c-1, %23 : index
            %25 = select %20, %24, %23 : index
            %26 = remi_signed %25, %c2 : index
            %27 = cmpi "slt", %26, %c0 : index
            %28 = addi %26, %c2 : index
            %29 = select %27, %28, %26 : index
            %30 = load %2[%15, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %31 = addf %5, %30 : vector<8xf32>
            store %31, %1[%c0, %c0] : memref<1x16xvector<8xf32>>
            %32 = addi %4, %c8 : index
            %33 = vector.transfer_read %arg2[%arg4, %32], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %34 = addi %arg5, %c8 : index
            %35 = cmpi "slt", %34, %c0 : index
            %36 = subi %c-1, %34 : index
            %37 = select %35, %36, %34 : index
            %38 = divi_signed %37, %c16 : index
            %39 = subi %c-1, %38 : index
            %40 = select %35, %39, %38 : index
            %41 = remi_signed %40, %c16 : index
            %42 = cmpi "slt", %41, %c0 : index
            %43 = addi %41, %c16 : index
            %44 = select %42, %43, %41 : index
            %45 = divi_signed %8, %c8 : index
            %46 = subi %c-1, %45 : index
            %47 = select %6, %46, %45 : index
            %48 = muli %40, %c-2 : index
            %49 = addi %47, %48 : index
            %50 = addi %49, %c1 : index
            %51 = cmpi "slt", %50, %c0 : index
            %52 = subi %c-1, %50 : index
            %53 = select %51, %52, %50 : index
            %54 = divi_signed %53, %c2 : index
            %55 = subi %c-1, %54 : index
            %56 = select %51, %55, %54 : index
            %57 = muli %56, %c-2 : index
            %58 = addi %49, %57 : index
            %59 = addi %58, %c1 : index
            %60 = load %2[%44, %c0, %59] : memref<16x6x2xvector<8xf32>>
            %61 = addf %33, %60 : vector<8xf32>
            store %61, %1[%c0, %c1] : memref<1x16xvector<8xf32>>
            %62 = addi %4, %c16 : index
            %63 = vector.transfer_read %arg2[%arg4, %62], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %64 = addi %11, %c1 : index
            %65 = cmpi "slt", %64, %c0 : index
            %66 = subi %c-1, %64 : index
            %67 = select %65, %66, %64 : index
            %68 = divi_signed %67, %c16 : index
            %69 = subi %c-1, %68 : index
            %70 = select %65, %69, %68 : index
            %71 = muli %70, %c-16 : index
            %72 = addi %11, %71 : index
            %73 = addi %72, %c1 : index
            %74 = load %2[%73, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %75 = addf %63, %74 : vector<8xf32>
            store %75, %1[%c0, %c2] : memref<1x16xvector<8xf32>>
            %76 = addi %4, %c24 : index
            %77 = vector.transfer_read %arg2[%arg4, %76], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %78 = addi %arg5, %c24 : index
            %79 = cmpi "slt", %78, %c0 : index
            %80 = subi %c-1, %78 : index
            %81 = select %79, %80, %78 : index
            %82 = divi_signed %81, %c16 : index
            %83 = subi %c-1, %82 : index
            %84 = select %79, %83, %82 : index
            %85 = remi_signed %84, %c16 : index
            %86 = cmpi "slt", %85, %c0 : index
            %87 = addi %85, %c16 : index
            %88 = select %86, %87, %85 : index
            %89 = muli %84, %c-2 : index
            %90 = addi %47, %89 : index
            %91 = addi %90, %c3 : index
            %92 = cmpi "slt", %91, %c0 : index
            %93 = subi %c-1, %91 : index
            %94 = select %92, %93, %91 : index
            %95 = divi_signed %94, %c2 : index
            %96 = subi %c-1, %95 : index
            %97 = select %92, %96, %95 : index
            %98 = muli %97, %c-2 : index
            %99 = addi %90, %98 : index
            %100 = addi %99, %c3 : index
            %101 = load %2[%88, %c0, %100] : memref<16x6x2xvector<8xf32>>
            %102 = addf %77, %101 : vector<8xf32>
            store %102, %1[%c0, %c3] : memref<1x16xvector<8xf32>>
            %103 = addi %4, %c32 : index
            %104 = vector.transfer_read %arg2[%arg4, %103], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %105 = addi %11, %c2 : index
            %106 = cmpi "slt", %105, %c0 : index
            %107 = subi %c-1, %105 : index
            %108 = select %106, %107, %105 : index
            %109 = divi_signed %108, %c16 : index
            %110 = subi %c-1, %109 : index
            %111 = select %106, %110, %109 : index
            %112 = muli %111, %c-16 : index
            %113 = addi %11, %112 : index
            %114 = addi %113, %c2 : index
            %115 = load %2[%114, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %116 = addf %104, %115 : vector<8xf32>
            store %116, %1[%c0, %c4] : memref<1x16xvector<8xf32>>
            %117 = addi %4, %c40 : index
            %118 = vector.transfer_read %arg2[%arg4, %117], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %119 = addi %arg5, %c40 : index
            %120 = cmpi "slt", %119, %c0 : index
            %121 = subi %c-1, %119 : index
            %122 = select %120, %121, %119 : index
            %123 = divi_signed %122, %c16 : index
            %124 = subi %c-1, %123 : index
            %125 = select %120, %124, %123 : index
            %126 = remi_signed %125, %c16 : index
            %127 = cmpi "slt", %126, %c0 : index
            %128 = addi %126, %c16 : index
            %129 = select %127, %128, %126 : index
            %130 = muli %125, %c-2 : index
            %131 = addi %47, %130 : index
            %132 = addi %131, %c5 : index
            %133 = cmpi "slt", %132, %c0 : index
            %134 = subi %c-1, %132 : index
            %135 = select %133, %134, %132 : index
            %136 = divi_signed %135, %c2 : index
            %137 = subi %c-1, %136 : index
            %138 = select %133, %137, %136 : index
            %139 = muli %138, %c-2 : index
            %140 = addi %131, %139 : index
            %141 = addi %140, %c5 : index
            %142 = load %2[%129, %c0, %141] : memref<16x6x2xvector<8xf32>>
            %143 = addf %118, %142 : vector<8xf32>
            store %143, %1[%c0, %c5] : memref<1x16xvector<8xf32>>
            %144 = addi %4, %c48 : index
            %145 = vector.transfer_read %arg2[%arg4, %144], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %146 = addi %11, %c3 : index
            %147 = cmpi "slt", %146, %c0 : index
            %148 = subi %c-1, %146 : index
            %149 = select %147, %148, %146 : index
            %150 = divi_signed %149, %c16 : index
            %151 = subi %c-1, %150 : index
            %152 = select %147, %151, %150 : index
            %153 = muli %152, %c-16 : index
            %154 = addi %11, %153 : index
            %155 = addi %154, %c3 : index
            %156 = load %2[%155, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %157 = addf %145, %156 : vector<8xf32>
            store %157, %1[%c0, %c6] : memref<1x16xvector<8xf32>>
            %158 = addi %4, %c56 : index
            %159 = vector.transfer_read %arg2[%arg4, %158], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %160 = addi %arg5, %c56 : index
            %161 = cmpi "slt", %160, %c0 : index
            %162 = subi %c-1, %160 : index
            %163 = select %161, %162, %160 : index
            %164 = divi_signed %163, %c16 : index
            %165 = subi %c-1, %164 : index
            %166 = select %161, %165, %164 : index
            %167 = remi_signed %166, %c16 : index
            %168 = cmpi "slt", %167, %c0 : index
            %169 = addi %167, %c16 : index
            %170 = select %168, %169, %167 : index
            %171 = muli %166, %c-2 : index
            %172 = addi %47, %171 : index
            %173 = addi %172, %c7 : index
            %174 = cmpi "slt", %173, %c0 : index
            %175 = subi %c-1, %173 : index
            %176 = select %174, %175, %173 : index
            %177 = divi_signed %176, %c2 : index
            %178 = subi %c-1, %177 : index
            %179 = select %174, %178, %177 : index
            %180 = muli %179, %c-2 : index
            %181 = addi %172, %180 : index
            %182 = addi %181, %c7 : index
            %183 = load %2[%170, %c0, %182] : memref<16x6x2xvector<8xf32>>
            %184 = addf %159, %183 : vector<8xf32>
            store %184, %1[%c0, %c7] : memref<1x16xvector<8xf32>>
            %185 = addi %4, %c64 : index
            %186 = vector.transfer_read %arg2[%arg4, %185], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %187 = addi %11, %c4 : index
            %188 = cmpi "slt", %187, %c0 : index
            %189 = subi %c-1, %187 : index
            %190 = select %188, %189, %187 : index
            %191 = divi_signed %190, %c16 : index
            %192 = subi %c-1, %191 : index
            %193 = select %188, %192, %191 : index
            %194 = muli %193, %c-16 : index
            %195 = addi %11, %194 : index
            %196 = addi %195, %c4 : index
            %197 = load %2[%196, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %198 = addf %186, %197 : vector<8xf32>
            store %198, %1[%c0, %c8] : memref<1x16xvector<8xf32>>
            %199 = addi %4, %c72 : index
            %200 = vector.transfer_read %arg2[%arg4, %199], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %201 = addi %arg5, %c72 : index
            %202 = cmpi "slt", %201, %c0 : index
            %203 = subi %c-1, %201 : index
            %204 = select %202, %203, %201 : index
            %205 = divi_signed %204, %c16 : index
            %206 = subi %c-1, %205 : index
            %207 = select %202, %206, %205 : index
            %208 = remi_signed %207, %c16 : index
            %209 = cmpi "slt", %208, %c0 : index
            %210 = addi %208, %c16 : index
            %211 = select %209, %210, %208 : index
            %212 = muli %207, %c-2 : index
            %213 = addi %47, %212 : index
            %214 = addi %213, %c9 : index
            %215 = cmpi "slt", %214, %c0 : index
            %216 = subi %c-1, %214 : index
            %217 = select %215, %216, %214 : index
            %218 = divi_signed %217, %c2 : index
            %219 = subi %c-1, %218 : index
            %220 = select %215, %219, %218 : index
            %221 = muli %220, %c-2 : index
            %222 = addi %213, %221 : index
            %223 = addi %222, %c9 : index
            %224 = load %2[%211, %c0, %223] : memref<16x6x2xvector<8xf32>>
            %225 = addf %200, %224 : vector<8xf32>
            store %225, %1[%c0, %c9] : memref<1x16xvector<8xf32>>
            %226 = addi %4, %c80 : index
            %227 = vector.transfer_read %arg2[%arg4, %226], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %228 = addi %11, %c5 : index
            %229 = cmpi "slt", %228, %c0 : index
            %230 = subi %c-1, %228 : index
            %231 = select %229, %230, %228 : index
            %232 = divi_signed %231, %c16 : index
            %233 = subi %c-1, %232 : index
            %234 = select %229, %233, %232 : index
            %235 = muli %234, %c-16 : index
            %236 = addi %11, %235 : index
            %237 = addi %236, %c5 : index
            %238 = load %2[%237, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %239 = addf %227, %238 : vector<8xf32>
            store %239, %1[%c0, %c10] : memref<1x16xvector<8xf32>>
            %240 = addi %4, %c88 : index
            %241 = vector.transfer_read %arg2[%arg4, %240], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %242 = addi %arg5, %c88 : index
            %243 = cmpi "slt", %242, %c0 : index
            %244 = subi %c-1, %242 : index
            %245 = select %243, %244, %242 : index
            %246 = divi_signed %245, %c16 : index
            %247 = subi %c-1, %246 : index
            %248 = select %243, %247, %246 : index
            %249 = remi_signed %248, %c16 : index
            %250 = cmpi "slt", %249, %c0 : index
            %251 = addi %249, %c16 : index
            %252 = select %250, %251, %249 : index
            %253 = muli %248, %c-2 : index
            %254 = addi %47, %253 : index
            %255 = addi %254, %c11 : index
            %256 = cmpi "slt", %255, %c0 : index
            %257 = subi %c-1, %255 : index
            %258 = select %256, %257, %255 : index
            %259 = divi_signed %258, %c2 : index
            %260 = subi %c-1, %259 : index
            %261 = select %256, %260, %259 : index
            %262 = muli %261, %c-2 : index
            %263 = addi %254, %262 : index
            %264 = addi %263, %c11 : index
            %265 = load %2[%252, %c0, %264] : memref<16x6x2xvector<8xf32>>
            %266 = addf %241, %265 : vector<8xf32>
            store %266, %1[%c0, %c11] : memref<1x16xvector<8xf32>>
            %267 = addi %4, %c96 : index
            %268 = vector.transfer_read %arg2[%arg4, %267], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %269 = addi %11, %c6 : index
            %270 = cmpi "slt", %269, %c0 : index
            %271 = subi %c-1, %269 : index
            %272 = select %270, %271, %269 : index
            %273 = divi_signed %272, %c16 : index
            %274 = subi %c-1, %273 : index
            %275 = select %270, %274, %273 : index
            %276 = muli %275, %c-16 : index
            %277 = addi %11, %276 : index
            %278 = addi %277, %c6 : index
            %279 = load %2[%278, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %280 = addf %268, %279 : vector<8xf32>
            store %280, %1[%c0, %c12] : memref<1x16xvector<8xf32>>
            %281 = addi %4, %c104 : index
            %282 = vector.transfer_read %arg2[%arg4, %281], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %283 = addi %arg5, %c104 : index
            %284 = cmpi "slt", %283, %c0 : index
            %285 = subi %c-1, %283 : index
            %286 = select %284, %285, %283 : index
            %287 = divi_signed %286, %c16 : index
            %288 = subi %c-1, %287 : index
            %289 = select %284, %288, %287 : index
            %290 = remi_signed %289, %c16 : index
            %291 = cmpi "slt", %290, %c0 : index
            %292 = addi %290, %c16 : index
            %293 = select %291, %292, %290 : index
            %294 = muli %289, %c-2 : index
            %295 = addi %47, %294 : index
            %296 = addi %295, %c13 : index
            %297 = cmpi "slt", %296, %c0 : index
            %298 = subi %c-1, %296 : index
            %299 = select %297, %298, %296 : index
            %300 = divi_signed %299, %c2 : index
            %301 = subi %c-1, %300 : index
            %302 = select %297, %301, %300 : index
            %303 = muli %302, %c-2 : index
            %304 = addi %295, %303 : index
            %305 = addi %304, %c13 : index
            %306 = load %2[%293, %c0, %305] : memref<16x6x2xvector<8xf32>>
            %307 = addf %282, %306 : vector<8xf32>
            store %307, %1[%c0, %c13] : memref<1x16xvector<8xf32>>
            %308 = addi %4, %c112 : index
            %309 = vector.transfer_read %arg2[%arg4, %308], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %310 = addi %11, %c7 : index
            %311 = cmpi "slt", %310, %c0 : index
            %312 = subi %c-1, %310 : index
            %313 = select %311, %312, %310 : index
            %314 = divi_signed %313, %c16 : index
            %315 = subi %c-1, %314 : index
            %316 = select %311, %315, %314 : index
            %317 = muli %316, %c-16 : index
            %318 = addi %11, %317 : index
            %319 = addi %318, %c7 : index
            %320 = load %2[%319, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %321 = addf %309, %320 : vector<8xf32>
            store %321, %1[%c0, %c14] : memref<1x16xvector<8xf32>>
            %322 = addi %4, %c120 : index
            %323 = vector.transfer_read %arg2[%arg4, %322], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %324 = addi %arg5, %c120 : index
            %325 = cmpi "slt", %324, %c0 : index
            %326 = subi %c-1, %324 : index
            %327 = select %325, %326, %324 : index
            %328 = divi_signed %327, %c16 : index
            %329 = subi %c-1, %328 : index
            %330 = select %325, %329, %328 : index
            %331 = remi_signed %330, %c16 : index
            %332 = cmpi "slt", %331, %c0 : index
            %333 = addi %331, %c16 : index
            %334 = select %332, %333, %331 : index
            %335 = muli %330, %c-2 : index
            %336 = addi %47, %335 : index
            %337 = addi %336, %c15 : index
            %338 = cmpi "slt", %337, %c0 : index
            %339 = subi %c-1, %337 : index
            %340 = select %338, %339, %337 : index
            %341 = divi_signed %340, %c2 : index
            %342 = subi %c-1, %341 : index
            %343 = select %338, %342, %341 : index
            %344 = muli %343, %c-2 : index
            %345 = addi %336, %344 : index
            %346 = addi %345, %c15 : index
            %347 = load %2[%334, %c0, %346] : memref<16x6x2xvector<8xf32>>
            %348 = addf %323, %347 : vector<8xf32>
            store %348, %1[%c0, %c15] : memref<1x16xvector<8xf32>>
            scf.for %arg6 = %c0 to %c16 step %c1 {
              %349 = muli %arg6, %c8 : index
              %350 = addi %4, %349 : index
              %351 = load %1[%c0, %arg6] : memref<1x16xvector<8xf32>>
              vector.transfer_write %351, %arg2[%arg4, %350] {masked = [false]} : vector<8xf32>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            }
          } else {
            %4 = addi %arg3, %arg5 : index
            %5 = vector.transfer_read %arg2[%arg4, %4], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %6 = cmpi "slt", %arg5, %c0 : index
            %7 = subi %c-1, %arg5 : index
            %8 = select %6, %7, %arg5 : index
            %9 = divi_signed %8, %c16 : index
            %10 = subi %c-1, %9 : index
            %11 = select %6, %10, %9 : index
            %12 = remi_signed %11, %c16 : index
            %13 = cmpi "slt", %12, %c0 : index
            %14 = addi %12, %c16 : index
            %15 = select %13, %14, %12 : index
            %16 = remi_signed %arg5, %c16 : index
            %17 = cmpi "slt", %16, %c0 : index
            %18 = addi %16, %c16 : index
            %19 = select %17, %18, %16 : index
            %20 = cmpi "slt", %19, %c0 : index
            %21 = subi %c-1, %19 : index
            %22 = select %20, %21, %19 : index
            %23 = divi_signed %22, %c8 : index
            %24 = subi %c-1, %23 : index
            %25 = select %20, %24, %23 : index
            %26 = remi_signed %25, %c2 : index
            %27 = cmpi "slt", %26, %c0 : index
            %28 = addi %26, %c2 : index
            %29 = select %27, %28, %26 : index
            %30 = load %2[%15, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %31 = addf %5, %30 : vector<8xf32>
            store %31, %1[%c0, %c0] : memref<1x16xvector<8xf32>>
            %32 = addi %4, %c8 : index
            %33 = vector.transfer_read %arg2[%arg4, %32], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %34 = addi %arg5, %c8 : index
            %35 = cmpi "slt", %34, %c0 : index
            %36 = subi %c-1, %34 : index
            %37 = select %35, %36, %34 : index
            %38 = divi_signed %37, %c16 : index
            %39 = subi %c-1, %38 : index
            %40 = select %35, %39, %38 : index
            %41 = remi_signed %40, %c16 : index
            %42 = cmpi "slt", %41, %c0 : index
            %43 = addi %41, %c16 : index
            %44 = select %42, %43, %41 : index
            %45 = divi_signed %8, %c8 : index
            %46 = subi %c-1, %45 : index
            %47 = select %6, %46, %45 : index
            %48 = muli %40, %c-2 : index
            %49 = addi %47, %48 : index
            %50 = addi %49, %c1 : index
            %51 = cmpi "slt", %50, %c0 : index
            %52 = subi %c-1, %50 : index
            %53 = select %51, %52, %50 : index
            %54 = divi_signed %53, %c2 : index
            %55 = subi %c-1, %54 : index
            %56 = select %51, %55, %54 : index
            %57 = muli %56, %c-2 : index
            %58 = addi %49, %57 : index
            %59 = addi %58, %c1 : index
            %60 = load %2[%44, %c0, %59] : memref<16x6x2xvector<8xf32>>
            %61 = addf %33, %60 : vector<8xf32>
            store %61, %1[%c0, %c1] : memref<1x16xvector<8xf32>>
            %62 = addi %4, %c16 : index
            %63 = vector.transfer_read %arg2[%arg4, %62], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %64 = addi %11, %c1 : index
            %65 = cmpi "slt", %64, %c0 : index
            %66 = subi %c-1, %64 : index
            %67 = select %65, %66, %64 : index
            %68 = divi_signed %67, %c16 : index
            %69 = subi %c-1, %68 : index
            %70 = select %65, %69, %68 : index
            %71 = muli %70, %c-16 : index
            %72 = addi %11, %71 : index
            %73 = addi %72, %c1 : index
            %74 = load %2[%73, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %75 = addf %63, %74 : vector<8xf32>
            store %75, %1[%c0, %c2] : memref<1x16xvector<8xf32>>
            %76 = addi %4, %c24 : index
            %77 = vector.transfer_read %arg2[%arg4, %76], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %78 = addi %arg5, %c24 : index
            %79 = cmpi "slt", %78, %c0 : index
            %80 = subi %c-1, %78 : index
            %81 = select %79, %80, %78 : index
            %82 = divi_signed %81, %c16 : index
            %83 = subi %c-1, %82 : index
            %84 = select %79, %83, %82 : index
            %85 = remi_signed %84, %c16 : index
            %86 = cmpi "slt", %85, %c0 : index
            %87 = addi %85, %c16 : index
            %88 = select %86, %87, %85 : index
            %89 = muli %84, %c-2 : index
            %90 = addi %47, %89 : index
            %91 = addi %90, %c3 : index
            %92 = cmpi "slt", %91, %c0 : index
            %93 = subi %c-1, %91 : index
            %94 = select %92, %93, %91 : index
            %95 = divi_signed %94, %c2 : index
            %96 = subi %c-1, %95 : index
            %97 = select %92, %96, %95 : index
            %98 = muli %97, %c-2 : index
            %99 = addi %90, %98 : index
            %100 = addi %99, %c3 : index
            %101 = load %2[%88, %c0, %100] : memref<16x6x2xvector<8xf32>>
            %102 = addf %77, %101 : vector<8xf32>
            store %102, %1[%c0, %c3] : memref<1x16xvector<8xf32>>
            %103 = addi %4, %c32 : index
            %104 = vector.transfer_read %arg2[%arg4, %103], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %105 = addi %11, %c2 : index
            %106 = cmpi "slt", %105, %c0 : index
            %107 = subi %c-1, %105 : index
            %108 = select %106, %107, %105 : index
            %109 = divi_signed %108, %c16 : index
            %110 = subi %c-1, %109 : index
            %111 = select %106, %110, %109 : index
            %112 = muli %111, %c-16 : index
            %113 = addi %11, %112 : index
            %114 = addi %113, %c2 : index
            %115 = load %2[%114, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %116 = addf %104, %115 : vector<8xf32>
            store %116, %1[%c0, %c4] : memref<1x16xvector<8xf32>>
            %117 = addi %4, %c40 : index
            %118 = vector.transfer_read %arg2[%arg4, %117], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %119 = addi %arg5, %c40 : index
            %120 = cmpi "slt", %119, %c0 : index
            %121 = subi %c-1, %119 : index
            %122 = select %120, %121, %119 : index
            %123 = divi_signed %122, %c16 : index
            %124 = subi %c-1, %123 : index
            %125 = select %120, %124, %123 : index
            %126 = remi_signed %125, %c16 : index
            %127 = cmpi "slt", %126, %c0 : index
            %128 = addi %126, %c16 : index
            %129 = select %127, %128, %126 : index
            %130 = muli %125, %c-2 : index
            %131 = addi %47, %130 : index
            %132 = addi %131, %c5 : index
            %133 = cmpi "slt", %132, %c0 : index
            %134 = subi %c-1, %132 : index
            %135 = select %133, %134, %132 : index
            %136 = divi_signed %135, %c2 : index
            %137 = subi %c-1, %136 : index
            %138 = select %133, %137, %136 : index
            %139 = muli %138, %c-2 : index
            %140 = addi %131, %139 : index
            %141 = addi %140, %c5 : index
            %142 = load %2[%129, %c0, %141] : memref<16x6x2xvector<8xf32>>
            %143 = addf %118, %142 : vector<8xf32>
            store %143, %1[%c0, %c5] : memref<1x16xvector<8xf32>>
            %144 = addi %4, %c48 : index
            %145 = vector.transfer_read %arg2[%arg4, %144], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %146 = addi %11, %c3 : index
            %147 = cmpi "slt", %146, %c0 : index
            %148 = subi %c-1, %146 : index
            %149 = select %147, %148, %146 : index
            %150 = divi_signed %149, %c16 : index
            %151 = subi %c-1, %150 : index
            %152 = select %147, %151, %150 : index
            %153 = muli %152, %c-16 : index
            %154 = addi %11, %153 : index
            %155 = addi %154, %c3 : index
            %156 = load %2[%155, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %157 = addf %145, %156 : vector<8xf32>
            store %157, %1[%c0, %c6] : memref<1x16xvector<8xf32>>
            %158 = addi %4, %c56 : index
            %159 = vector.transfer_read %arg2[%arg4, %158], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %160 = addi %arg5, %c56 : index
            %161 = cmpi "slt", %160, %c0 : index
            %162 = subi %c-1, %160 : index
            %163 = select %161, %162, %160 : index
            %164 = divi_signed %163, %c16 : index
            %165 = subi %c-1, %164 : index
            %166 = select %161, %165, %164 : index
            %167 = remi_signed %166, %c16 : index
            %168 = cmpi "slt", %167, %c0 : index
            %169 = addi %167, %c16 : index
            %170 = select %168, %169, %167 : index
            %171 = muli %166, %c-2 : index
            %172 = addi %47, %171 : index
            %173 = addi %172, %c7 : index
            %174 = cmpi "slt", %173, %c0 : index
            %175 = subi %c-1, %173 : index
            %176 = select %174, %175, %173 : index
            %177 = divi_signed %176, %c2 : index
            %178 = subi %c-1, %177 : index
            %179 = select %174, %178, %177 : index
            %180 = muli %179, %c-2 : index
            %181 = addi %172, %180 : index
            %182 = addi %181, %c7 : index
            %183 = load %2[%170, %c0, %182] : memref<16x6x2xvector<8xf32>>
            %184 = addf %159, %183 : vector<8xf32>
            store %184, %1[%c0, %c7] : memref<1x16xvector<8xf32>>
            %185 = addi %4, %c64 : index
            %186 = vector.transfer_read %arg2[%arg4, %185], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %187 = addi %11, %c4 : index
            %188 = cmpi "slt", %187, %c0 : index
            %189 = subi %c-1, %187 : index
            %190 = select %188, %189, %187 : index
            %191 = divi_signed %190, %c16 : index
            %192 = subi %c-1, %191 : index
            %193 = select %188, %192, %191 : index
            %194 = muli %193, %c-16 : index
            %195 = addi %11, %194 : index
            %196 = addi %195, %c4 : index
            %197 = load %2[%196, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %198 = addf %186, %197 : vector<8xf32>
            store %198, %1[%c0, %c8] : memref<1x16xvector<8xf32>>
            %199 = addi %4, %c72 : index
            %200 = vector.transfer_read %arg2[%arg4, %199], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %201 = addi %arg5, %c72 : index
            %202 = cmpi "slt", %201, %c0 : index
            %203 = subi %c-1, %201 : index
            %204 = select %202, %203, %201 : index
            %205 = divi_signed %204, %c16 : index
            %206 = subi %c-1, %205 : index
            %207 = select %202, %206, %205 : index
            %208 = remi_signed %207, %c16 : index
            %209 = cmpi "slt", %208, %c0 : index
            %210 = addi %208, %c16 : index
            %211 = select %209, %210, %208 : index
            %212 = muli %207, %c-2 : index
            %213 = addi %47, %212 : index
            %214 = addi %213, %c9 : index
            %215 = cmpi "slt", %214, %c0 : index
            %216 = subi %c-1, %214 : index
            %217 = select %215, %216, %214 : index
            %218 = divi_signed %217, %c2 : index
            %219 = subi %c-1, %218 : index
            %220 = select %215, %219, %218 : index
            %221 = muli %220, %c-2 : index
            %222 = addi %213, %221 : index
            %223 = addi %222, %c9 : index
            %224 = load %2[%211, %c0, %223] : memref<16x6x2xvector<8xf32>>
            %225 = addf %200, %224 : vector<8xf32>
            store %225, %1[%c0, %c9] : memref<1x16xvector<8xf32>>
            %226 = addi %4, %c80 : index
            %227 = vector.transfer_read %arg2[%arg4, %226], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %228 = addi %11, %c5 : index
            %229 = cmpi "slt", %228, %c0 : index
            %230 = subi %c-1, %228 : index
            %231 = select %229, %230, %228 : index
            %232 = divi_signed %231, %c16 : index
            %233 = subi %c-1, %232 : index
            %234 = select %229, %233, %232 : index
            %235 = muli %234, %c-16 : index
            %236 = addi %11, %235 : index
            %237 = addi %236, %c5 : index
            %238 = load %2[%237, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %239 = addf %227, %238 : vector<8xf32>
            store %239, %1[%c0, %c10] : memref<1x16xvector<8xf32>>
            %240 = addi %4, %c88 : index
            %241 = vector.transfer_read %arg2[%arg4, %240], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %242 = addi %arg5, %c88 : index
            %243 = cmpi "slt", %242, %c0 : index
            %244 = subi %c-1, %242 : index
            %245 = select %243, %244, %242 : index
            %246 = divi_signed %245, %c16 : index
            %247 = subi %c-1, %246 : index
            %248 = select %243, %247, %246 : index
            %249 = remi_signed %248, %c16 : index
            %250 = cmpi "slt", %249, %c0 : index
            %251 = addi %249, %c16 : index
            %252 = select %250, %251, %249 : index
            %253 = muli %248, %c-2 : index
            %254 = addi %47, %253 : index
            %255 = addi %254, %c11 : index
            %256 = cmpi "slt", %255, %c0 : index
            %257 = subi %c-1, %255 : index
            %258 = select %256, %257, %255 : index
            %259 = divi_signed %258, %c2 : index
            %260 = subi %c-1, %259 : index
            %261 = select %256, %260, %259 : index
            %262 = muli %261, %c-2 : index
            %263 = addi %254, %262 : index
            %264 = addi %263, %c11 : index
            %265 = load %2[%252, %c0, %264] : memref<16x6x2xvector<8xf32>>
            %266 = addf %241, %265 : vector<8xf32>
            store %266, %1[%c0, %c11] : memref<1x16xvector<8xf32>>
            %267 = addi %4, %c96 : index
            %268 = vector.transfer_read %arg2[%arg4, %267], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %269 = addi %11, %c6 : index
            %270 = cmpi "slt", %269, %c0 : index
            %271 = subi %c-1, %269 : index
            %272 = select %270, %271, %269 : index
            %273 = divi_signed %272, %c16 : index
            %274 = subi %c-1, %273 : index
            %275 = select %270, %274, %273 : index
            %276 = muli %275, %c-16 : index
            %277 = addi %11, %276 : index
            %278 = addi %277, %c6 : index
            %279 = load %2[%278, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %280 = addf %268, %279 : vector<8xf32>
            store %280, %1[%c0, %c12] : memref<1x16xvector<8xf32>>
            %281 = addi %4, %c104 : index
            %282 = vector.transfer_read %arg2[%arg4, %281], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %283 = addi %arg5, %c104 : index
            %284 = cmpi "slt", %283, %c0 : index
            %285 = subi %c-1, %283 : index
            %286 = select %284, %285, %283 : index
            %287 = divi_signed %286, %c16 : index
            %288 = subi %c-1, %287 : index
            %289 = select %284, %288, %287 : index
            %290 = remi_signed %289, %c16 : index
            %291 = cmpi "slt", %290, %c0 : index
            %292 = addi %290, %c16 : index
            %293 = select %291, %292, %290 : index
            %294 = muli %289, %c-2 : index
            %295 = addi %47, %294 : index
            %296 = addi %295, %c13 : index
            %297 = cmpi "slt", %296, %c0 : index
            %298 = subi %c-1, %296 : index
            %299 = select %297, %298, %296 : index
            %300 = divi_signed %299, %c2 : index
            %301 = subi %c-1, %300 : index
            %302 = select %297, %301, %300 : index
            %303 = muli %302, %c-2 : index
            %304 = addi %295, %303 : index
            %305 = addi %304, %c13 : index
            %306 = load %2[%293, %c0, %305] : memref<16x6x2xvector<8xf32>>
            %307 = addf %282, %306 : vector<8xf32>
            store %307, %1[%c0, %c13] : memref<1x16xvector<8xf32>>
            %308 = addi %4, %c112 : index
            %309 = vector.transfer_read %arg2[%arg4, %308], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %310 = addi %11, %c7 : index
            %311 = cmpi "slt", %310, %c0 : index
            %312 = subi %c-1, %310 : index
            %313 = select %311, %312, %310 : index
            %314 = divi_signed %313, %c16 : index
            %315 = subi %c-1, %314 : index
            %316 = select %311, %315, %314 : index
            %317 = muli %316, %c-16 : index
            %318 = addi %11, %317 : index
            %319 = addi %318, %c7 : index
            %320 = load %2[%319, %c0, %29] : memref<16x6x2xvector<8xf32>>
            %321 = addf %309, %320 : vector<8xf32>
            store %321, %1[%c0, %c14] : memref<1x16xvector<8xf32>>
            %322 = addi %4, %c120 : index
            %323 = vector.transfer_read %arg2[%arg4, %322], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
            %324 = addi %arg5, %c120 : index
            %325 = cmpi "slt", %324, %c0 : index
            %326 = subi %c-1, %324 : index
            %327 = select %325, %326, %324 : index
            %328 = divi_signed %327, %c16 : index
            %329 = subi %c-1, %328 : index
            %330 = select %325, %329, %328 : index
            %331 = remi_signed %330, %c16 : index
            %332 = cmpi "slt", %331, %c0 : index
            %333 = addi %331, %c16 : index
            %334 = select %332, %333, %331 : index
            %335 = muli %330, %c-2 : index
            %336 = addi %47, %335 : index
            %337 = addi %336, %c15 : index
            %338 = cmpi "slt", %337, %c0 : index
            %339 = subi %c-1, %337 : index
            %340 = select %338, %339, %337 : index
            %341 = divi_signed %340, %c2 : index
            %342 = subi %c-1, %341 : index
            %343 = select %338, %342, %341 : index
            %344 = muli %343, %c-2 : index
            %345 = addi %336, %344 : index
            %346 = addi %345, %c15 : index
            %347 = load %2[%334, %c0, %346] : memref<16x6x2xvector<8xf32>>
            %348 = addf %323, %347 : vector<8xf32>
            store %348, %1[%c0, %c15] : memref<1x16xvector<8xf32>>
            scf.for %arg6 = %c0 to %c16 step %c1 {
              %349 = muli %arg6, %c8 : index
              %350 = addi %4, %349 : index
              %351 = load %1[%c0, %arg6] : memref<1x16xvector<8xf32>>
              vector.transfer_write %351, %arg2[%arg4, %350] : vector<8xf32>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
            }
          }
        }
      }
    }
    return
  }
  func @optimized_matmul_py_4a6286d9(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "optimized_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
    call @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0, %arg1, %arg2) : (memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) -> ()
    return
  }
}
