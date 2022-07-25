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
    br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb53
    %5 = cmpi "slt", %4, %c512 : index
    cond_br %5, ^bb2, ^bb54
  ^bb2:  // pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%6: index):  // 2 preds: ^bb2, ^bb10
    %7 = cmpi "slt", %6, %c128 : index
    cond_br %7, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    br ^bb5(%c0 : index)
  ^bb5(%8: index):  // 2 preds: ^bb4, ^bb9
    %9 = cmpi "slt", %8, %c256 : index
    cond_br %9, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    cond_br %true, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %10 = addi %4, %8 : index
    %11 = vector.transfer_read %arg1[%6, %10], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %11, %0[%c0, %c0] : memref<1x16xvector<8xf32>>
    %12 = addi %10, %c8 : index
    %13 = vector.transfer_read %arg1[%6, %12], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %13, %0[%c0, %c1] : memref<1x16xvector<8xf32>>
    %14 = addi %10, %c16 : index
    %15 = vector.transfer_read %arg1[%6, %14], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %15, %0[%c0, %c2] : memref<1x16xvector<8xf32>>
    %16 = addi %10, %c24 : index
    %17 = vector.transfer_read %arg1[%6, %16], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %17, %0[%c0, %c3] : memref<1x16xvector<8xf32>>
    %18 = addi %10, %c32 : index
    %19 = vector.transfer_read %arg1[%6, %18], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %19, %0[%c0, %c4] : memref<1x16xvector<8xf32>>
    %20 = addi %10, %c40 : index
    %21 = vector.transfer_read %arg1[%6, %20], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %21, %0[%c0, %c5] : memref<1x16xvector<8xf32>>
    %22 = addi %10, %c48 : index
    %23 = vector.transfer_read %arg1[%6, %22], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %23, %0[%c0, %c6] : memref<1x16xvector<8xf32>>
    %24 = addi %10, %c56 : index
    %25 = vector.transfer_read %arg1[%6, %24], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %25, %0[%c0, %c7] : memref<1x16xvector<8xf32>>
    %26 = addi %10, %c64 : index
    %27 = vector.transfer_read %arg1[%6, %26], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %27, %0[%c0, %c8] : memref<1x16xvector<8xf32>>
    %28 = addi %10, %c72 : index
    %29 = vector.transfer_read %arg1[%6, %28], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %29, %0[%c0, %c9] : memref<1x16xvector<8xf32>>
    %30 = addi %10, %c80 : index
    %31 = vector.transfer_read %arg1[%6, %30], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %31, %0[%c0, %c10] : memref<1x16xvector<8xf32>>
    %32 = addi %10, %c88 : index
    %33 = vector.transfer_read %arg1[%6, %32], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %33, %0[%c0, %c11] : memref<1x16xvector<8xf32>>
    %34 = addi %10, %c96 : index
    %35 = vector.transfer_read %arg1[%6, %34], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %35, %0[%c0, %c12] : memref<1x16xvector<8xf32>>
    %36 = addi %10, %c104 : index
    %37 = vector.transfer_read %arg1[%6, %36], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %37, %0[%c0, %c13] : memref<1x16xvector<8xf32>>
    %38 = addi %10, %c112 : index
    %39 = vector.transfer_read %arg1[%6, %38], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %39, %0[%c0, %c14] : memref<1x16xvector<8xf32>>
    %40 = addi %10, %c120 : index
    %41 = vector.transfer_read %arg1[%6, %40], %cst {masked = [false]} : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %41, %0[%c0, %c15] : memref<1x16xvector<8xf32>>
    %42 = load %0[%c0, %c0] : memref<1x16xvector<8xf32>>
    %43 = cmpi "slt", %8, %c0 : index
    %44 = subi %c-1, %8 : index
    %45 = select %43, %44, %8 : index
    %46 = divi_signed %45, %c16 : index
    %47 = subi %c-1, %46 : index
    %48 = select %43, %47, %46 : index
    %49 = remi_signed %48, %c16 : index
    %50 = cmpi "slt", %49, %c0 : index
    %51 = addi %49, %c16 : index
    %52 = select %50, %51, %49 : index
    %53 = remi_signed %6, %c128 : index
    %54 = cmpi "slt", %53, %c0 : index
    %55 = addi %53, %c128 : index
    %56 = select %54, %55, %53 : index
    %57 = remi_signed %8, %c16 : index
    %58 = cmpi "slt", %57, %c0 : index
    %59 = addi %57, %c16 : index
    %60 = select %58, %59, %57 : index
    %61 = cmpi "slt", %60, %c0 : index
    %62 = subi %c-1, %60 : index
    %63 = select %61, %62, %60 : index
    %64 = divi_signed %63, %c8 : index
    %65 = subi %c-1, %64 : index
    %66 = select %61, %65, %64 : index
    %67 = remi_signed %66, %c2 : index
    %68 = cmpi "slt", %67, %c0 : index
    %69 = addi %67, %c2 : index
    %70 = select %68, %69, %67 : index
    store %42, %3[%52, %56, %70] : memref<16x128x2xvector<8xf32>>
    %71 = load %0[%c0, %c1] : memref<1x16xvector<8xf32>>
    %72 = addi %8, %c8 : index
    %73 = cmpi "slt", %72, %c0 : index
    %74 = subi %c-1, %72 : index
    %75 = select %73, %74, %72 : index
    %76 = divi_signed %75, %c16 : index
    %77 = subi %c-1, %76 : index
    %78 = select %73, %77, %76 : index
    %79 = remi_signed %78, %c16 : index
    %80 = cmpi "slt", %79, %c0 : index
    %81 = addi %79, %c16 : index
    %82 = select %80, %81, %79 : index
    %83 = divi_signed %45, %c8 : index
    %84 = subi %c-1, %83 : index
    %85 = select %43, %84, %83 : index
    %86 = muli %78, %c-2 : index
    %87 = addi %85, %86 : index
    %88 = addi %87, %c1 : index
    %89 = cmpi "slt", %88, %c0 : index
    %90 = subi %c-1, %88 : index
    %91 = select %89, %90, %88 : index
    %92 = divi_signed %91, %c2 : index
    %93 = subi %c-1, %92 : index
    %94 = select %89, %93, %92 : index
    %95 = muli %94, %c-2 : index
    %96 = addi %87, %95 : index
    %97 = addi %96, %c1 : index
    store %71, %3[%82, %56, %97] : memref<16x128x2xvector<8xf32>>
    %98 = load %0[%c0, %c2] : memref<1x16xvector<8xf32>>
    %99 = addi %48, %c1 : index
    %100 = cmpi "slt", %99, %c0 : index
    %101 = subi %c-1, %99 : index
    %102 = select %100, %101, %99 : index
    %103 = divi_signed %102, %c16 : index
    %104 = subi %c-1, %103 : index
    %105 = select %100, %104, %103 : index
    %106 = muli %105, %c-16 : index
    %107 = addi %48, %106 : index
    %108 = addi %107, %c1 : index
    store %98, %3[%108, %56, %70] : memref<16x128x2xvector<8xf32>>
    %109 = load %0[%c0, %c3] : memref<1x16xvector<8xf32>>
    %110 = addi %8, %c24 : index
    %111 = cmpi "slt", %110, %c0 : index
    %112 = subi %c-1, %110 : index
    %113 = select %111, %112, %110 : index
    %114 = divi_signed %113, %c16 : index
    %115 = subi %c-1, %114 : index
    %116 = select %111, %115, %114 : index
    %117 = remi_signed %116, %c16 : index
    %118 = cmpi "slt", %117, %c0 : index
    %119 = addi %117, %c16 : index
    %120 = select %118, %119, %117 : index
    %121 = muli %116, %c-2 : index
    %122 = addi %85, %121 : index
    %123 = addi %122, %c3 : index
    %124 = cmpi "slt", %123, %c0 : index
    %125 = subi %c-1, %123 : index
    %126 = select %124, %125, %123 : index
    %127 = divi_signed %126, %c2 : index
    %128 = subi %c-1, %127 : index
    %129 = select %124, %128, %127 : index
    %130 = muli %129, %c-2 : index
    %131 = addi %122, %130 : index
    %132 = addi %131, %c3 : index
    store %109, %3[%120, %56, %132] : memref<16x128x2xvector<8xf32>>
    %133 = load %0[%c0, %c4] : memref<1x16xvector<8xf32>>
    %134 = addi %48, %c2 : index
    %135 = cmpi "slt", %134, %c0 : index
    %136 = subi %c-1, %134 : index
    %137 = select %135, %136, %134 : index
    %138 = divi_signed %137, %c16 : index
    %139 = subi %c-1, %138 : index
    %140 = select %135, %139, %138 : index
    %141 = muli %140, %c-16 : index
    %142 = addi %48, %141 : index
    %143 = addi %142, %c2 : index
    store %133, %3[%143, %56, %70] : memref<16x128x2xvector<8xf32>>
    %144 = load %0[%c0, %c5] : memref<1x16xvector<8xf32>>
    %145 = addi %8, %c40 : index
    %146 = cmpi "slt", %145, %c0 : index
    %147 = subi %c-1, %145 : index
    %148 = select %146, %147, %145 : index
    %149 = divi_signed %148, %c16 : index
    %150 = subi %c-1, %149 : index
    %151 = select %146, %150, %149 : index
    %152 = remi_signed %151, %c16 : index
    %153 = cmpi "slt", %152, %c0 : index
    %154 = addi %152, %c16 : index
    %155 = select %153, %154, %152 : index
    %156 = muli %151, %c-2 : index
    %157 = addi %85, %156 : index
    %158 = addi %157, %c5 : index
    %159 = cmpi "slt", %158, %c0 : index
    %160 = subi %c-1, %158 : index
    %161 = select %159, %160, %158 : index
    %162 = divi_signed %161, %c2 : index
    %163 = subi %c-1, %162 : index
    %164 = select %159, %163, %162 : index
    %165 = muli %164, %c-2 : index
    %166 = addi %157, %165 : index
    %167 = addi %166, %c5 : index
    store %144, %3[%155, %56, %167] : memref<16x128x2xvector<8xf32>>
    %168 = load %0[%c0, %c6] : memref<1x16xvector<8xf32>>
    %169 = addi %48, %c3 : index
    %170 = cmpi "slt", %169, %c0 : index
    %171 = subi %c-1, %169 : index
    %172 = select %170, %171, %169 : index
    %173 = divi_signed %172, %c16 : index
    %174 = subi %c-1, %173 : index
    %175 = select %170, %174, %173 : index
    %176 = muli %175, %c-16 : index
    %177 = addi %48, %176 : index
    %178 = addi %177, %c3 : index
    store %168, %3[%178, %56, %70] : memref<16x128x2xvector<8xf32>>
    %179 = load %0[%c0, %c7] : memref<1x16xvector<8xf32>>
    %180 = addi %8, %c56 : index
    %181 = cmpi "slt", %180, %c0 : index
    %182 = subi %c-1, %180 : index
    %183 = select %181, %182, %180 : index
    %184 = divi_signed %183, %c16 : index
    %185 = subi %c-1, %184 : index
    %186 = select %181, %185, %184 : index
    %187 = remi_signed %186, %c16 : index
    %188 = cmpi "slt", %187, %c0 : index
    %189 = addi %187, %c16 : index
    %190 = select %188, %189, %187 : index
    %191 = muli %186, %c-2 : index
    %192 = addi %85, %191 : index
    %193 = addi %192, %c7 : index
    %194 = cmpi "slt", %193, %c0 : index
    %195 = subi %c-1, %193 : index
    %196 = select %194, %195, %193 : index
    %197 = divi_signed %196, %c2 : index
    %198 = subi %c-1, %197 : index
    %199 = select %194, %198, %197 : index
    %200 = muli %199, %c-2 : index
    %201 = addi %192, %200 : index
    %202 = addi %201, %c7 : index
    store %179, %3[%190, %56, %202] : memref<16x128x2xvector<8xf32>>
    %203 = load %0[%c0, %c8] : memref<1x16xvector<8xf32>>
    %204 = addi %48, %c4 : index
    %205 = cmpi "slt", %204, %c0 : index
    %206 = subi %c-1, %204 : index
    %207 = select %205, %206, %204 : index
    %208 = divi_signed %207, %c16 : index
    %209 = subi %c-1, %208 : index
    %210 = select %205, %209, %208 : index
    %211 = muli %210, %c-16 : index
    %212 = addi %48, %211 : index
    %213 = addi %212, %c4 : index
    store %203, %3[%213, %56, %70] : memref<16x128x2xvector<8xf32>>
    %214 = load %0[%c0, %c9] : memref<1x16xvector<8xf32>>
    %215 = addi %8, %c72 : index
    %216 = cmpi "slt", %215, %c0 : index
    %217 = subi %c-1, %215 : index
    %218 = select %216, %217, %215 : index
    %219 = divi_signed %218, %c16 : index
    %220 = subi %c-1, %219 : index
    %221 = select %216, %220, %219 : index
    %222 = remi_signed %221, %c16 : index
    %223 = cmpi "slt", %222, %c0 : index
    %224 = addi %222, %c16 : index
    %225 = select %223, %224, %222 : index
    %226 = muli %221, %c-2 : index
    %227 = addi %85, %226 : index
    %228 = addi %227, %c9 : index
    %229 = cmpi "slt", %228, %c0 : index
    %230 = subi %c-1, %228 : index
    %231 = select %229, %230, %228 : index
    %232 = divi_signed %231, %c2 : index
    %233 = subi %c-1, %232 : index
    %234 = select %229, %233, %232 : index
    %235 = muli %234, %c-2 : index
    %236 = addi %227, %235 : index
    %237 = addi %236, %c9 : index
    store %214, %3[%225, %56, %237] : memref<16x128x2xvector<8xf32>>
    %238 = load %0[%c0, %c10] : memref<1x16xvector<8xf32>>
    %239 = addi %48, %c5 : index
    %240 = cmpi "slt", %239, %c0 : index
    %241 = subi %c-1, %239 : index
    %242 = select %240, %241, %239 : index
    %243 = divi_signed %242, %c16 : index
    %244 = subi %c-1, %243 : index
    %245 = select %240, %244, %243 : index
    %246 = muli %245, %c-16 : index
    %247 = addi %48, %246 : index
    %248 = addi %247, %c5 : index
    store %238, %3[%248, %56, %70] : memref<16x128x2xvector<8xf32>>
    %249 = load %0[%c0, %c11] : memref<1x16xvector<8xf32>>
    %250 = addi %8, %c88 : index
    %251 = cmpi "slt", %250, %c0 : index
    %252 = subi %c-1, %250 : index
    %253 = select %251, %252, %250 : index
    %254 = divi_signed %253, %c16 : index
    %255 = subi %c-1, %254 : index
    %256 = select %251, %255, %254 : index
    %257 = remi_signed %256, %c16 : index
    %258 = cmpi "slt", %257, %c0 : index
    %259 = addi %257, %c16 : index
    %260 = select %258, %259, %257 : index
    %261 = muli %256, %c-2 : index
    %262 = addi %85, %261 : index
    %263 = addi %262, %c11 : index
    %264 = cmpi "slt", %263, %c0 : index
    %265 = subi %c-1, %263 : index
    %266 = select %264, %265, %263 : index
    %267 = divi_signed %266, %c2 : index
    %268 = subi %c-1, %267 : index
    %269 = select %264, %268, %267 : index
    %270 = muli %269, %c-2 : index
    %271 = addi %262, %270 : index
    %272 = addi %271, %c11 : index
    store %249, %3[%260, %56, %272] : memref<16x128x2xvector<8xf32>>
    %273 = load %0[%c0, %c12] : memref<1x16xvector<8xf32>>
    %274 = addi %48, %c6 : index
    %275 = cmpi "slt", %274, %c0 : index
    %276 = subi %c-1, %274 : index
    %277 = select %275, %276, %274 : index
    %278 = divi_signed %277, %c16 : index
    %279 = subi %c-1, %278 : index
    %280 = select %275, %279, %278 : index
    %281 = muli %280, %c-16 : index
    %282 = addi %48, %281 : index
    %283 = addi %282, %c6 : index
    store %273, %3[%283, %56, %70] : memref<16x128x2xvector<8xf32>>
    %284 = load %0[%c0, %c13] : memref<1x16xvector<8xf32>>
    %285 = addi %8, %c104 : index
    %286 = cmpi "slt", %285, %c0 : index
    %287 = subi %c-1, %285 : index
    %288 = select %286, %287, %285 : index
    %289 = divi_signed %288, %c16 : index
    %290 = subi %c-1, %289 : index
    %291 = select %286, %290, %289 : index
    %292 = remi_signed %291, %c16 : index
    %293 = cmpi "slt", %292, %c0 : index
    %294 = addi %292, %c16 : index
    %295 = select %293, %294, %292 : index
    %296 = muli %291, %c-2 : index
    %297 = addi %85, %296 : index
    %298 = addi %297, %c13 : index
    %299 = cmpi "slt", %298, %c0 : index
    %300 = subi %c-1, %298 : index
    %301 = select %299, %300, %298 : index
    %302 = divi_signed %301, %c2 : index
    %303 = subi %c-1, %302 : index
    %304 = select %299, %303, %302 : index
    %305 = muli %304, %c-2 : index
    %306 = addi %297, %305 : index
    %307 = addi %306, %c13 : index
    store %284, %3[%295, %56, %307] : memref<16x128x2xvector<8xf32>>
    %308 = load %0[%c0, %c14] : memref<1x16xvector<8xf32>>
    %309 = addi %48, %c7 : index
    %310 = cmpi "slt", %309, %c0 : index
    %311 = subi %c-1, %309 : index
    %312 = select %310, %311, %309 : index
    %313 = divi_signed %312, %c16 : index
    %314 = subi %c-1, %313 : index
    %315 = select %310, %314, %313 : index
    %316 = muli %315, %c-16 : index
    %317 = addi %48, %316 : index
    %318 = addi %317, %c7 : index
    store %308, %3[%318, %56, %70] : memref<16x128x2xvector<8xf32>>
    %319 = load %0[%c0, %c15] : memref<1x16xvector<8xf32>>
    %320 = addi %8, %c120 : index
    %321 = cmpi "slt", %320, %c0 : index
    %322 = subi %c-1, %320 : index
    %323 = select %321, %322, %320 : index
    %324 = divi_signed %323, %c16 : index
    %325 = subi %c-1, %324 : index
    %326 = select %321, %325, %324 : index
    %327 = remi_signed %326, %c16 : index
    %328 = cmpi "slt", %327, %c0 : index
    %329 = addi %327, %c16 : index
    %330 = select %328, %329, %327 : index
    %331 = muli %326, %c-2 : index
    %332 = addi %85, %331 : index
    %333 = addi %332, %c15 : index
    %334 = cmpi "slt", %333, %c0 : index
    %335 = subi %c-1, %333 : index
    %336 = select %334, %335, %333 : index
    %337 = divi_signed %336, %c2 : index
    %338 = subi %c-1, %337 : index
    %339 = select %334, %338, %337 : index
    %340 = muli %339, %c-2 : index
    %341 = addi %332, %340 : index
    %342 = addi %341, %c15 : index
    store %319, %3[%330, %56, %342] : memref<16x128x2xvector<8xf32>>
    br ^bb9
  ^bb8:  // pred: ^bb6
    %343 = addi %4, %8 : index
    %344 = vector.transfer_read %arg1[%6, %343], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %344, %0[%c0, %c0] : memref<1x16xvector<8xf32>>
    %345 = addi %343, %c8 : index
    %346 = vector.transfer_read %arg1[%6, %345], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %346, %0[%c0, %c1] : memref<1x16xvector<8xf32>>
    %347 = addi %343, %c16 : index
    %348 = vector.transfer_read %arg1[%6, %347], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %348, %0[%c0, %c2] : memref<1x16xvector<8xf32>>
    %349 = addi %343, %c24 : index
    %350 = vector.transfer_read %arg1[%6, %349], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %350, %0[%c0, %c3] : memref<1x16xvector<8xf32>>
    %351 = addi %343, %c32 : index
    %352 = vector.transfer_read %arg1[%6, %351], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %352, %0[%c0, %c4] : memref<1x16xvector<8xf32>>
    %353 = addi %343, %c40 : index
    %354 = vector.transfer_read %arg1[%6, %353], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %354, %0[%c0, %c5] : memref<1x16xvector<8xf32>>
    %355 = addi %343, %c48 : index
    %356 = vector.transfer_read %arg1[%6, %355], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %356, %0[%c0, %c6] : memref<1x16xvector<8xf32>>
    %357 = addi %343, %c56 : index
    %358 = vector.transfer_read %arg1[%6, %357], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %358, %0[%c0, %c7] : memref<1x16xvector<8xf32>>
    %359 = addi %343, %c64 : index
    %360 = vector.transfer_read %arg1[%6, %359], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %360, %0[%c0, %c8] : memref<1x16xvector<8xf32>>
    %361 = addi %343, %c72 : index
    %362 = vector.transfer_read %arg1[%6, %361], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %362, %0[%c0, %c9] : memref<1x16xvector<8xf32>>
    %363 = addi %343, %c80 : index
    %364 = vector.transfer_read %arg1[%6, %363], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %364, %0[%c0, %c10] : memref<1x16xvector<8xf32>>
    %365 = addi %343, %c88 : index
    %366 = vector.transfer_read %arg1[%6, %365], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %366, %0[%c0, %c11] : memref<1x16xvector<8xf32>>
    %367 = addi %343, %c96 : index
    %368 = vector.transfer_read %arg1[%6, %367], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %368, %0[%c0, %c12] : memref<1x16xvector<8xf32>>
    %369 = addi %343, %c104 : index
    %370 = vector.transfer_read %arg1[%6, %369], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %370, %0[%c0, %c13] : memref<1x16xvector<8xf32>>
    %371 = addi %343, %c112 : index
    %372 = vector.transfer_read %arg1[%6, %371], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %372, %0[%c0, %c14] : memref<1x16xvector<8xf32>>
    %373 = addi %343, %c120 : index
    %374 = vector.transfer_read %arg1[%6, %373], %cst : memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    store %374, %0[%c0, %c15] : memref<1x16xvector<8xf32>>
    %375 = load %0[%c0, %c0] : memref<1x16xvector<8xf32>>
    %376 = cmpi "slt", %8, %c0 : index
    %377 = subi %c-1, %8 : index
    %378 = select %376, %377, %8 : index
    %379 = divi_signed %378, %c16 : index
    %380 = subi %c-1, %379 : index
    %381 = select %376, %380, %379 : index
    %382 = remi_signed %381, %c16 : index
    %383 = cmpi "slt", %382, %c0 : index
    %384 = addi %382, %c16 : index
    %385 = select %383, %384, %382 : index
    %386 = remi_signed %6, %c128 : index
    %387 = cmpi "slt", %386, %c0 : index
    %388 = addi %386, %c128 : index
    %389 = select %387, %388, %386 : index
    %390 = remi_signed %8, %c16 : index
    %391 = cmpi "slt", %390, %c0 : index
    %392 = addi %390, %c16 : index
    %393 = select %391, %392, %390 : index
    %394 = cmpi "slt", %393, %c0 : index
    %395 = subi %c-1, %393 : index
    %396 = select %394, %395, %393 : index
    %397 = divi_signed %396, %c8 : index
    %398 = subi %c-1, %397 : index
    %399 = select %394, %398, %397 : index
    %400 = remi_signed %399, %c2 : index
    %401 = cmpi "slt", %400, %c0 : index
    %402 = addi %400, %c2 : index
    %403 = select %401, %402, %400 : index
    store %375, %3[%385, %389, %403] : memref<16x128x2xvector<8xf32>>
    %404 = load %0[%c0, %c1] : memref<1x16xvector<8xf32>>
    %405 = addi %8, %c8 : index
    %406 = cmpi "slt", %405, %c0 : index
    %407 = subi %c-1, %405 : index
    %408 = select %406, %407, %405 : index
    %409 = divi_signed %408, %c16 : index
    %410 = subi %c-1, %409 : index
    %411 = select %406, %410, %409 : index
    %412 = remi_signed %411, %c16 : index
    %413 = cmpi "slt", %412, %c0 : index
    %414 = addi %412, %c16 : index
    %415 = select %413, %414, %412 : index
    %416 = divi_signed %378, %c8 : index
    %417 = subi %c-1, %416 : index
    %418 = select %376, %417, %416 : index
    %419 = muli %411, %c-2 : index
    %420 = addi %418, %419 : index
    %421 = addi %420, %c1 : index
    %422 = cmpi "slt", %421, %c0 : index
    %423 = subi %c-1, %421 : index
    %424 = select %422, %423, %421 : index
    %425 = divi_signed %424, %c2 : index
    %426 = subi %c-1, %425 : index
    %427 = select %422, %426, %425 : index
    %428 = muli %427, %c-2 : index
    %429 = addi %420, %428 : index
    %430 = addi %429, %c1 : index
    store %404, %3[%415, %389, %430] : memref<16x128x2xvector<8xf32>>
    %431 = load %0[%c0, %c2] : memref<1x16xvector<8xf32>>
    %432 = addi %381, %c1 : index
    %433 = cmpi "slt", %432, %c0 : index
    %434 = subi %c-1, %432 : index
    %435 = select %433, %434, %432 : index
    %436 = divi_signed %435, %c16 : index
    %437 = subi %c-1, %436 : index
    %438 = select %433, %437, %436 : index
    %439 = muli %438, %c-16 : index
    %440 = addi %381, %439 : index
    %441 = addi %440, %c1 : index
    store %431, %3[%441, %389, %403] : memref<16x128x2xvector<8xf32>>
    %442 = load %0[%c0, %c3] : memref<1x16xvector<8xf32>>
    %443 = addi %8, %c24 : index
    %444 = cmpi "slt", %443, %c0 : index
    %445 = subi %c-1, %443 : index
    %446 = select %444, %445, %443 : index
    %447 = divi_signed %446, %c16 : index
    %448 = subi %c-1, %447 : index
    %449 = select %444, %448, %447 : index
    %450 = remi_signed %449, %c16 : index
    %451 = cmpi "slt", %450, %c0 : index
    %452 = addi %450, %c16 : index
    %453 = select %451, %452, %450 : index
    %454 = muli %449, %c-2 : index
    %455 = addi %418, %454 : index
    %456 = addi %455, %c3 : index
    %457 = cmpi "slt", %456, %c0 : index
    %458 = subi %c-1, %456 : index
    %459 = select %457, %458, %456 : index
    %460 = divi_signed %459, %c2 : index
    %461 = subi %c-1, %460 : index
    %462 = select %457, %461, %460 : index
    %463 = muli %462, %c-2 : index
    %464 = addi %455, %463 : index
    %465 = addi %464, %c3 : index
    store %442, %3[%453, %389, %465] : memref<16x128x2xvector<8xf32>>
    %466 = load %0[%c0, %c4] : memref<1x16xvector<8xf32>>
    %467 = addi %381, %c2 : index
    %468 = cmpi "slt", %467, %c0 : index
    %469 = subi %c-1, %467 : index
    %470 = select %468, %469, %467 : index
    %471 = divi_signed %470, %c16 : index
    %472 = subi %c-1, %471 : index
    %473 = select %468, %472, %471 : index
    %474 = muli %473, %c-16 : index
    %475 = addi %381, %474 : index
    %476 = addi %475, %c2 : index
    store %466, %3[%476, %389, %403] : memref<16x128x2xvector<8xf32>>
    %477 = load %0[%c0, %c5] : memref<1x16xvector<8xf32>>
    %478 = addi %8, %c40 : index
    %479 = cmpi "slt", %478, %c0 : index
    %480 = subi %c-1, %478 : index
    %481 = select %479, %480, %478 : index
    %482 = divi_signed %481, %c16 : index
    %483 = subi %c-1, %482 : index
    %484 = select %479, %483, %482 : index
    %485 = remi_signed %484, %c16 : index
    %486 = cmpi "slt", %485, %c0 : index
    %487 = addi %485, %c16 : index
    %488 = select %486, %487, %485 : index
    %489 = muli %484, %c-2 : index
    %490 = addi %418, %489 : index
    %491 = addi %490, %c5 : index
    %492 = cmpi "slt", %491, %c0 : index
    %493 = subi %c-1, %491 : index
    %494 = select %492, %493, %491 : index
    %495 = divi_signed %494, %c2 : index
    %496 = subi %c-1, %495 : index
    %497 = select %492, %496, %495 : index
    %498 = muli %497, %c-2 : index
    %499 = addi %490, %498 : index
    %500 = addi %499, %c5 : index
    store %477, %3[%488, %389, %500] : memref<16x128x2xvector<8xf32>>
    %501 = load %0[%c0, %c6] : memref<1x16xvector<8xf32>>
    %502 = addi %381, %c3 : index
    %503 = cmpi "slt", %502, %c0 : index
    %504 = subi %c-1, %502 : index
    %505 = select %503, %504, %502 : index
    %506 = divi_signed %505, %c16 : index
    %507 = subi %c-1, %506 : index
    %508 = select %503, %507, %506 : index
    %509 = muli %508, %c-16 : index
    %510 = addi %381, %509 : index
    %511 = addi %510, %c3 : index
    store %501, %3[%511, %389, %403] : memref<16x128x2xvector<8xf32>>
    %512 = load %0[%c0, %c7] : memref<1x16xvector<8xf32>>
    %513 = addi %8, %c56 : index
    %514 = cmpi "slt", %513, %c0 : index
    %515 = subi %c-1, %513 : index
    %516 = select %514, %515, %513 : index
    %517 = divi_signed %516, %c16 : index
    %518 = subi %c-1, %517 : index
    %519 = select %514, %518, %517 : index
    %520 = remi_signed %519, %c16 : index
    %521 = cmpi "slt", %520, %c0 : index
    %522 = addi %520, %c16 : index
    %523 = select %521, %522, %520 : index
    %524 = muli %519, %c-2 : index
    %525 = addi %418, %524 : index
    %526 = addi %525, %c7 : index
    %527 = cmpi "slt", %526, %c0 : index
    %528 = subi %c-1, %526 : index
    %529 = select %527, %528, %526 : index
    %530 = divi_signed %529, %c2 : index
    %531 = subi %c-1, %530 : index
    %532 = select %527, %531, %530 : index
    %533 = muli %532, %c-2 : index
    %534 = addi %525, %533 : index
    %535 = addi %534, %c7 : index
    store %512, %3[%523, %389, %535] : memref<16x128x2xvector<8xf32>>
    %536 = load %0[%c0, %c8] : memref<1x16xvector<8xf32>>
    %537 = addi %381, %c4 : index
    %538 = cmpi "slt", %537, %c0 : index
    %539 = subi %c-1, %537 : index
    %540 = select %538, %539, %537 : index
    %541 = divi_signed %540, %c16 : index
    %542 = subi %c-1, %541 : index
    %543 = select %538, %542, %541 : index
    %544 = muli %543, %c-16 : index
    %545 = addi %381, %544 : index
    %546 = addi %545, %c4 : index
    store %536, %3[%546, %389, %403] : memref<16x128x2xvector<8xf32>>
    %547 = load %0[%c0, %c9] : memref<1x16xvector<8xf32>>
    %548 = addi %8, %c72 : index
    %549 = cmpi "slt", %548, %c0 : index
    %550 = subi %c-1, %548 : index
    %551 = select %549, %550, %548 : index
    %552 = divi_signed %551, %c16 : index
    %553 = subi %c-1, %552 : index
    %554 = select %549, %553, %552 : index
    %555 = remi_signed %554, %c16 : index
    %556 = cmpi "slt", %555, %c0 : index
    %557 = addi %555, %c16 : index
    %558 = select %556, %557, %555 : index
    %559 = muli %554, %c-2 : index
    %560 = addi %418, %559 : index
    %561 = addi %560, %c9 : index
    %562 = cmpi "slt", %561, %c0 : index
    %563 = subi %c-1, %561 : index
    %564 = select %562, %563, %561 : index
    %565 = divi_signed %564, %c2 : index
    %566 = subi %c-1, %565 : index
    %567 = select %562, %566, %565 : index
    %568 = muli %567, %c-2 : index
    %569 = addi %560, %568 : index
    %570 = addi %569, %c9 : index
    store %547, %3[%558, %389, %570] : memref<16x128x2xvector<8xf32>>
    %571 = load %0[%c0, %c10] : memref<1x16xvector<8xf32>>
    %572 = addi %381, %c5 : index
    %573 = cmpi "slt", %572, %c0 : index
    %574 = subi %c-1, %572 : index
    %575 = select %573, %574, %572 : index
    %576 = divi_signed %575, %c16 : index
    %577 = subi %c-1, %576 : index
    %578 = select %573, %577, %576 : index
    %579 = muli %578, %c-16 : index
    %580 = addi %381, %579 : index
    %581 = addi %580, %c5 : index
    store %571, %3[%581, %389, %403] : memref<16x128x2xvector<8xf32>>
    %582 = load %0[%c0, %c11] : memref<1x16xvector<8xf32>>
    %583 = addi %8, %c88 : index
    %584 = cmpi "slt", %583, %c0 : index
    %585 = subi %c-1, %583 : index
    %586 = select %584, %585, %583 : index
    %587 = divi_signed %586, %c16 : index
    %588 = subi %c-1, %587 : index
    %589 = select %584, %588, %587 : index
    %590 = remi_signed %589, %c16 : index
    %591 = cmpi "slt", %590, %c0 : index
    %592 = addi %590, %c16 : index
    %593 = select %591, %592, %590 : index
    %594 = muli %589, %c-2 : index
    %595 = addi %418, %594 : index
    %596 = addi %595, %c11 : index
    %597 = cmpi "slt", %596, %c0 : index
    %598 = subi %c-1, %596 : index
    %599 = select %597, %598, %596 : index
    %600 = divi_signed %599, %c2 : index
    %601 = subi %c-1, %600 : index
    %602 = select %597, %601, %600 : index
    %603 = muli %602, %c-2 : index
    %604 = addi %595, %603 : index
    %605 = addi %604, %c11 : index
    store %582, %3[%593, %389, %605] : memref<16x128x2xvector<8xf32>>
    %606 = load %0[%c0, %c12] : memref<1x16xvector<8xf32>>
    %607 = addi %381, %c6 : index
    %608 = cmpi "slt", %607, %c0 : index
    %609 = subi %c-1, %607 : index
    %610 = select %608, %609, %607 : index
    %611 = divi_signed %610, %c16 : index
    %612 = subi %c-1, %611 : index
    %613 = select %608, %612, %611 : index
    %614 = muli %613, %c-16 : index
    %615 = addi %381, %614 : index
    %616 = addi %615, %c6 : index
    store %606, %3[%616, %389, %403] : memref<16x128x2xvector<8xf32>>
    %617 = load %0[%c0, %c13] : memref<1x16xvector<8xf32>>
    %618 = addi %8, %c104 : index
    %619 = cmpi "slt", %618, %c0 : index
    %620 = subi %c-1, %618 : index
    %621 = select %619, %620, %618 : index
    %622 = divi_signed %621, %c16 : index
    %623 = subi %c-1, %622 : index
    %624 = select %619, %623, %622 : index
    %625 = remi_signed %624, %c16 : index
    %626 = cmpi "slt", %625, %c0 : index
    %627 = addi %625, %c16 : index
    %628 = select %626, %627, %625 : index
    %629 = muli %624, %c-2 : index
    %630 = addi %418, %629 : index
    %631 = addi %630, %c13 : index
    %632 = cmpi "slt", %631, %c0 : index
    %633 = subi %c-1, %631 : index
    %634 = select %632, %633, %631 : index
    %635 = divi_signed %634, %c2 : index
    %636 = subi %c-1, %635 : index
    %637 = select %632, %636, %635 : index
    %638 = muli %637, %c-2 : index
    %639 = addi %630, %638 : index
    %640 = addi %639, %c13 : index
    store %617, %3[%628, %389, %640] : memref<16x128x2xvector<8xf32>>
    %641 = load %0[%c0, %c14] : memref<1x16xvector<8xf32>>
    %642 = addi %381, %c7 : index
    %643 = cmpi "slt", %642, %c0 : index
    %644 = subi %c-1, %642 : index
    %645 = select %643, %644, %642 : index
    %646 = divi_signed %645, %c16 : index
    %647 = subi %c-1, %646 : index
    %648 = select %643, %647, %646 : index
    %649 = muli %648, %c-16 : index
    %650 = addi %381, %649 : index
    %651 = addi %650, %c7 : index
    store %641, %3[%651, %389, %403] : memref<16x128x2xvector<8xf32>>
    %652 = load %0[%c0, %c15] : memref<1x16xvector<8xf32>>
    %653 = addi %8, %c120 : index
    %654 = cmpi "slt", %653, %c0 : index
    %655 = subi %c-1, %653 : index
    %656 = select %654, %655, %653 : index
    %657 = divi_signed %656, %c16 : index
    %658 = subi %c-1, %657 : index
    %659 = select %654, %658, %657 : index
    %660 = remi_signed %659, %c16 : index
    %661 = cmpi "slt", %660, %c0 : index
    %662 = addi %660, %c16 : index
    %663 = select %661, %662, %660 : index
    %664 = muli %659, %c-2 : index
    %665 = addi %418, %664 : index
    %666 = addi %665, %c15 : index
    %667 = cmpi "slt", %666, %c0 : index
    %668 = subi %c-1, %666 : index
    %669 = select %667, %668, %666 : index
    %670 = divi_signed %669, %c2 : index
    %671 = subi %c-1, %670 : index
    %672 = select %667, %671, %670 : index
    %673 = muli %672, %c-2 : index
    %674 = addi %665, %673 : index
    %675 = addi %674, %c15 : index
    store %652, %3[%663, %389, %675] : memref<16x128x2xvector<8xf32>>
    br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    %676 = addi %8, %c128 : index
    br ^bb5(%676 : index)
  ^bb10:  // pred: ^bb5
    %677 = addi %6, %c1 : index
    br ^bb3(%677 : index)
  ^bb11:  // pred: ^bb3
    br ^bb12(%c0 : index)
  ^bb12(%678: index):  // 2 preds: ^bb11, ^bb52
    %679 = cmpi "slt", %678, %c784 : index
    cond_br %679, ^bb13, ^bb53
  ^bb13:  // pred: ^bb12
    br ^bb14(%c0 : index)
  ^bb14(%680: index):  // 2 preds: ^bb13, ^bb21
    %681 = cmpi "slt", %680, %c16 : index
    cond_br %681, ^bb15, ^bb22
  ^bb15:  // pred: ^bb14
    br ^bb16(%c0 : index)
  ^bb16(%682: index):  // 2 preds: ^bb15, ^bb20
    %683 = cmpi "slt", %682, %c6 : index
    cond_br %683, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    br ^bb18(%c0 : index)
  ^bb18(%684: index):  // 2 preds: ^bb17, ^bb19
    %685 = cmpi "slt", %684, %c2 : index
    cond_br %685, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    store %cst_0, %2[%680, %682, %684] : memref<16x6x2xvector<8xf32>>
    %686 = addi %684, %c1 : index
    br ^bb18(%686 : index)
  ^bb20:  // pred: ^bb18
    %687 = addi %682, %c1 : index
    br ^bb16(%687 : index)
  ^bb21:  // pred: ^bb16
    %688 = addi %680, %c1 : index
    br ^bb14(%688 : index)
  ^bb22:  // pred: ^bb14
    br ^bb23(%c0 : index)
  ^bb23(%689: index):  // 2 preds: ^bb22, ^bb39
    %690 = cmpi "slt", %689, %c256 : index
    cond_br %690, ^bb24, ^bb40
  ^bb24:  // pred: ^bb23
    br ^bb25(%c0 : index)
  ^bb25(%691: index):  // 2 preds: ^bb24, ^bb38
    %692 = cmpi "slt", %691, %c128 : index
    cond_br %692, ^bb26, ^bb39
  ^bb26:  // pred: ^bb25
    br ^bb27(%c0 : index)
  ^bb27(%693: index):  // 2 preds: ^bb26, ^bb34
    %694 = cmpi "slt", %693, %c0 : index
    cond_br %694, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    br ^bb29(%c0 : index)
  ^bb29(%695: index):  // 2 preds: ^bb28, ^bb33
    %696 = cmpi "slt", %695, %c4 : index
    cond_br %696, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    br ^bb31(%c0 : index)
  ^bb31(%697: index):  // 2 preds: ^bb30, ^bb32
    %698 = cmpi "slt", %697, %c0 : index
    cond_br %698, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %699 = addi %678, %693 : index
    %700 = addi %699, %697 : index
    %701 = addi %691, %695 : index
    %702 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %703 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %704 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %705 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %706 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %707 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %708 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %709 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %710 = cmpi "slt", %689, %c0 : index
    %711 = subi %c-1, %689 : index
    %712 = select %710, %711, %689 : index
    %713 = divi_signed %712, %c16 : index
    %714 = subi %c-1, %713 : index
    %715 = select %710, %714, %713 : index
    %716 = remi_signed %715, %c16 : index
    %717 = cmpi "slt", %716, %c0 : index
    %718 = addi %716, %c16 : index
    %719 = select %717, %718, %716 : index
    %720 = remi_signed %701, %c128 : index
    %721 = cmpi "slt", %720, %c0 : index
    %722 = addi %720, %c128 : index
    %723 = select %721, %722, %720 : index
    %724 = remi_signed %689, %c16 : index
    %725 = cmpi "slt", %724, %c0 : index
    %726 = addi %724, %c16 : index
    %727 = select %725, %726, %724 : index
    %728 = cmpi "slt", %727, %c0 : index
    %729 = subi %c-1, %727 : index
    %730 = select %728, %729, %727 : index
    %731 = divi_signed %730, %c8 : index
    %732 = subi %c-1, %731 : index
    %733 = select %728, %732, %731 : index
    %734 = remi_signed %733, %c2 : index
    %735 = cmpi "slt", %734, %c0 : index
    %736 = addi %734, %c2 : index
    %737 = select %735, %736, %734 : index
    %738 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %739 = vector.extractelement %738[%c0_i64 : i64] : vector<8xf32>
    %740 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %741 = vector.extractelement %740[%c1_i64 : i64] : vector<8xf32>
    %742 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %743 = vector.extractelement %742[%c2_i64 : i64] : vector<8xf32>
    %744 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %745 = vector.extractelement %744[%c3_i64 : i64] : vector<8xf32>
    %746 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %747 = vector.extractelement %746[%c4_i64 : i64] : vector<8xf32>
    %748 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %749 = vector.extractelement %748[%c5_i64 : i64] : vector<8xf32>
    %750 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %751 = vector.extractelement %750[%c6_i64 : i64] : vector<8xf32>
    %752 = load %3[%719, %723, %737] : memref<16x128x2xvector<8xf32>>
    %753 = vector.extractelement %752[%c7_i64 : i64] : vector<8xf32>
    %754 = mulf %702, %739 {RelaxedPrecision} : f32
    %755 = mulf %703, %741 {RelaxedPrecision} : f32
    %756 = mulf %704, %743 {RelaxedPrecision} : f32
    %757 = mulf %705, %745 {RelaxedPrecision} : f32
    %758 = mulf %706, %747 {RelaxedPrecision} : f32
    %759 = mulf %707, %749 {RelaxedPrecision} : f32
    %760 = mulf %708, %751 {RelaxedPrecision} : f32
    %761 = mulf %709, %753 {RelaxedPrecision} : f32
    %762 = addi %693, %697 : index
    %763 = remi_signed %762, %c6 : index
    %764 = cmpi "slt", %763, %c0 : index
    %765 = addi %763, %c6 : index
    %766 = select %764, %765, %763 : index
    %767 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %768 = vector.extractelement %767[%c0_i64 : i64] : vector<8xf32>
    %769 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %770 = vector.extractelement %769[%c1_i64 : i64] : vector<8xf32>
    %771 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %772 = vector.extractelement %771[%c2_i64 : i64] : vector<8xf32>
    %773 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %774 = vector.extractelement %773[%c3_i64 : i64] : vector<8xf32>
    %775 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %776 = vector.extractelement %775[%c4_i64 : i64] : vector<8xf32>
    %777 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %778 = vector.extractelement %777[%c5_i64 : i64] : vector<8xf32>
    %779 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %780 = vector.extractelement %779[%c6_i64 : i64] : vector<8xf32>
    %781 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %782 = vector.extractelement %781[%c7_i64 : i64] : vector<8xf32>
    %783 = addf %768, %754 {RelaxedPrecision} : f32
    %784 = addf %770, %755 {RelaxedPrecision} : f32
    %785 = addf %772, %756 {RelaxedPrecision} : f32
    %786 = addf %774, %757 {RelaxedPrecision} : f32
    %787 = addf %776, %758 {RelaxedPrecision} : f32
    %788 = addf %778, %759 {RelaxedPrecision} : f32
    %789 = addf %780, %760 {RelaxedPrecision} : f32
    %790 = addf %782, %761 {RelaxedPrecision} : f32
    %791 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %792 = vector.insertelement %783, %791[%c0_i64 : i64] : vector<8xf32>
    store %792, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %793 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %794 = vector.insertelement %784, %793[%c1_i64 : i64] : vector<8xf32>
    store %794, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %795 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %796 = vector.insertelement %785, %795[%c2_i64 : i64] : vector<8xf32>
    store %796, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %797 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %798 = vector.insertelement %786, %797[%c3_i64 : i64] : vector<8xf32>
    store %798, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %799 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %800 = vector.insertelement %787, %799[%c4_i64 : i64] : vector<8xf32>
    store %800, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %801 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %802 = vector.insertelement %788, %801[%c5_i64 : i64] : vector<8xf32>
    store %802, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %803 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %804 = vector.insertelement %789, %803[%c6_i64 : i64] : vector<8xf32>
    store %804, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %805 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %806 = vector.insertelement %790, %805[%c7_i64 : i64] : vector<8xf32>
    store %806, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %807 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %808 = vector.insertelement %783, %807[%c0_i64 : i64] : vector<8xf32>
    store %808, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %809 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %810 = vector.insertelement %784, %809[%c1_i64 : i64] : vector<8xf32>
    store %810, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %811 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %812 = vector.insertelement %785, %811[%c2_i64 : i64] : vector<8xf32>
    store %812, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %813 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %814 = vector.insertelement %786, %813[%c3_i64 : i64] : vector<8xf32>
    store %814, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %815 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %816 = vector.insertelement %787, %815[%c4_i64 : i64] : vector<8xf32>
    store %816, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %817 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %818 = vector.insertelement %788, %817[%c5_i64 : i64] : vector<8xf32>
    store %818, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %819 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %820 = vector.insertelement %789, %819[%c6_i64 : i64] : vector<8xf32>
    store %820, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %821 = load %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %822 = vector.insertelement %790, %821[%c7_i64 : i64] : vector<8xf32>
    store %822, %2[%719, %766, %737] : memref<16x6x2xvector<8xf32>>
    %823 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %824 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %825 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %826 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %827 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %828 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %829 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %830 = load %arg0[%700, %701] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %831 = addi %689, %c8 : index
    %832 = cmpi "slt", %831, %c0 : index
    %833 = subi %c-1, %831 : index
    %834 = select %832, %833, %831 : index
    %835 = divi_signed %834, %c16 : index
    %836 = subi %c-1, %835 : index
    %837 = select %832, %836, %835 : index
    %838 = remi_signed %837, %c16 : index
    %839 = cmpi "slt", %838, %c0 : index
    %840 = addi %838, %c16 : index
    %841 = select %839, %840, %838 : index
    %842 = divi_signed %712, %c8 : index
    %843 = subi %c-1, %842 : index
    %844 = select %710, %843, %842 : index
    %845 = muli %837, %c-2 : index
    %846 = addi %844, %845 : index
    %847 = addi %846, %c1 : index
    %848 = cmpi "slt", %847, %c0 : index
    %849 = subi %c-1, %847 : index
    %850 = select %848, %849, %847 : index
    %851 = divi_signed %850, %c2 : index
    %852 = subi %c-1, %851 : index
    %853 = select %848, %852, %851 : index
    %854 = muli %853, %c-2 : index
    %855 = addi %846, %854 : index
    %856 = addi %855, %c1 : index
    %857 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %858 = vector.extractelement %857[%c0_i64 : i64] : vector<8xf32>
    %859 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %860 = vector.extractelement %859[%c1_i64 : i64] : vector<8xf32>
    %861 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %862 = vector.extractelement %861[%c2_i64 : i64] : vector<8xf32>
    %863 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %864 = vector.extractelement %863[%c3_i64 : i64] : vector<8xf32>
    %865 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %866 = vector.extractelement %865[%c4_i64 : i64] : vector<8xf32>
    %867 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %868 = vector.extractelement %867[%c5_i64 : i64] : vector<8xf32>
    %869 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %870 = vector.extractelement %869[%c6_i64 : i64] : vector<8xf32>
    %871 = load %3[%841, %723, %856] : memref<16x128x2xvector<8xf32>>
    %872 = vector.extractelement %871[%c7_i64 : i64] : vector<8xf32>
    %873 = mulf %823, %858 {RelaxedPrecision} : f32
    %874 = mulf %824, %860 {RelaxedPrecision} : f32
    %875 = mulf %825, %862 {RelaxedPrecision} : f32
    %876 = mulf %826, %864 {RelaxedPrecision} : f32
    %877 = mulf %827, %866 {RelaxedPrecision} : f32
    %878 = mulf %828, %868 {RelaxedPrecision} : f32
    %879 = mulf %829, %870 {RelaxedPrecision} : f32
    %880 = mulf %830, %872 {RelaxedPrecision} : f32
    %881 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %882 = vector.extractelement %881[%c0_i64 : i64] : vector<8xf32>
    %883 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %884 = vector.extractelement %883[%c1_i64 : i64] : vector<8xf32>
    %885 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %886 = vector.extractelement %885[%c2_i64 : i64] : vector<8xf32>
    %887 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %888 = vector.extractelement %887[%c3_i64 : i64] : vector<8xf32>
    %889 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %890 = vector.extractelement %889[%c4_i64 : i64] : vector<8xf32>
    %891 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %892 = vector.extractelement %891[%c5_i64 : i64] : vector<8xf32>
    %893 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %894 = vector.extractelement %893[%c6_i64 : i64] : vector<8xf32>
    %895 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %896 = vector.extractelement %895[%c7_i64 : i64] : vector<8xf32>
    %897 = addf %882, %873 {RelaxedPrecision} : f32
    %898 = addf %884, %874 {RelaxedPrecision} : f32
    %899 = addf %886, %875 {RelaxedPrecision} : f32
    %900 = addf %888, %876 {RelaxedPrecision} : f32
    %901 = addf %890, %877 {RelaxedPrecision} : f32
    %902 = addf %892, %878 {RelaxedPrecision} : f32
    %903 = addf %894, %879 {RelaxedPrecision} : f32
    %904 = addf %896, %880 {RelaxedPrecision} : f32
    %905 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %906 = vector.insertelement %897, %905[%c0_i64 : i64] : vector<8xf32>
    store %906, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %907 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %908 = vector.insertelement %898, %907[%c1_i64 : i64] : vector<8xf32>
    store %908, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %909 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %910 = vector.insertelement %899, %909[%c2_i64 : i64] : vector<8xf32>
    store %910, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %911 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %912 = vector.insertelement %900, %911[%c3_i64 : i64] : vector<8xf32>
    store %912, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %913 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %914 = vector.insertelement %901, %913[%c4_i64 : i64] : vector<8xf32>
    store %914, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %915 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %916 = vector.insertelement %902, %915[%c5_i64 : i64] : vector<8xf32>
    store %916, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %917 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %918 = vector.insertelement %903, %917[%c6_i64 : i64] : vector<8xf32>
    store %918, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %919 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %920 = vector.insertelement %904, %919[%c7_i64 : i64] : vector<8xf32>
    store %920, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %921 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %922 = vector.insertelement %897, %921[%c0_i64 : i64] : vector<8xf32>
    store %922, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %923 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %924 = vector.insertelement %898, %923[%c1_i64 : i64] : vector<8xf32>
    store %924, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %925 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %926 = vector.insertelement %899, %925[%c2_i64 : i64] : vector<8xf32>
    store %926, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %927 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %928 = vector.insertelement %900, %927[%c3_i64 : i64] : vector<8xf32>
    store %928, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %929 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %930 = vector.insertelement %901, %929[%c4_i64 : i64] : vector<8xf32>
    store %930, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %931 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %932 = vector.insertelement %902, %931[%c5_i64 : i64] : vector<8xf32>
    store %932, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %933 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %934 = vector.insertelement %903, %933[%c6_i64 : i64] : vector<8xf32>
    store %934, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %935 = load %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %936 = vector.insertelement %904, %935[%c7_i64 : i64] : vector<8xf32>
    store %936, %2[%841, %766, %856] : memref<16x6x2xvector<8xf32>>
    %937 = addi %697, %c1 : index
    br ^bb31(%937 : index)
  ^bb33:  // pred: ^bb31
    %938 = addi %695, %c1 : index
    br ^bb29(%938 : index)
  ^bb34:  // pred: ^bb29
    %939 = addi %693, %c6 : index
    br ^bb27(%939 : index)
  ^bb35:  // pred: ^bb27
    br ^bb36(%c0 : index)
  ^bb36(%940: index):  // 2 preds: ^bb35, ^bb37
    %941 = cmpi "slt", %940, %c4 : index
    cond_br %941, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    %942 = addi %691, %940 : index
    %943 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %944 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %945 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %946 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %947 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %948 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %949 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %950 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %951 = cmpi "slt", %689, %c0 : index
    %952 = subi %c-1, %689 : index
    %953 = select %951, %952, %689 : index
    %954 = divi_signed %953, %c16 : index
    %955 = subi %c-1, %954 : index
    %956 = select %951, %955, %954 : index
    %957 = remi_signed %956, %c16 : index
    %958 = cmpi "slt", %957, %c0 : index
    %959 = addi %957, %c16 : index
    %960 = select %958, %959, %957 : index
    %961 = remi_signed %942, %c128 : index
    %962 = cmpi "slt", %961, %c0 : index
    %963 = addi %961, %c128 : index
    %964 = select %962, %963, %961 : index
    %965 = remi_signed %689, %c16 : index
    %966 = cmpi "slt", %965, %c0 : index
    %967 = addi %965, %c16 : index
    %968 = select %966, %967, %965 : index
    %969 = cmpi "slt", %968, %c0 : index
    %970 = subi %c-1, %968 : index
    %971 = select %969, %970, %968 : index
    %972 = divi_signed %971, %c8 : index
    %973 = subi %c-1, %972 : index
    %974 = select %969, %973, %972 : index
    %975 = remi_signed %974, %c2 : index
    %976 = cmpi "slt", %975, %c0 : index
    %977 = addi %975, %c2 : index
    %978 = select %976, %977, %975 : index
    %979 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %980 = vector.extractelement %979[%c0_i64 : i64] : vector<8xf32>
    %981 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %982 = vector.extractelement %981[%c1_i64 : i64] : vector<8xf32>
    %983 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %984 = vector.extractelement %983[%c2_i64 : i64] : vector<8xf32>
    %985 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %986 = vector.extractelement %985[%c3_i64 : i64] : vector<8xf32>
    %987 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %988 = vector.extractelement %987[%c4_i64 : i64] : vector<8xf32>
    %989 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %990 = vector.extractelement %989[%c5_i64 : i64] : vector<8xf32>
    %991 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %992 = vector.extractelement %991[%c6_i64 : i64] : vector<8xf32>
    %993 = load %3[%960, %964, %978] : memref<16x128x2xvector<8xf32>>
    %994 = vector.extractelement %993[%c7_i64 : i64] : vector<8xf32>
    %995 = mulf %943, %980 {RelaxedPrecision} : f32
    %996 = mulf %944, %982 {RelaxedPrecision} : f32
    %997 = mulf %945, %984 {RelaxedPrecision} : f32
    %998 = mulf %946, %986 {RelaxedPrecision} : f32
    %999 = mulf %947, %988 {RelaxedPrecision} : f32
    %1000 = mulf %948, %990 {RelaxedPrecision} : f32
    %1001 = mulf %949, %992 {RelaxedPrecision} : f32
    %1002 = mulf %950, %994 {RelaxedPrecision} : f32
    %1003 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1004 = vector.extractelement %1003[%c0_i64 : i64] : vector<8xf32>
    %1005 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1006 = vector.extractelement %1005[%c1_i64 : i64] : vector<8xf32>
    %1007 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1008 = vector.extractelement %1007[%c2_i64 : i64] : vector<8xf32>
    %1009 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1010 = vector.extractelement %1009[%c3_i64 : i64] : vector<8xf32>
    %1011 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1012 = vector.extractelement %1011[%c4_i64 : i64] : vector<8xf32>
    %1013 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1014 = vector.extractelement %1013[%c5_i64 : i64] : vector<8xf32>
    %1015 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1016 = vector.extractelement %1015[%c6_i64 : i64] : vector<8xf32>
    %1017 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1018 = vector.extractelement %1017[%c7_i64 : i64] : vector<8xf32>
    %1019 = addf %1004, %995 {RelaxedPrecision} : f32
    %1020 = addf %1006, %996 {RelaxedPrecision} : f32
    %1021 = addf %1008, %997 {RelaxedPrecision} : f32
    %1022 = addf %1010, %998 {RelaxedPrecision} : f32
    %1023 = addf %1012, %999 {RelaxedPrecision} : f32
    %1024 = addf %1014, %1000 {RelaxedPrecision} : f32
    %1025 = addf %1016, %1001 {RelaxedPrecision} : f32
    %1026 = addf %1018, %1002 {RelaxedPrecision} : f32
    %1027 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1028 = vector.insertelement %1019, %1027[%c0_i64 : i64] : vector<8xf32>
    store %1028, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1029 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1030 = vector.insertelement %1020, %1029[%c1_i64 : i64] : vector<8xf32>
    store %1030, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1031 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1032 = vector.insertelement %1021, %1031[%c2_i64 : i64] : vector<8xf32>
    store %1032, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1033 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1034 = vector.insertelement %1022, %1033[%c3_i64 : i64] : vector<8xf32>
    store %1034, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1035 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1036 = vector.insertelement %1023, %1035[%c4_i64 : i64] : vector<8xf32>
    store %1036, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1037 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1038 = vector.insertelement %1024, %1037[%c5_i64 : i64] : vector<8xf32>
    store %1038, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1039 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1040 = vector.insertelement %1025, %1039[%c6_i64 : i64] : vector<8xf32>
    store %1040, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1041 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1042 = vector.insertelement %1026, %1041[%c7_i64 : i64] : vector<8xf32>
    store %1042, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1043 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1044 = vector.insertelement %1019, %1043[%c0_i64 : i64] : vector<8xf32>
    store %1044, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1045 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1046 = vector.insertelement %1020, %1045[%c1_i64 : i64] : vector<8xf32>
    store %1046, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1047 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1048 = vector.insertelement %1021, %1047[%c2_i64 : i64] : vector<8xf32>
    store %1048, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1049 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1050 = vector.insertelement %1022, %1049[%c3_i64 : i64] : vector<8xf32>
    store %1050, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1051 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1052 = vector.insertelement %1023, %1051[%c4_i64 : i64] : vector<8xf32>
    store %1052, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1053 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1054 = vector.insertelement %1024, %1053[%c5_i64 : i64] : vector<8xf32>
    store %1054, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1055 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1056 = vector.insertelement %1025, %1055[%c6_i64 : i64] : vector<8xf32>
    store %1056, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1057 = load %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1058 = vector.insertelement %1026, %1057[%c7_i64 : i64] : vector<8xf32>
    store %1058, %2[%960, %c0, %978] : memref<16x6x2xvector<8xf32>>
    %1059 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1060 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1061 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1062 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1063 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1064 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1065 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1066 = load %arg0[%678, %942] : memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>
    %1067 = addi %689, %c8 : index
    %1068 = cmpi "slt", %1067, %c0 : index
    %1069 = subi %c-1, %1067 : index
    %1070 = select %1068, %1069, %1067 : index
    %1071 = divi_signed %1070, %c16 : index
    %1072 = subi %c-1, %1071 : index
    %1073 = select %1068, %1072, %1071 : index
    %1074 = remi_signed %1073, %c16 : index
    %1075 = cmpi "slt", %1074, %c0 : index
    %1076 = addi %1074, %c16 : index
    %1077 = select %1075, %1076, %1074 : index
    %1078 = divi_signed %953, %c8 : index
    %1079 = subi %c-1, %1078 : index
    %1080 = select %951, %1079, %1078 : index
    %1081 = muli %1073, %c-2 : index
    %1082 = addi %1080, %1081 : index
    %1083 = addi %1082, %c1 : index
    %1084 = cmpi "slt", %1083, %c0 : index
    %1085 = subi %c-1, %1083 : index
    %1086 = select %1084, %1085, %1083 : index
    %1087 = divi_signed %1086, %c2 : index
    %1088 = subi %c-1, %1087 : index
    %1089 = select %1084, %1088, %1087 : index
    %1090 = muli %1089, %c-2 : index
    %1091 = addi %1082, %1090 : index
    %1092 = addi %1091, %c1 : index
    %1093 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1094 = vector.extractelement %1093[%c0_i64 : i64] : vector<8xf32>
    %1095 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1096 = vector.extractelement %1095[%c1_i64 : i64] : vector<8xf32>
    %1097 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1098 = vector.extractelement %1097[%c2_i64 : i64] : vector<8xf32>
    %1099 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1100 = vector.extractelement %1099[%c3_i64 : i64] : vector<8xf32>
    %1101 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1102 = vector.extractelement %1101[%c4_i64 : i64] : vector<8xf32>
    %1103 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1104 = vector.extractelement %1103[%c5_i64 : i64] : vector<8xf32>
    %1105 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1106 = vector.extractelement %1105[%c6_i64 : i64] : vector<8xf32>
    %1107 = load %3[%1077, %964, %1092] : memref<16x128x2xvector<8xf32>>
    %1108 = vector.extractelement %1107[%c7_i64 : i64] : vector<8xf32>
    %1109 = mulf %1059, %1094 {RelaxedPrecision} : f32
    %1110 = mulf %1060, %1096 {RelaxedPrecision} : f32
    %1111 = mulf %1061, %1098 {RelaxedPrecision} : f32
    %1112 = mulf %1062, %1100 {RelaxedPrecision} : f32
    %1113 = mulf %1063, %1102 {RelaxedPrecision} : f32
    %1114 = mulf %1064, %1104 {RelaxedPrecision} : f32
    %1115 = mulf %1065, %1106 {RelaxedPrecision} : f32
    %1116 = mulf %1066, %1108 {RelaxedPrecision} : f32
    %1117 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1118 = vector.extractelement %1117[%c0_i64 : i64] : vector<8xf32>
    %1119 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1120 = vector.extractelement %1119[%c1_i64 : i64] : vector<8xf32>
    %1121 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1122 = vector.extractelement %1121[%c2_i64 : i64] : vector<8xf32>
    %1123 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1124 = vector.extractelement %1123[%c3_i64 : i64] : vector<8xf32>
    %1125 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1126 = vector.extractelement %1125[%c4_i64 : i64] : vector<8xf32>
    %1127 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1128 = vector.extractelement %1127[%c5_i64 : i64] : vector<8xf32>
    %1129 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1130 = vector.extractelement %1129[%c6_i64 : i64] : vector<8xf32>
    %1131 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1132 = vector.extractelement %1131[%c7_i64 : i64] : vector<8xf32>
    %1133 = addf %1118, %1109 {RelaxedPrecision} : f32
    %1134 = addf %1120, %1110 {RelaxedPrecision} : f32
    %1135 = addf %1122, %1111 {RelaxedPrecision} : f32
    %1136 = addf %1124, %1112 {RelaxedPrecision} : f32
    %1137 = addf %1126, %1113 {RelaxedPrecision} : f32
    %1138 = addf %1128, %1114 {RelaxedPrecision} : f32
    %1139 = addf %1130, %1115 {RelaxedPrecision} : f32
    %1140 = addf %1132, %1116 {RelaxedPrecision} : f32
    %1141 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1142 = vector.insertelement %1133, %1141[%c0_i64 : i64] : vector<8xf32>
    store %1142, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1143 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1144 = vector.insertelement %1134, %1143[%c1_i64 : i64] : vector<8xf32>
    store %1144, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1145 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1146 = vector.insertelement %1135, %1145[%c2_i64 : i64] : vector<8xf32>
    store %1146, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1147 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1148 = vector.insertelement %1136, %1147[%c3_i64 : i64] : vector<8xf32>
    store %1148, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1149 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1150 = vector.insertelement %1137, %1149[%c4_i64 : i64] : vector<8xf32>
    store %1150, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1151 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1152 = vector.insertelement %1138, %1151[%c5_i64 : i64] : vector<8xf32>
    store %1152, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1153 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1154 = vector.insertelement %1139, %1153[%c6_i64 : i64] : vector<8xf32>
    store %1154, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1155 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1156 = vector.insertelement %1140, %1155[%c7_i64 : i64] : vector<8xf32>
    store %1156, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1157 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1158 = vector.insertelement %1133, %1157[%c0_i64 : i64] : vector<8xf32>
    store %1158, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1159 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1160 = vector.insertelement %1134, %1159[%c1_i64 : i64] : vector<8xf32>
    store %1160, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1161 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1162 = vector.insertelement %1135, %1161[%c2_i64 : i64] : vector<8xf32>
    store %1162, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1163 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1164 = vector.insertelement %1136, %1163[%c3_i64 : i64] : vector<8xf32>
    store %1164, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1165 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1166 = vector.insertelement %1137, %1165[%c4_i64 : i64] : vector<8xf32>
    store %1166, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1167 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1168 = vector.insertelement %1138, %1167[%c5_i64 : i64] : vector<8xf32>
    store %1168, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1169 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1170 = vector.insertelement %1139, %1169[%c6_i64 : i64] : vector<8xf32>
    store %1170, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1171 = load %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1172 = vector.insertelement %1140, %1171[%c7_i64 : i64] : vector<8xf32>
    store %1172, %2[%1077, %c0, %1092] : memref<16x6x2xvector<8xf32>>
    %1173 = addi %940, %c1 : index
    br ^bb36(%1173 : index)
  ^bb38:  // pred: ^bb36
    %1174 = addi %691, %c4 : index
    br ^bb25(%1174 : index)
  ^bb39:  // pred: ^bb25
    %1175 = addi %689, %c16 : index
    br ^bb23(%1175 : index)
  ^bb40:  // pred: ^bb23
    br ^bb41(%c0 : index)
  ^bb41(%1176: index):  // 2 preds: ^bb40, ^bb51
    %1177 = cmpi "slt", %1176, %c256 : index
    cond_br %1177, ^bb42, ^bb52
  ^bb42:  // pred: ^bb41
    cond_br %true, ^bb43, ^bb47
  ^bb43:  // pred: ^bb42
    %1178 = addi %4, %1176 : index
    %1179 = vector.transfer_read %arg2[%678, %1178], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1180 = cmpi "slt", %1176, %c0 : index
    %1181 = subi %c-1, %1176 : index
    %1182 = select %1180, %1181, %1176 : index
    %1183 = divi_signed %1182, %c16 : index
    %1184 = subi %c-1, %1183 : index
    %1185 = select %1180, %1184, %1183 : index
    %1186 = remi_signed %1185, %c16 : index
    %1187 = cmpi "slt", %1186, %c0 : index
    %1188 = addi %1186, %c16 : index
    %1189 = select %1187, %1188, %1186 : index
    %1190 = remi_signed %1176, %c16 : index
    %1191 = cmpi "slt", %1190, %c0 : index
    %1192 = addi %1190, %c16 : index
    %1193 = select %1191, %1192, %1190 : index
    %1194 = cmpi "slt", %1193, %c0 : index
    %1195 = subi %c-1, %1193 : index
    %1196 = select %1194, %1195, %1193 : index
    %1197 = divi_signed %1196, %c8 : index
    %1198 = subi %c-1, %1197 : index
    %1199 = select %1194, %1198, %1197 : index
    %1200 = remi_signed %1199, %c2 : index
    %1201 = cmpi "slt", %1200, %c0 : index
    %1202 = addi %1200, %c2 : index
    %1203 = select %1201, %1202, %1200 : index
    %1204 = load %2[%1189, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1205 = addf %1179, %1204 : vector<8xf32>
    store %1205, %1[%c0, %c0] : memref<1x16xvector<8xf32>>
    %1206 = addi %1178, %c8 : index
    %1207 = vector.transfer_read %arg2[%678, %1206], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1208 = addi %1176, %c8 : index
    %1209 = cmpi "slt", %1208, %c0 : index
    %1210 = subi %c-1, %1208 : index
    %1211 = select %1209, %1210, %1208 : index
    %1212 = divi_signed %1211, %c16 : index
    %1213 = subi %c-1, %1212 : index
    %1214 = select %1209, %1213, %1212 : index
    %1215 = remi_signed %1214, %c16 : index
    %1216 = cmpi "slt", %1215, %c0 : index
    %1217 = addi %1215, %c16 : index
    %1218 = select %1216, %1217, %1215 : index
    %1219 = divi_signed %1182, %c8 : index
    %1220 = subi %c-1, %1219 : index
    %1221 = select %1180, %1220, %1219 : index
    %1222 = muli %1214, %c-2 : index
    %1223 = addi %1221, %1222 : index
    %1224 = addi %1223, %c1 : index
    %1225 = cmpi "slt", %1224, %c0 : index
    %1226 = subi %c-1, %1224 : index
    %1227 = select %1225, %1226, %1224 : index
    %1228 = divi_signed %1227, %c2 : index
    %1229 = subi %c-1, %1228 : index
    %1230 = select %1225, %1229, %1228 : index
    %1231 = muli %1230, %c-2 : index
    %1232 = addi %1223, %1231 : index
    %1233 = addi %1232, %c1 : index
    %1234 = load %2[%1218, %c0, %1233] : memref<16x6x2xvector<8xf32>>
    %1235 = addf %1207, %1234 : vector<8xf32>
    store %1235, %1[%c0, %c1] : memref<1x16xvector<8xf32>>
    %1236 = addi %1178, %c16 : index
    %1237 = vector.transfer_read %arg2[%678, %1236], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1238 = addi %1185, %c1 : index
    %1239 = cmpi "slt", %1238, %c0 : index
    %1240 = subi %c-1, %1238 : index
    %1241 = select %1239, %1240, %1238 : index
    %1242 = divi_signed %1241, %c16 : index
    %1243 = subi %c-1, %1242 : index
    %1244 = select %1239, %1243, %1242 : index
    %1245 = muli %1244, %c-16 : index
    %1246 = addi %1185, %1245 : index
    %1247 = addi %1246, %c1 : index
    %1248 = load %2[%1247, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1249 = addf %1237, %1248 : vector<8xf32>
    store %1249, %1[%c0, %c2] : memref<1x16xvector<8xf32>>
    %1250 = addi %1178, %c24 : index
    %1251 = vector.transfer_read %arg2[%678, %1250], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1252 = addi %1176, %c24 : index
    %1253 = cmpi "slt", %1252, %c0 : index
    %1254 = subi %c-1, %1252 : index
    %1255 = select %1253, %1254, %1252 : index
    %1256 = divi_signed %1255, %c16 : index
    %1257 = subi %c-1, %1256 : index
    %1258 = select %1253, %1257, %1256 : index
    %1259 = remi_signed %1258, %c16 : index
    %1260 = cmpi "slt", %1259, %c0 : index
    %1261 = addi %1259, %c16 : index
    %1262 = select %1260, %1261, %1259 : index
    %1263 = muli %1258, %c-2 : index
    %1264 = addi %1221, %1263 : index
    %1265 = addi %1264, %c3 : index
    %1266 = cmpi "slt", %1265, %c0 : index
    %1267 = subi %c-1, %1265 : index
    %1268 = select %1266, %1267, %1265 : index
    %1269 = divi_signed %1268, %c2 : index
    %1270 = subi %c-1, %1269 : index
    %1271 = select %1266, %1270, %1269 : index
    %1272 = muli %1271, %c-2 : index
    %1273 = addi %1264, %1272 : index
    %1274 = addi %1273, %c3 : index
    %1275 = load %2[%1262, %c0, %1274] : memref<16x6x2xvector<8xf32>>
    %1276 = addf %1251, %1275 : vector<8xf32>
    store %1276, %1[%c0, %c3] : memref<1x16xvector<8xf32>>
    %1277 = addi %1178, %c32 : index
    %1278 = vector.transfer_read %arg2[%678, %1277], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1279 = addi %1185, %c2 : index
    %1280 = cmpi "slt", %1279, %c0 : index
    %1281 = subi %c-1, %1279 : index
    %1282 = select %1280, %1281, %1279 : index
    %1283 = divi_signed %1282, %c16 : index
    %1284 = subi %c-1, %1283 : index
    %1285 = select %1280, %1284, %1283 : index
    %1286 = muli %1285, %c-16 : index
    %1287 = addi %1185, %1286 : index
    %1288 = addi %1287, %c2 : index
    %1289 = load %2[%1288, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1290 = addf %1278, %1289 : vector<8xf32>
    store %1290, %1[%c0, %c4] : memref<1x16xvector<8xf32>>
    %1291 = addi %1178, %c40 : index
    %1292 = vector.transfer_read %arg2[%678, %1291], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1293 = addi %1176, %c40 : index
    %1294 = cmpi "slt", %1293, %c0 : index
    %1295 = subi %c-1, %1293 : index
    %1296 = select %1294, %1295, %1293 : index
    %1297 = divi_signed %1296, %c16 : index
    %1298 = subi %c-1, %1297 : index
    %1299 = select %1294, %1298, %1297 : index
    %1300 = remi_signed %1299, %c16 : index
    %1301 = cmpi "slt", %1300, %c0 : index
    %1302 = addi %1300, %c16 : index
    %1303 = select %1301, %1302, %1300 : index
    %1304 = muli %1299, %c-2 : index
    %1305 = addi %1221, %1304 : index
    %1306 = addi %1305, %c5 : index
    %1307 = cmpi "slt", %1306, %c0 : index
    %1308 = subi %c-1, %1306 : index
    %1309 = select %1307, %1308, %1306 : index
    %1310 = divi_signed %1309, %c2 : index
    %1311 = subi %c-1, %1310 : index
    %1312 = select %1307, %1311, %1310 : index
    %1313 = muli %1312, %c-2 : index
    %1314 = addi %1305, %1313 : index
    %1315 = addi %1314, %c5 : index
    %1316 = load %2[%1303, %c0, %1315] : memref<16x6x2xvector<8xf32>>
    %1317 = addf %1292, %1316 : vector<8xf32>
    store %1317, %1[%c0, %c5] : memref<1x16xvector<8xf32>>
    %1318 = addi %1178, %c48 : index
    %1319 = vector.transfer_read %arg2[%678, %1318], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1320 = addi %1185, %c3 : index
    %1321 = cmpi "slt", %1320, %c0 : index
    %1322 = subi %c-1, %1320 : index
    %1323 = select %1321, %1322, %1320 : index
    %1324 = divi_signed %1323, %c16 : index
    %1325 = subi %c-1, %1324 : index
    %1326 = select %1321, %1325, %1324 : index
    %1327 = muli %1326, %c-16 : index
    %1328 = addi %1185, %1327 : index
    %1329 = addi %1328, %c3 : index
    %1330 = load %2[%1329, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1331 = addf %1319, %1330 : vector<8xf32>
    store %1331, %1[%c0, %c6] : memref<1x16xvector<8xf32>>
    %1332 = addi %1178, %c56 : index
    %1333 = vector.transfer_read %arg2[%678, %1332], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1334 = addi %1176, %c56 : index
    %1335 = cmpi "slt", %1334, %c0 : index
    %1336 = subi %c-1, %1334 : index
    %1337 = select %1335, %1336, %1334 : index
    %1338 = divi_signed %1337, %c16 : index
    %1339 = subi %c-1, %1338 : index
    %1340 = select %1335, %1339, %1338 : index
    %1341 = remi_signed %1340, %c16 : index
    %1342 = cmpi "slt", %1341, %c0 : index
    %1343 = addi %1341, %c16 : index
    %1344 = select %1342, %1343, %1341 : index
    %1345 = muli %1340, %c-2 : index
    %1346 = addi %1221, %1345 : index
    %1347 = addi %1346, %c7 : index
    %1348 = cmpi "slt", %1347, %c0 : index
    %1349 = subi %c-1, %1347 : index
    %1350 = select %1348, %1349, %1347 : index
    %1351 = divi_signed %1350, %c2 : index
    %1352 = subi %c-1, %1351 : index
    %1353 = select %1348, %1352, %1351 : index
    %1354 = muli %1353, %c-2 : index
    %1355 = addi %1346, %1354 : index
    %1356 = addi %1355, %c7 : index
    %1357 = load %2[%1344, %c0, %1356] : memref<16x6x2xvector<8xf32>>
    %1358 = addf %1333, %1357 : vector<8xf32>
    store %1358, %1[%c0, %c7] : memref<1x16xvector<8xf32>>
    %1359 = addi %1178, %c64 : index
    %1360 = vector.transfer_read %arg2[%678, %1359], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1361 = addi %1185, %c4 : index
    %1362 = cmpi "slt", %1361, %c0 : index
    %1363 = subi %c-1, %1361 : index
    %1364 = select %1362, %1363, %1361 : index
    %1365 = divi_signed %1364, %c16 : index
    %1366 = subi %c-1, %1365 : index
    %1367 = select %1362, %1366, %1365 : index
    %1368 = muli %1367, %c-16 : index
    %1369 = addi %1185, %1368 : index
    %1370 = addi %1369, %c4 : index
    %1371 = load %2[%1370, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1372 = addf %1360, %1371 : vector<8xf32>
    store %1372, %1[%c0, %c8] : memref<1x16xvector<8xf32>>
    %1373 = addi %1178, %c72 : index
    %1374 = vector.transfer_read %arg2[%678, %1373], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1375 = addi %1176, %c72 : index
    %1376 = cmpi "slt", %1375, %c0 : index
    %1377 = subi %c-1, %1375 : index
    %1378 = select %1376, %1377, %1375 : index
    %1379 = divi_signed %1378, %c16 : index
    %1380 = subi %c-1, %1379 : index
    %1381 = select %1376, %1380, %1379 : index
    %1382 = remi_signed %1381, %c16 : index
    %1383 = cmpi "slt", %1382, %c0 : index
    %1384 = addi %1382, %c16 : index
    %1385 = select %1383, %1384, %1382 : index
    %1386 = muli %1381, %c-2 : index
    %1387 = addi %1221, %1386 : index
    %1388 = addi %1387, %c9 : index
    %1389 = cmpi "slt", %1388, %c0 : index
    %1390 = subi %c-1, %1388 : index
    %1391 = select %1389, %1390, %1388 : index
    %1392 = divi_signed %1391, %c2 : index
    %1393 = subi %c-1, %1392 : index
    %1394 = select %1389, %1393, %1392 : index
    %1395 = muli %1394, %c-2 : index
    %1396 = addi %1387, %1395 : index
    %1397 = addi %1396, %c9 : index
    %1398 = load %2[%1385, %c0, %1397] : memref<16x6x2xvector<8xf32>>
    %1399 = addf %1374, %1398 : vector<8xf32>
    store %1399, %1[%c0, %c9] : memref<1x16xvector<8xf32>>
    %1400 = addi %1178, %c80 : index
    %1401 = vector.transfer_read %arg2[%678, %1400], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1402 = addi %1185, %c5 : index
    %1403 = cmpi "slt", %1402, %c0 : index
    %1404 = subi %c-1, %1402 : index
    %1405 = select %1403, %1404, %1402 : index
    %1406 = divi_signed %1405, %c16 : index
    %1407 = subi %c-1, %1406 : index
    %1408 = select %1403, %1407, %1406 : index
    %1409 = muli %1408, %c-16 : index
    %1410 = addi %1185, %1409 : index
    %1411 = addi %1410, %c5 : index
    %1412 = load %2[%1411, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1413 = addf %1401, %1412 : vector<8xf32>
    store %1413, %1[%c0, %c10] : memref<1x16xvector<8xf32>>
    %1414 = addi %1178, %c88 : index
    %1415 = vector.transfer_read %arg2[%678, %1414], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1416 = addi %1176, %c88 : index
    %1417 = cmpi "slt", %1416, %c0 : index
    %1418 = subi %c-1, %1416 : index
    %1419 = select %1417, %1418, %1416 : index
    %1420 = divi_signed %1419, %c16 : index
    %1421 = subi %c-1, %1420 : index
    %1422 = select %1417, %1421, %1420 : index
    %1423 = remi_signed %1422, %c16 : index
    %1424 = cmpi "slt", %1423, %c0 : index
    %1425 = addi %1423, %c16 : index
    %1426 = select %1424, %1425, %1423 : index
    %1427 = muli %1422, %c-2 : index
    %1428 = addi %1221, %1427 : index
    %1429 = addi %1428, %c11 : index
    %1430 = cmpi "slt", %1429, %c0 : index
    %1431 = subi %c-1, %1429 : index
    %1432 = select %1430, %1431, %1429 : index
    %1433 = divi_signed %1432, %c2 : index
    %1434 = subi %c-1, %1433 : index
    %1435 = select %1430, %1434, %1433 : index
    %1436 = muli %1435, %c-2 : index
    %1437 = addi %1428, %1436 : index
    %1438 = addi %1437, %c11 : index
    %1439 = load %2[%1426, %c0, %1438] : memref<16x6x2xvector<8xf32>>
    %1440 = addf %1415, %1439 : vector<8xf32>
    store %1440, %1[%c0, %c11] : memref<1x16xvector<8xf32>>
    %1441 = addi %1178, %c96 : index
    %1442 = vector.transfer_read %arg2[%678, %1441], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1443 = addi %1185, %c6 : index
    %1444 = cmpi "slt", %1443, %c0 : index
    %1445 = subi %c-1, %1443 : index
    %1446 = select %1444, %1445, %1443 : index
    %1447 = divi_signed %1446, %c16 : index
    %1448 = subi %c-1, %1447 : index
    %1449 = select %1444, %1448, %1447 : index
    %1450 = muli %1449, %c-16 : index
    %1451 = addi %1185, %1450 : index
    %1452 = addi %1451, %c6 : index
    %1453 = load %2[%1452, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1454 = addf %1442, %1453 : vector<8xf32>
    store %1454, %1[%c0, %c12] : memref<1x16xvector<8xf32>>
    %1455 = addi %1178, %c104 : index
    %1456 = vector.transfer_read %arg2[%678, %1455], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1457 = addi %1176, %c104 : index
    %1458 = cmpi "slt", %1457, %c0 : index
    %1459 = subi %c-1, %1457 : index
    %1460 = select %1458, %1459, %1457 : index
    %1461 = divi_signed %1460, %c16 : index
    %1462 = subi %c-1, %1461 : index
    %1463 = select %1458, %1462, %1461 : index
    %1464 = remi_signed %1463, %c16 : index
    %1465 = cmpi "slt", %1464, %c0 : index
    %1466 = addi %1464, %c16 : index
    %1467 = select %1465, %1466, %1464 : index
    %1468 = muli %1463, %c-2 : index
    %1469 = addi %1221, %1468 : index
    %1470 = addi %1469, %c13 : index
    %1471 = cmpi "slt", %1470, %c0 : index
    %1472 = subi %c-1, %1470 : index
    %1473 = select %1471, %1472, %1470 : index
    %1474 = divi_signed %1473, %c2 : index
    %1475 = subi %c-1, %1474 : index
    %1476 = select %1471, %1475, %1474 : index
    %1477 = muli %1476, %c-2 : index
    %1478 = addi %1469, %1477 : index
    %1479 = addi %1478, %c13 : index
    %1480 = load %2[%1467, %c0, %1479] : memref<16x6x2xvector<8xf32>>
    %1481 = addf %1456, %1480 : vector<8xf32>
    store %1481, %1[%c0, %c13] : memref<1x16xvector<8xf32>>
    %1482 = addi %1178, %c112 : index
    %1483 = vector.transfer_read %arg2[%678, %1482], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1484 = addi %1185, %c7 : index
    %1485 = cmpi "slt", %1484, %c0 : index
    %1486 = subi %c-1, %1484 : index
    %1487 = select %1485, %1486, %1484 : index
    %1488 = divi_signed %1487, %c16 : index
    %1489 = subi %c-1, %1488 : index
    %1490 = select %1485, %1489, %1488 : index
    %1491 = muli %1490, %c-16 : index
    %1492 = addi %1185, %1491 : index
    %1493 = addi %1492, %c7 : index
    %1494 = load %2[%1493, %c0, %1203] : memref<16x6x2xvector<8xf32>>
    %1495 = addf %1483, %1494 : vector<8xf32>
    store %1495, %1[%c0, %c14] : memref<1x16xvector<8xf32>>
    %1496 = addi %1178, %c120 : index
    %1497 = vector.transfer_read %arg2[%678, %1496], %cst {masked = [false]} : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1498 = addi %1176, %c120 : index
    %1499 = cmpi "slt", %1498, %c0 : index
    %1500 = subi %c-1, %1498 : index
    %1501 = select %1499, %1500, %1498 : index
    %1502 = divi_signed %1501, %c16 : index
    %1503 = subi %c-1, %1502 : index
    %1504 = select %1499, %1503, %1502 : index
    %1505 = remi_signed %1504, %c16 : index
    %1506 = cmpi "slt", %1505, %c0 : index
    %1507 = addi %1505, %c16 : index
    %1508 = select %1506, %1507, %1505 : index
    %1509 = muli %1504, %c-2 : index
    %1510 = addi %1221, %1509 : index
    %1511 = addi %1510, %c15 : index
    %1512 = cmpi "slt", %1511, %c0 : index
    %1513 = subi %c-1, %1511 : index
    %1514 = select %1512, %1513, %1511 : index
    %1515 = divi_signed %1514, %c2 : index
    %1516 = subi %c-1, %1515 : index
    %1517 = select %1512, %1516, %1515 : index
    %1518 = muli %1517, %c-2 : index
    %1519 = addi %1510, %1518 : index
    %1520 = addi %1519, %c15 : index
    %1521 = load %2[%1508, %c0, %1520] : memref<16x6x2xvector<8xf32>>
    %1522 = addf %1497, %1521 : vector<8xf32>
    store %1522, %1[%c0, %c15] : memref<1x16xvector<8xf32>>
    br ^bb44(%c0 : index)
  ^bb44(%1523: index):  // 2 preds: ^bb43, ^bb45
    %1524 = cmpi "slt", %1523, %c16 : index
    cond_br %1524, ^bb45, ^bb46
  ^bb45:  // pred: ^bb44
    %1525 = muli %1523, %c8 : index
    %1526 = addi %1178, %1525 : index
    %1527 = load %1[%c0, %1523] : memref<1x16xvector<8xf32>>
    vector.transfer_write %1527, %arg2[%678, %1526] {masked = [false]} : vector<8xf32>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1528 = addi %1523, %c1 : index
    br ^bb44(%1528 : index)
  ^bb46:  // pred: ^bb44
    br ^bb51
  ^bb47:  // pred: ^bb42
    %1529 = addi %4, %1176 : index
    %1530 = vector.transfer_read %arg2[%678, %1529], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1531 = cmpi "slt", %1176, %c0 : index
    %1532 = subi %c-1, %1176 : index
    %1533 = select %1531, %1532, %1176 : index
    %1534 = divi_signed %1533, %c16 : index
    %1535 = subi %c-1, %1534 : index
    %1536 = select %1531, %1535, %1534 : index
    %1537 = remi_signed %1536, %c16 : index
    %1538 = cmpi "slt", %1537, %c0 : index
    %1539 = addi %1537, %c16 : index
    %1540 = select %1538, %1539, %1537 : index
    %1541 = remi_signed %1176, %c16 : index
    %1542 = cmpi "slt", %1541, %c0 : index
    %1543 = addi %1541, %c16 : index
    %1544 = select %1542, %1543, %1541 : index
    %1545 = cmpi "slt", %1544, %c0 : index
    %1546 = subi %c-1, %1544 : index
    %1547 = select %1545, %1546, %1544 : index
    %1548 = divi_signed %1547, %c8 : index
    %1549 = subi %c-1, %1548 : index
    %1550 = select %1545, %1549, %1548 : index
    %1551 = remi_signed %1550, %c2 : index
    %1552 = cmpi "slt", %1551, %c0 : index
    %1553 = addi %1551, %c2 : index
    %1554 = select %1552, %1553, %1551 : index
    %1555 = load %2[%1540, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1556 = addf %1530, %1555 : vector<8xf32>
    store %1556, %1[%c0, %c0] : memref<1x16xvector<8xf32>>
    %1557 = addi %1529, %c8 : index
    %1558 = vector.transfer_read %arg2[%678, %1557], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1559 = addi %1176, %c8 : index
    %1560 = cmpi "slt", %1559, %c0 : index
    %1561 = subi %c-1, %1559 : index
    %1562 = select %1560, %1561, %1559 : index
    %1563 = divi_signed %1562, %c16 : index
    %1564 = subi %c-1, %1563 : index
    %1565 = select %1560, %1564, %1563 : index
    %1566 = remi_signed %1565, %c16 : index
    %1567 = cmpi "slt", %1566, %c0 : index
    %1568 = addi %1566, %c16 : index
    %1569 = select %1567, %1568, %1566 : index
    %1570 = divi_signed %1533, %c8 : index
    %1571 = subi %c-1, %1570 : index
    %1572 = select %1531, %1571, %1570 : index
    %1573 = muli %1565, %c-2 : index
    %1574 = addi %1572, %1573 : index
    %1575 = addi %1574, %c1 : index
    %1576 = cmpi "slt", %1575, %c0 : index
    %1577 = subi %c-1, %1575 : index
    %1578 = select %1576, %1577, %1575 : index
    %1579 = divi_signed %1578, %c2 : index
    %1580 = subi %c-1, %1579 : index
    %1581 = select %1576, %1580, %1579 : index
    %1582 = muli %1581, %c-2 : index
    %1583 = addi %1574, %1582 : index
    %1584 = addi %1583, %c1 : index
    %1585 = load %2[%1569, %c0, %1584] : memref<16x6x2xvector<8xf32>>
    %1586 = addf %1558, %1585 : vector<8xf32>
    store %1586, %1[%c0, %c1] : memref<1x16xvector<8xf32>>
    %1587 = addi %1529, %c16 : index
    %1588 = vector.transfer_read %arg2[%678, %1587], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1589 = addi %1536, %c1 : index
    %1590 = cmpi "slt", %1589, %c0 : index
    %1591 = subi %c-1, %1589 : index
    %1592 = select %1590, %1591, %1589 : index
    %1593 = divi_signed %1592, %c16 : index
    %1594 = subi %c-1, %1593 : index
    %1595 = select %1590, %1594, %1593 : index
    %1596 = muli %1595, %c-16 : index
    %1597 = addi %1536, %1596 : index
    %1598 = addi %1597, %c1 : index
    %1599 = load %2[%1598, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1600 = addf %1588, %1599 : vector<8xf32>
    store %1600, %1[%c0, %c2] : memref<1x16xvector<8xf32>>
    %1601 = addi %1529, %c24 : index
    %1602 = vector.transfer_read %arg2[%678, %1601], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1603 = addi %1176, %c24 : index
    %1604 = cmpi "slt", %1603, %c0 : index
    %1605 = subi %c-1, %1603 : index
    %1606 = select %1604, %1605, %1603 : index
    %1607 = divi_signed %1606, %c16 : index
    %1608 = subi %c-1, %1607 : index
    %1609 = select %1604, %1608, %1607 : index
    %1610 = remi_signed %1609, %c16 : index
    %1611 = cmpi "slt", %1610, %c0 : index
    %1612 = addi %1610, %c16 : index
    %1613 = select %1611, %1612, %1610 : index
    %1614 = muli %1609, %c-2 : index
    %1615 = addi %1572, %1614 : index
    %1616 = addi %1615, %c3 : index
    %1617 = cmpi "slt", %1616, %c0 : index
    %1618 = subi %c-1, %1616 : index
    %1619 = select %1617, %1618, %1616 : index
    %1620 = divi_signed %1619, %c2 : index
    %1621 = subi %c-1, %1620 : index
    %1622 = select %1617, %1621, %1620 : index
    %1623 = muli %1622, %c-2 : index
    %1624 = addi %1615, %1623 : index
    %1625 = addi %1624, %c3 : index
    %1626 = load %2[%1613, %c0, %1625] : memref<16x6x2xvector<8xf32>>
    %1627 = addf %1602, %1626 : vector<8xf32>
    store %1627, %1[%c0, %c3] : memref<1x16xvector<8xf32>>
    %1628 = addi %1529, %c32 : index
    %1629 = vector.transfer_read %arg2[%678, %1628], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1630 = addi %1536, %c2 : index
    %1631 = cmpi "slt", %1630, %c0 : index
    %1632 = subi %c-1, %1630 : index
    %1633 = select %1631, %1632, %1630 : index
    %1634 = divi_signed %1633, %c16 : index
    %1635 = subi %c-1, %1634 : index
    %1636 = select %1631, %1635, %1634 : index
    %1637 = muli %1636, %c-16 : index
    %1638 = addi %1536, %1637 : index
    %1639 = addi %1638, %c2 : index
    %1640 = load %2[%1639, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1641 = addf %1629, %1640 : vector<8xf32>
    store %1641, %1[%c0, %c4] : memref<1x16xvector<8xf32>>
    %1642 = addi %1529, %c40 : index
    %1643 = vector.transfer_read %arg2[%678, %1642], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1644 = addi %1176, %c40 : index
    %1645 = cmpi "slt", %1644, %c0 : index
    %1646 = subi %c-1, %1644 : index
    %1647 = select %1645, %1646, %1644 : index
    %1648 = divi_signed %1647, %c16 : index
    %1649 = subi %c-1, %1648 : index
    %1650 = select %1645, %1649, %1648 : index
    %1651 = remi_signed %1650, %c16 : index
    %1652 = cmpi "slt", %1651, %c0 : index
    %1653 = addi %1651, %c16 : index
    %1654 = select %1652, %1653, %1651 : index
    %1655 = muli %1650, %c-2 : index
    %1656 = addi %1572, %1655 : index
    %1657 = addi %1656, %c5 : index
    %1658 = cmpi "slt", %1657, %c0 : index
    %1659 = subi %c-1, %1657 : index
    %1660 = select %1658, %1659, %1657 : index
    %1661 = divi_signed %1660, %c2 : index
    %1662 = subi %c-1, %1661 : index
    %1663 = select %1658, %1662, %1661 : index
    %1664 = muli %1663, %c-2 : index
    %1665 = addi %1656, %1664 : index
    %1666 = addi %1665, %c5 : index
    %1667 = load %2[%1654, %c0, %1666] : memref<16x6x2xvector<8xf32>>
    %1668 = addf %1643, %1667 : vector<8xf32>
    store %1668, %1[%c0, %c5] : memref<1x16xvector<8xf32>>
    %1669 = addi %1529, %c48 : index
    %1670 = vector.transfer_read %arg2[%678, %1669], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1671 = addi %1536, %c3 : index
    %1672 = cmpi "slt", %1671, %c0 : index
    %1673 = subi %c-1, %1671 : index
    %1674 = select %1672, %1673, %1671 : index
    %1675 = divi_signed %1674, %c16 : index
    %1676 = subi %c-1, %1675 : index
    %1677 = select %1672, %1676, %1675 : index
    %1678 = muli %1677, %c-16 : index
    %1679 = addi %1536, %1678 : index
    %1680 = addi %1679, %c3 : index
    %1681 = load %2[%1680, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1682 = addf %1670, %1681 : vector<8xf32>
    store %1682, %1[%c0, %c6] : memref<1x16xvector<8xf32>>
    %1683 = addi %1529, %c56 : index
    %1684 = vector.transfer_read %arg2[%678, %1683], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1685 = addi %1176, %c56 : index
    %1686 = cmpi "slt", %1685, %c0 : index
    %1687 = subi %c-1, %1685 : index
    %1688 = select %1686, %1687, %1685 : index
    %1689 = divi_signed %1688, %c16 : index
    %1690 = subi %c-1, %1689 : index
    %1691 = select %1686, %1690, %1689 : index
    %1692 = remi_signed %1691, %c16 : index
    %1693 = cmpi "slt", %1692, %c0 : index
    %1694 = addi %1692, %c16 : index
    %1695 = select %1693, %1694, %1692 : index
    %1696 = muli %1691, %c-2 : index
    %1697 = addi %1572, %1696 : index
    %1698 = addi %1697, %c7 : index
    %1699 = cmpi "slt", %1698, %c0 : index
    %1700 = subi %c-1, %1698 : index
    %1701 = select %1699, %1700, %1698 : index
    %1702 = divi_signed %1701, %c2 : index
    %1703 = subi %c-1, %1702 : index
    %1704 = select %1699, %1703, %1702 : index
    %1705 = muli %1704, %c-2 : index
    %1706 = addi %1697, %1705 : index
    %1707 = addi %1706, %c7 : index
    %1708 = load %2[%1695, %c0, %1707] : memref<16x6x2xvector<8xf32>>
    %1709 = addf %1684, %1708 : vector<8xf32>
    store %1709, %1[%c0, %c7] : memref<1x16xvector<8xf32>>
    %1710 = addi %1529, %c64 : index
    %1711 = vector.transfer_read %arg2[%678, %1710], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1712 = addi %1536, %c4 : index
    %1713 = cmpi "slt", %1712, %c0 : index
    %1714 = subi %c-1, %1712 : index
    %1715 = select %1713, %1714, %1712 : index
    %1716 = divi_signed %1715, %c16 : index
    %1717 = subi %c-1, %1716 : index
    %1718 = select %1713, %1717, %1716 : index
    %1719 = muli %1718, %c-16 : index
    %1720 = addi %1536, %1719 : index
    %1721 = addi %1720, %c4 : index
    %1722 = load %2[%1721, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1723 = addf %1711, %1722 : vector<8xf32>
    store %1723, %1[%c0, %c8] : memref<1x16xvector<8xf32>>
    %1724 = addi %1529, %c72 : index
    %1725 = vector.transfer_read %arg2[%678, %1724], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1726 = addi %1176, %c72 : index
    %1727 = cmpi "slt", %1726, %c0 : index
    %1728 = subi %c-1, %1726 : index
    %1729 = select %1727, %1728, %1726 : index
    %1730 = divi_signed %1729, %c16 : index
    %1731 = subi %c-1, %1730 : index
    %1732 = select %1727, %1731, %1730 : index
    %1733 = remi_signed %1732, %c16 : index
    %1734 = cmpi "slt", %1733, %c0 : index
    %1735 = addi %1733, %c16 : index
    %1736 = select %1734, %1735, %1733 : index
    %1737 = muli %1732, %c-2 : index
    %1738 = addi %1572, %1737 : index
    %1739 = addi %1738, %c9 : index
    %1740 = cmpi "slt", %1739, %c0 : index
    %1741 = subi %c-1, %1739 : index
    %1742 = select %1740, %1741, %1739 : index
    %1743 = divi_signed %1742, %c2 : index
    %1744 = subi %c-1, %1743 : index
    %1745 = select %1740, %1744, %1743 : index
    %1746 = muli %1745, %c-2 : index
    %1747 = addi %1738, %1746 : index
    %1748 = addi %1747, %c9 : index
    %1749 = load %2[%1736, %c0, %1748] : memref<16x6x2xvector<8xf32>>
    %1750 = addf %1725, %1749 : vector<8xf32>
    store %1750, %1[%c0, %c9] : memref<1x16xvector<8xf32>>
    %1751 = addi %1529, %c80 : index
    %1752 = vector.transfer_read %arg2[%678, %1751], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1753 = addi %1536, %c5 : index
    %1754 = cmpi "slt", %1753, %c0 : index
    %1755 = subi %c-1, %1753 : index
    %1756 = select %1754, %1755, %1753 : index
    %1757 = divi_signed %1756, %c16 : index
    %1758 = subi %c-1, %1757 : index
    %1759 = select %1754, %1758, %1757 : index
    %1760 = muli %1759, %c-16 : index
    %1761 = addi %1536, %1760 : index
    %1762 = addi %1761, %c5 : index
    %1763 = load %2[%1762, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1764 = addf %1752, %1763 : vector<8xf32>
    store %1764, %1[%c0, %c10] : memref<1x16xvector<8xf32>>
    %1765 = addi %1529, %c88 : index
    %1766 = vector.transfer_read %arg2[%678, %1765], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1767 = addi %1176, %c88 : index
    %1768 = cmpi "slt", %1767, %c0 : index
    %1769 = subi %c-1, %1767 : index
    %1770 = select %1768, %1769, %1767 : index
    %1771 = divi_signed %1770, %c16 : index
    %1772 = subi %c-1, %1771 : index
    %1773 = select %1768, %1772, %1771 : index
    %1774 = remi_signed %1773, %c16 : index
    %1775 = cmpi "slt", %1774, %c0 : index
    %1776 = addi %1774, %c16 : index
    %1777 = select %1775, %1776, %1774 : index
    %1778 = muli %1773, %c-2 : index
    %1779 = addi %1572, %1778 : index
    %1780 = addi %1779, %c11 : index
    %1781 = cmpi "slt", %1780, %c0 : index
    %1782 = subi %c-1, %1780 : index
    %1783 = select %1781, %1782, %1780 : index
    %1784 = divi_signed %1783, %c2 : index
    %1785 = subi %c-1, %1784 : index
    %1786 = select %1781, %1785, %1784 : index
    %1787 = muli %1786, %c-2 : index
    %1788 = addi %1779, %1787 : index
    %1789 = addi %1788, %c11 : index
    %1790 = load %2[%1777, %c0, %1789] : memref<16x6x2xvector<8xf32>>
    %1791 = addf %1766, %1790 : vector<8xf32>
    store %1791, %1[%c0, %c11] : memref<1x16xvector<8xf32>>
    %1792 = addi %1529, %c96 : index
    %1793 = vector.transfer_read %arg2[%678, %1792], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1794 = addi %1536, %c6 : index
    %1795 = cmpi "slt", %1794, %c0 : index
    %1796 = subi %c-1, %1794 : index
    %1797 = select %1795, %1796, %1794 : index
    %1798 = divi_signed %1797, %c16 : index
    %1799 = subi %c-1, %1798 : index
    %1800 = select %1795, %1799, %1798 : index
    %1801 = muli %1800, %c-16 : index
    %1802 = addi %1536, %1801 : index
    %1803 = addi %1802, %c6 : index
    %1804 = load %2[%1803, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1805 = addf %1793, %1804 : vector<8xf32>
    store %1805, %1[%c0, %c12] : memref<1x16xvector<8xf32>>
    %1806 = addi %1529, %c104 : index
    %1807 = vector.transfer_read %arg2[%678, %1806], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1808 = addi %1176, %c104 : index
    %1809 = cmpi "slt", %1808, %c0 : index
    %1810 = subi %c-1, %1808 : index
    %1811 = select %1809, %1810, %1808 : index
    %1812 = divi_signed %1811, %c16 : index
    %1813 = subi %c-1, %1812 : index
    %1814 = select %1809, %1813, %1812 : index
    %1815 = remi_signed %1814, %c16 : index
    %1816 = cmpi "slt", %1815, %c0 : index
    %1817 = addi %1815, %c16 : index
    %1818 = select %1816, %1817, %1815 : index
    %1819 = muli %1814, %c-2 : index
    %1820 = addi %1572, %1819 : index
    %1821 = addi %1820, %c13 : index
    %1822 = cmpi "slt", %1821, %c0 : index
    %1823 = subi %c-1, %1821 : index
    %1824 = select %1822, %1823, %1821 : index
    %1825 = divi_signed %1824, %c2 : index
    %1826 = subi %c-1, %1825 : index
    %1827 = select %1822, %1826, %1825 : index
    %1828 = muli %1827, %c-2 : index
    %1829 = addi %1820, %1828 : index
    %1830 = addi %1829, %c13 : index
    %1831 = load %2[%1818, %c0, %1830] : memref<16x6x2xvector<8xf32>>
    %1832 = addf %1807, %1831 : vector<8xf32>
    store %1832, %1[%c0, %c13] : memref<1x16xvector<8xf32>>
    %1833 = addi %1529, %c112 : index
    %1834 = vector.transfer_read %arg2[%678, %1833], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1835 = addi %1536, %c7 : index
    %1836 = cmpi "slt", %1835, %c0 : index
    %1837 = subi %c-1, %1835 : index
    %1838 = select %1836, %1837, %1835 : index
    %1839 = divi_signed %1838, %c16 : index
    %1840 = subi %c-1, %1839 : index
    %1841 = select %1836, %1840, %1839 : index
    %1842 = muli %1841, %c-16 : index
    %1843 = addi %1536, %1842 : index
    %1844 = addi %1843, %c7 : index
    %1845 = load %2[%1844, %c0, %1554] : memref<16x6x2xvector<8xf32>>
    %1846 = addf %1834, %1845 : vector<8xf32>
    store %1846, %1[%c0, %c14] : memref<1x16xvector<8xf32>>
    %1847 = addi %1529, %c120 : index
    %1848 = vector.transfer_read %arg2[%678, %1847], %cst : memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, vector<8xf32>
    %1849 = addi %1176, %c120 : index
    %1850 = cmpi "slt", %1849, %c0 : index
    %1851 = subi %c-1, %1849 : index
    %1852 = select %1850, %1851, %1849 : index
    %1853 = divi_signed %1852, %c16 : index
    %1854 = subi %c-1, %1853 : index
    %1855 = select %1850, %1854, %1853 : index
    %1856 = remi_signed %1855, %c16 : index
    %1857 = cmpi "slt", %1856, %c0 : index
    %1858 = addi %1856, %c16 : index
    %1859 = select %1857, %1858, %1856 : index
    %1860 = muli %1855, %c-2 : index
    %1861 = addi %1572, %1860 : index
    %1862 = addi %1861, %c15 : index
    %1863 = cmpi "slt", %1862, %c0 : index
    %1864 = subi %c-1, %1862 : index
    %1865 = select %1863, %1864, %1862 : index
    %1866 = divi_signed %1865, %c2 : index
    %1867 = subi %c-1, %1866 : index
    %1868 = select %1863, %1867, %1866 : index
    %1869 = muli %1868, %c-2 : index
    %1870 = addi %1861, %1869 : index
    %1871 = addi %1870, %c15 : index
    %1872 = load %2[%1859, %c0, %1871] : memref<16x6x2xvector<8xf32>>
    %1873 = addf %1848, %1872 : vector<8xf32>
    store %1873, %1[%c0, %c15] : memref<1x16xvector<8xf32>>
    br ^bb48(%c0 : index)
  ^bb48(%1874: index):  // 2 preds: ^bb47, ^bb49
    %1875 = cmpi "slt", %1874, %c16 : index
    cond_br %1875, ^bb49, ^bb50
  ^bb49:  // pred: ^bb48
    %1876 = muli %1874, %c8 : index
    %1877 = addi %1529, %1876 : index
    %1878 = load %1[%c0, %1874] : memref<1x16xvector<8xf32>>
    vector.transfer_write %1878, %arg2[%678, %1877] : vector<8xf32>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>
    %1879 = addi %1874, %c1 : index
    br ^bb48(%1879 : index)
  ^bb50:  // pred: ^bb48
    br ^bb51
  ^bb51:  // 2 preds: ^bb46, ^bb50
    %1880 = addi %1176, %c128 : index
    br ^bb41(%1880 : index)
  ^bb52:  // pred: ^bb41
    %1881 = addi %678, %c1 : index
    br ^bb12(%1881 : index)
  ^bb53:  // pred: ^bb12
    %1882 = addi %4, %c256 : index
    br ^bb1(%1882 : index)
  ^bb54:  // pred: ^bb1
    return
  }
  func @optimized_matmul_py_4a6286d9(%arg0: memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, %arg1: memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, %arg2: memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) attributes {exec_target = 0 : i64, accv.base_name = "optimized_matmul_py", accv.emit_header_decl, accv.emit_raw_pointer_api} {
    call @optimized_matmul_py_4a6286d9_impl_17630232307017152746(%arg0, %arg1, %arg2) : (memref<784x128xf32, affine_map<(d0, d1) -> (d0 * 128 + d1)>>, memref<128x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>, memref<784x512xf32, affine_map<(d0, d1) -> (d0 * 512 + d1)>>) -> ()
    return
  }
}
