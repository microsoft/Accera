// RUN: acc-opt --convert-value-to-std %s | FileCheck %s

// CHECK-LABEL: module @test_cast_module
module @test_cast_module {
  accv.module "test_cast_module" {

    //
    // Integer to float
    //

    // Signless to float
    // CHECK-LABEL: func @test_i32_to_f32(%arg0: memref<1xi32>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = arith.sitofp %0 : i32 to f32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf32>
    builtin.func @test_i32_to_f32(%arg0: memref<1xi32>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    // Signed to float
    // CHECK-LABEL: func @test_si32_to_f32(%arg0: memref<1xsi32>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    %2 = arith.sitofp %1 : i32 to f32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xf32>
    builtin.func @test_si32_to_f32(%arg0: memref<1xsi32>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    // Unsigned to float 

    // CHECK-LABEL: func @test_ui32_to_f32(%arg0: memref<1xui32>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    %2 = arith.uitofp %1 : i32 to f32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xf32>
    builtin.func @test_ui32_to_f32(%arg0: memref<1xui32>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    // Index to float
    // CHECK-LABEL: func @test_index_to_f32(%arg0: memref<1xindex>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xindex>
    // CHECK-NEXT:    %1 = arith.index_cast %0 : index to i64
    // CHECK-NEXT:    %2 = arith.sitofp %1 : i64 to f32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xf32>
    builtin.func @test_index_to_f32(%arg0: memref<1xindex>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xindex>
      %1 = "accv.cast"(%0) : (index) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    //
    // Float to integer
    //

    // Float to signless
    // CHECK-LABEL: func @test_f32_to_i32(%arg0: memref<1xf32>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = arith.fptosi %0 : f32 to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_f32_to_i32(%arg0: memref<1xf32>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // Float to signed
    // CHECK-LABEL: func @test_f32_to_si32(%arg0: memref<1xf32>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = arith.fptosi %0 : f32 to i32
    // CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : i32 to si32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xsi32>
    builtin.func @test_f32_to_si32(%arg0: memref<1xf32>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }

    // Float to unsigned
    // CHECK-LABEL: func @test_f32_to_ui32(%arg0: memref<1xf32>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = arith.fptoui %0 : f32 to i32
    // CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : i32 to ui32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xui32>
    builtin.func @test_f32_to_ui32(%arg0: memref<1xf32>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }

    // Float to index
    // CHECK-LABEL: func @test_f32_to_index(%arg0: memref<1xf32>, %arg1: memref<1xindex>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = arith.fptosi %0 : f32 to i64
    // CHECK-NEXT:    %2 = arith.index_cast %1 : i64 to index
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xindex>
    builtin.func @test_f32_to_index(%arg0: memref<1xf32>, %arg1: memref<1xindex>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> index
      affine.store %1, %arg1[0] : memref<1xindex>
      return
    }

    //
    // Integers to integer
    //

    // Signless to signed
    // CHECK-LABEL: func @test_i32_to_si32(%arg0: memref<1xi32>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : i32 to si32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xsi32>
    builtin.func @test_i32_to_si32(%arg0: memref<1xi32>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }

    // Signless to unsigned
    // CHECK-LABEL: func @test_i32_to_ui32(%arg0: memref<1xi32>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : i32 to ui32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xui32>
    builtin.func @test_i32_to_ui32(%arg0: memref<1xi32>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }

    // Signless to index
    // CHECK-LABEL: func @test_i32_to_index(%arg0: memref<1xi32>, %arg1: memref<1xindex>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = arith.index_cast %0 : i32 to index
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xindex>
    builtin.func @test_i32_to_index(%arg0: memref<1xi32>, %arg1: memref<1xindex>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> index
      affine.store %1, %arg1[0] : memref<1xindex>
      return
    }

    // Signed to signless
    // CHECK-LABEL: func @test_si32_to_i32(%arg0: memref<1xsi32>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_si32_to_i32(%arg0: memref<1xsi32>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // Signed to unsigned
    // CHECK-LABEL: func @test_si32_to_ui32(%arg0: memref<1xsi32>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : i32 to ui32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xui32>
    builtin.func @test_si32_to_ui32(%arg0: memref<1xsi32>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }

    // Signed to index
    // CHECK-LABEL: func @test_si32_to_index(%arg0: memref<1xsi32>, %arg1: memref<1xindex>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    %2 = arith.index_cast %1 : i32 to index
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xindex>
    builtin.func @test_si32_to_index(%arg0: memref<1xsi32>, %arg1: memref<1xindex>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> index
      affine.store %1, %arg1[0] : memref<1xindex>
      return
    }

    // Unsigned to signless
    // CHECK-LABEL: func @test_ui32_to_i32(%arg0: memref<1xui32>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_ui32_to_i32(%arg0: memref<1xui32>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // Unsigned to signed
    // CHECK-LABEL: func @test_ui32_to_si32(%arg0: memref<1xui32>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : i32 to si32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xsi32>
    builtin.func @test_ui32_to_si32(%arg0: memref<1xui32>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }

    // Unsigned to index
    // CHECK-LABEL: func @test_ui32_to_index(%arg0: memref<1xui32>, %arg1: memref<1xindex>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    %2 = arith.index_cast %1 : i32 to index
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xindex>
    builtin.func @test_ui32_to_index(%arg0: memref<1xui32>, %arg1: memref<1xindex>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> index
      affine.store %1, %arg1[0] : memref<1xindex>
      return
    }

    // Index to signed
    // CHECK-LABEL: func @test_index_to_si32(%arg0: memref<1xindex>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xindex>
    // CHECK-NEXT:    %1 = arith.index_cast %0 : index to i32
    // CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : i32 to si32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xsi32>
    builtin.func @test_index_to_si32(%arg0: memref<1xindex>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xindex>
      %1 = "accv.cast"(%0) : (index) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }

    // Index to signless
    // CHECK-LABEL: func @test_index_to_i32(%arg0: memref<1xindex>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xindex>
    // CHECK-NEXT:    %1 = arith.index_cast %0 : index to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_index_to_i32(%arg0: memref<1xindex>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xindex>
      %1 = "accv.cast"(%0) : (index) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // Index to unsigned
    // CHECK-LABEL: func @test_index_to_ui32(%arg0: memref<1xindex>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xindex>
    // CHECK-NEXT:    %1 = arith.index_cast %0 : index to i32
    // CHECK-NEXT:    %2 = builtin.unrealized_conversion_cast %1 : i32 to ui32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xui32>
    builtin.func @test_index_to_ui32(%arg0: memref<1xindex>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xindex>
      %1 = "accv.cast"(%0) : (index) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }


    //
    // Widen integers
    //

    // Signless integers
    // CHECK-LABEL: func @test_i8_to_i16(%arg0: memref<1xi8>, %arg1: memref<1xi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi8>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i8 to i16
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi16>
    builtin.func @test_i8_to_i16(%arg0: memref<1xi8>, %arg1: memref<1xi16>) {
      %0 = affine.load %arg0[0] : memref<1xi8>
      %1 = "accv.cast"(%0) : (i8) -> i16
      affine.store %1, %arg1[0] : memref<1xi16>
      return
    }

    // CHECK-LABEL: func @test_i8_to_i32(%arg0: memref<1xi8>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi8>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i8 to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_i8_to_i32(%arg0: memref<1xi8>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi8>
      %1 = "accv.cast"(%0) : (i8) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @test_i8_to_i64(%arg0: memref<1xi8>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi8>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i8 to i64
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi64>
    builtin.func @test_i8_to_i64(%arg0: memref<1xi8>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi8>
      %1 = "accv.cast"(%0) : (i8) -> i64
      affine.store %1, %arg1[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @test_i16_to_i32(%arg0: memref<1xi16>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi16>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i16 to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_i16_to_i32(%arg0: memref<1xi16>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi16>
      %1 = "accv.cast"(%0) : (i16) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @test_i16_to_i64(%arg0: memref<1xi16>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi16>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i16 to i64
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi64>
    builtin.func @test_i16_to_i64(%arg0: memref<1xi16>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi16>
      %1 = "accv.cast"(%0) : (i16) -> i64
      affine.store %1, %arg1[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @test_i32_to_i64(%arg0: memref<1xi32>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i32 to i64
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi64>
    builtin.func @test_i32_to_i64(%arg0: memref<1xi32>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> i64
      affine.store %1, %arg1[0] : memref<1xi64>
      return
    }

    // Signed integers

    // CHECK-LABEL: func @test_si8_to_si16(%arg0: memref<1xsi8>, %arg1: memref<1xsi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi8>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si8 to i8
    // CHECK-NEXT:    %2 = arith.extsi %1 : i8 to i16
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i16 to si16
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi16>
    builtin.func @test_si8_to_si16(%arg0: memref<1xsi8>, %arg1: memref<1xsi16>) {
      %0 = affine.load %arg0[0] : memref<1xsi8>
      %1 = "accv.cast"(%0) : (si8) -> si16
      affine.store %1, %arg1[0] : memref<1xsi16>
      return
    }
    
    // CHECK-LABEL: func @test_si8_to_si32(%arg0: memref<1xsi8>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi8>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si8 to i8
    // CHECK-NEXT:    %2 = arith.extsi %1 : i8 to i32
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i32 to si32
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi32>
    builtin.func @test_si8_to_si32(%arg0: memref<1xsi8>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xsi8>
      %1 = "accv.cast"(%0) : (si8) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }
    
    // CHECK-LABEL: func @test_si8_to_si64(%arg0: memref<1xsi8>, %arg1: memref<1xsi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi8>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si8 to i8
    // CHECK-NEXT:    %2 = arith.extsi %1 : i8 to i64
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i64 to si64
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi64>
    builtin.func @test_si8_to_si64(%arg0: memref<1xsi8>, %arg1: memref<1xsi64>) {
      %0 = affine.load %arg0[0] : memref<1xsi8>
      %1 = "accv.cast"(%0) : (si8) -> si64
      affine.store %1, %arg1[0] : memref<1xsi64>
      return
    }
    
    // CHECK-LABEL: func @test_si16_to_si32(%arg0: memref<1xsi16>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi16>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si16 to i16
    // CHECK-NEXT:    %2 = arith.extsi %1 : i16 to i32
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i32 to si32
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi32>
    builtin.func @test_si16_to_si32(%arg0: memref<1xsi16>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xsi16>
      %1 = "accv.cast"(%0) : (si16) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }
    
    // CHECK-LABEL: func @test_si16_to_si64(%arg0: memref<1xsi16>, %arg1: memref<1xsi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi16>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si16 to i16
    // CHECK-NEXT:    %2 = arith.extsi %1 : i16 to i64
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i64 to si64
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi64>
    builtin.func @test_si16_to_si64(%arg0: memref<1xsi16>, %arg1: memref<1xsi64>) {
      %0 = affine.load %arg0[0] : memref<1xsi16>
      %1 = "accv.cast"(%0) : (si16) -> si64
      affine.store %1, %arg1[0] : memref<1xsi64>
      return
    }
    
    // CHECK-LABEL: func @test_si32_to_si64(%arg0: memref<1xsi32>, %arg1: memref<1xsi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    %2 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i64 to si64
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi64>
    builtin.func @test_si32_to_si64(%arg0: memref<1xsi32>, %arg1: memref<1xsi64>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> si64
      affine.store %1, %arg1[0] : memref<1xsi64>
      return
    }

    // Unsigned integers
    
    // CHECK-LABEL: func @test_ui8_to_ui16(%arg0: memref<1xui8>, %arg1: memref<1xui16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui8>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui8 to i8
    // CHECK-NEXT:    %2 = arith.extui %1 : i8 to i16
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i16 to ui16
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui16>
    builtin.func @test_ui8_to_ui16(%arg0: memref<1xui8>, %arg1: memref<1xui16>) {
      %0 = affine.load %arg0[0] : memref<1xui8>
      %1 = "accv.cast"(%0) : (ui8) -> ui16
      affine.store %1, %arg1[0] : memref<1xui16>
      return
    }
    
    // CHECK-LABEL: func @test_ui8_to_ui32(%arg0: memref<1xui8>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui8>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui8 to i8
    // CHECK-NEXT:    %2 = arith.extui %1 : i8 to i32
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i32 to ui32
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui32>
    builtin.func @test_ui8_to_ui32(%arg0: memref<1xui8>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xui8>
      %1 = "accv.cast"(%0) : (ui8) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }
    
    // CHECK-LABEL: func @test_ui8_to_ui64(%arg0: memref<1xui8>, %arg1: memref<1xui64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui8>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui8 to i8
    // CHECK-NEXT:    %2 = arith.extui %1 : i8 to i64
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i64 to ui64
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui64>
    builtin.func @test_ui8_to_ui64(%arg0: memref<1xui8>, %arg1: memref<1xui64>) {
      %0 = affine.load %arg0[0] : memref<1xui8>
      %1 = "accv.cast"(%0) : (ui8) -> ui64
      affine.store %1, %arg1[0] : memref<1xui64>
      return
    }
    
    // CHECK-LABEL: func @test_ui16_to_ui32(%arg0: memref<1xui16>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui16>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui16 to i16
    // CHECK-NEXT:    %2 = arith.extui %1 : i16 to i32
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i32 to ui32
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui32>
    builtin.func @test_ui16_to_ui32(%arg0: memref<1xui16>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xui16>
      %1 = "accv.cast"(%0) : (ui16) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }
    
    // CHECK-LABEL: func @test_ui16_to_ui64(%arg0: memref<1xui16>, %arg1: memref<1xui64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui16>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui16 to i16
    // CHECK-NEXT:    %2 = arith.extui %1 : i16 to i64
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i64 to ui64
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui64>
    builtin.func @test_ui16_to_ui64(%arg0: memref<1xui16>, %arg1: memref<1xui64>) {
      %0 = affine.load %arg0[0] : memref<1xui16>
      %1 = "accv.cast"(%0) : (ui16) -> ui64
      affine.store %1, %arg1[0] : memref<1xui64>
      return
    }
    
    // CHECK-LABEL: func @test_ui32_to_ui64(%arg0: memref<1xui32>, %arg1: memref<1xui64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    %2 = arith.extui %1 : i32 to i64
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i64 to ui64
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui64>
    builtin.func @test_ui32_to_ui64(%arg0: memref<1xui32>, %arg1: memref<1xui64>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> ui64
      affine.store %1, %arg1[0] : memref<1xui64>
      return
    }

    //
    // Narrow integers
    //

    // Signless integers
    // CHECK-LABEL: func @test_i16_to_i8(%arg0: memref<1xi16>, %arg1: memref<1xi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi16>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i16 to i8
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi8>
    builtin.func @test_i16_to_i8(%arg0: memref<1xi16>, %arg1: memref<1xi8>) {
      %0 = affine.load %arg0[0] : memref<1xi16>
      %1 = "accv.cast"(%0) : (i16) -> i8
      affine.store %1, %arg1[0] : memref<1xi8>
      return
    }

    // CHECK-LABEL: func @test_i32_to_i8(%arg0: memref<1xi32>, %arg1: memref<1xi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i32 to i8
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi8>
    builtin.func @test_i32_to_i8(%arg0: memref<1xi32>, %arg1: memref<1xi8>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> i8
      affine.store %1, %arg1[0] : memref<1xi8>
      return
    }

    // CHECK-LABEL: func @test_i64_to_i8(%arg0: memref<1xi64>, %arg1: memref<1xi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i64 to i8
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi8>
    builtin.func @test_i64_to_i8(%arg0: memref<1xi64>, %arg1: memref<1xi8>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) : (i64) -> i8
      affine.store %1, %arg1[0] : memref<1xi8>
      return
    }

    // CHECK-LABEL: func @test_i32_to_i16(%arg0: memref<1xi32>, %arg1: memref<1xi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i32 to i16
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi16>
    builtin.func @test_i32_to_i16(%arg0: memref<1xi32>, %arg1: memref<1xi16>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> i16
      affine.store %1, %arg1[0] : memref<1xi16>
      return
    }

    // CHECK-LABEL: func @test_i64_to_i16(%arg0: memref<1xi64>, %arg1: memref<1xi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i64 to i16
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi16>
    builtin.func @test_i64_to_i16(%arg0: memref<1xi64>, %arg1: memref<1xi16>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) : (i64) -> i16
      affine.store %1, %arg1[0] : memref<1xi16>
      return
    }

    // CHECK-LABEL: func @test_i64_to_i32(%arg0: memref<1xi64>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i64 to i32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi32>
    builtin.func @test_i64_to_i32(%arg0: memref<1xi64>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) : (i64) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // Signed integers

    // CHECK-LABEL: func @test_si16_to_si8(%arg0: memref<1xsi16>, %arg1: memref<1xsi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi16>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si16 to i16
    // CHECK-NEXT:    %2 = arith.trunci %1 : i16 to i8
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i8 to si8
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi8>
    builtin.func @test_si16_to_si8(%arg0: memref<1xsi16>, %arg1: memref<1xsi8>) {
      %0 = affine.load %arg0[0] : memref<1xsi16>
      %1 = "accv.cast"(%0) : (si16) -> si8
      affine.store %1, %arg1[0] : memref<1xsi8>
      return
    }
    
    // CHECK-LABEL: func @test_si32_to_si8(%arg0: memref<1xsi32>, %arg1: memref<1xsi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    %2 = arith.trunci %1 : i32 to i8
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i8 to si8
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi8>
    builtin.func @test_si32_to_si8(%arg0: memref<1xsi32>, %arg1: memref<1xsi8>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> si8
      affine.store %1, %arg1[0] : memref<1xsi8>
      return
    }
    
    // CHECK-LABEL: func @test_si64_to_si8(%arg0: memref<1xsi64>, %arg1: memref<1xsi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi64>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si64 to i64
    // CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i8
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i8 to si8
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi8>
    builtin.func @test_si64_to_si8(%arg0: memref<1xsi64>, %arg1: memref<1xsi8>) {
      %0 = affine.load %arg0[0] : memref<1xsi64>
      %1 = "accv.cast"(%0) : (si64) -> si8
      affine.store %1, %arg1[0] : memref<1xsi8>
      return
    }
    
    // CHECK-LABEL: func @test_si32_to_si16(%arg0: memref<1xsi32>, %arg1: memref<1xsi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si32 to i32
    // CHECK-NEXT:    %2 = arith.trunci %1 : i32 to i16
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i16 to si16
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi16>
    builtin.func @test_si32_to_si16(%arg0: memref<1xsi32>, %arg1: memref<1xsi16>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> si16
      affine.store %1, %arg1[0] : memref<1xsi16>
      return
    }
    
    // CHECK-LABEL: func @test_si64_to_si16(%arg0: memref<1xsi64>, %arg1: memref<1xsi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi64>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si64 to i64
    // CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i16
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i16 to si16
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi16>
    builtin.func @test_si64_to_si16(%arg0: memref<1xsi64>, %arg1: memref<1xsi16>) {
      %0 = affine.load %arg0[0] : memref<1xsi64>
      %1 = "accv.cast"(%0) : (si64) -> si16
      affine.store %1, %arg1[0] : memref<1xsi16>
      return
    }
    
    // CHECK-LABEL: func @test_si64_to_si32(%arg0: memref<1xsi64>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi64>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : si64 to i64
    // CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i32
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i32 to si32
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xsi32>
    builtin.func @test_si64_to_si32(%arg0: memref<1xsi64>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xsi64>
      %1 = "accv.cast"(%0) : (si64) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }

    // Unsigned integers
    
    // CHECK-LABEL: func @test_ui16_to_ui8(%arg0: memref<1xui16>, %arg1: memref<1xui8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui16>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui16 to i16
    // CHECK-NEXT:    %2 = arith.trunci %1 : i16 to i8
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i8 to ui8
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui8>
    builtin.func @test_ui16_to_ui8(%arg0: memref<1xui16>, %arg1: memref<1xui8>) {
      %0 = affine.load %arg0[0] : memref<1xui16>
      %1 = "accv.cast"(%0) : (ui16) -> ui8
      affine.store %1, %arg1[0] : memref<1xui8>
      return
    }
    
    // CHECK-LABEL: func @test_ui32_to_ui8(%arg0: memref<1xui32>, %arg1: memref<1xui8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    %2 = arith.trunci %1 : i32 to i8
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i8 to ui8
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui8>
    builtin.func @test_ui32_to_ui8(%arg0: memref<1xui32>, %arg1: memref<1xui8>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> ui8
      affine.store %1, %arg1[0] : memref<1xui8>
      return
    }
    
    // CHECK-LABEL: func @test_ui64_to_ui8(%arg0: memref<1xui64>, %arg1: memref<1xui8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui64>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui64 to i64
    // CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i8
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i8 to ui8
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui8>
    builtin.func @test_ui64_to_ui8(%arg0: memref<1xui64>, %arg1: memref<1xui8>) {
      %0 = affine.load %arg0[0] : memref<1xui64>
      %1 = "accv.cast"(%0) : (ui64) -> ui8
      affine.store %1, %arg1[0] : memref<1xui8>
      return
    }
    
    // CHECK-LABEL: func @test_ui32_to_ui16(%arg0: memref<1xui32>, %arg1: memref<1xui16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui32 to i32
    // CHECK-NEXT:    %2 = arith.trunci %1 : i32 to i16
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i16 to ui16
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui16>
    builtin.func @test_ui32_to_ui16(%arg0: memref<1xui32>, %arg1: memref<1xui16>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> ui16
      affine.store %1, %arg1[0] : memref<1xui16>
      return
    }
    
    // CHECK-LABEL: func @test_ui64_to_ui16(%arg0: memref<1xui64>, %arg1: memref<1xui16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui64>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui64 to i64
    // CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i16
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i16 to ui16
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui16>
    builtin.func @test_ui64_to_ui16(%arg0: memref<1xui64>, %arg1: memref<1xui16>) {
      %0 = affine.load %arg0[0] : memref<1xui64>
      %1 = "accv.cast"(%0) : (ui64) -> ui16
      affine.store %1, %arg1[0] : memref<1xui16>
      return
    }
    
    // CHECK-LABEL: func @test_ui64_to_ui32(%arg0: memref<1xui64>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui64>
    // CHECK-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : ui64 to i64
    // CHECK-NEXT:    %2 = arith.trunci %1 : i64 to i32
    // CHECK-NEXT:    %3 = builtin.unrealized_conversion_cast %2 : i32 to ui32
    // CHECK-NEXT:    affine.store %3, %arg1[0] : memref<1xui32>
    builtin.func @test_ui64_to_ui32(%arg0: memref<1xui64>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xui64>
      %1 = "accv.cast"(%0) : (ui64) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }

    //
    // Widen floats
    //

    // CHECK-LABEL: func @test_f16_to_f32(%arg0: memref<1xf16>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf16>
    // CHECK-NEXT:    %1 = arith.extf %0 : f16 to f32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf32>
    builtin.func @test_f16_to_f32(%arg0: memref<1xf16>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xf16>
      %1 = "accv.cast"(%0) : (f16) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    // CHECK-LABEL: func @test_f16_to_f64(%arg0: memref<1xf16>, %arg1: memref<1xf64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf16>
    // CHECK-NEXT:    %1 = arith.extf %0 : f16 to f64
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf64>
    builtin.func @test_f16_to_f64(%arg0: memref<1xf16>, %arg1: memref<1xf64>) {
      %0 = affine.load %arg0[0] : memref<1xf16>
      %1 = "accv.cast"(%0) : (f16) -> f64
      affine.store %1, %arg1[0] : memref<1xf64>
      return
    }

    // CHECK-LABEL: func @test_f32_to_f64(%arg0: memref<1xf32>, %arg1: memref<1xf64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = arith.extf %0 : f32 to f64
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf64>
    builtin.func @test_f32_to_f64(%arg0: memref<1xf32>, %arg1: memref<1xf64>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> f64
      affine.store %1, %arg1[0] : memref<1xf64>
      return
    }

    //
    // Narrow floats
    //

    // CHECK-LABEL: func @test_f32_to_f16(%arg0: memref<1xf32>, %arg1: memref<1xf16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    %1 = arith.truncf %0 : f32 to f16
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf16>
    builtin.func @test_f32_to_f16(%arg0: memref<1xf32>, %arg1: memref<1xf16>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> f16
      affine.store %1, %arg1[0] : memref<1xf16>
      return
    }

    // CHECK-LABEL: func @test_f64_to_f16(%arg0: memref<1xf64>, %arg1: memref<1xf16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf64>
    // CHECK-NEXT:    %1 = arith.truncf %0 : f64 to f16
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf16>
    builtin.func @test_f64_to_f16(%arg0: memref<1xf64>, %arg1: memref<1xf16>) {
      %0 = affine.load %arg0[0] : memref<1xf64>
      %1 = "accv.cast"(%0) : (f64) -> f16
      affine.store %1, %arg1[0] : memref<1xf16>
      return
    }

    // CHECK-LABEL: func @test_f64_to_f32(%arg0: memref<1xf64>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf64>
    // CHECK-NEXT:    %1 = arith.truncf %0 : f64 to f32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf32>
    builtin.func @test_f64_to_f32(%arg0: memref<1xf64>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xf64>
      %1 = "accv.cast"(%0) : (f64) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    //
    // No-op casts
    //

    // CHECK-LABEL: func @test_i8_to_i8(%arg0: memref<1xi8>, %arg1: memref<1xi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi8>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xi8>
    builtin.func @test_i8_to_i8(%arg0: memref<1xi8>, %arg1: memref<1xi8>) {
      %0 = affine.load %arg0[0] : memref<1xi8>
      %1 = "accv.cast"(%0) : (i8) -> i8
      affine.store %1, %arg1[0] : memref<1xi8>
      return
    }

    // CHECK-LABEL: func @test_i16_to_i16(%arg0: memref<1xi16>, %arg1: memref<1xi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi16>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xi16>
    builtin.func @test_i16_to_i16(%arg0: memref<1xi16>, %arg1: memref<1xi16>) {
      %0 = affine.load %arg0[0] : memref<1xi16>
      %1 = "accv.cast"(%0) : (i16) -> i16
      affine.store %1, %arg1[0] : memref<1xi16>
      return
    }

    // CHECK-LABEL: func @test_i32_to_i32(%arg0: memref<1xi32>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi32>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xi32>
    builtin.func @test_i32_to_i32(%arg0: memref<1xi32>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xi32>
      %1 = "accv.cast"(%0) : (i32) -> i32
      affine.store %1, %arg1[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @test_i64_to_i64(%arg0: memref<1xi64>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xi64>
    builtin.func @test_i64_to_i64(%arg0: memref<1xi64>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) : (i64) -> i64
      affine.store %1, %arg1[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @test_si8_to_si8(%arg0: memref<1xsi8>, %arg1: memref<1xsi8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi8>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xsi8>
    builtin.func @test_si8_to_si8(%arg0: memref<1xsi8>, %arg1: memref<1xsi8>) {
      %0 = affine.load %arg0[0] : memref<1xsi8>
      %1 = "accv.cast"(%0) : (si8) -> si8
      affine.store %1, %arg1[0] : memref<1xsi8>
      return
    }

    // CHECK-LABEL: func @test_si16_to_si16(%arg0: memref<1xsi16>, %arg1: memref<1xsi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi16>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xsi16>
    builtin.func @test_si16_to_si16(%arg0: memref<1xsi16>, %arg1: memref<1xsi16>) {
      %0 = affine.load %arg0[0] : memref<1xsi16>
      %1 = "accv.cast"(%0) : (si16) -> si16
      affine.store %1, %arg1[0] : memref<1xsi16>
      return
    }

    // CHECK-LABEL: func @test_si32_to_si32(%arg0: memref<1xsi32>, %arg1: memref<1xsi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi32>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xsi32>
    builtin.func @test_si32_to_si32(%arg0: memref<1xsi32>, %arg1: memref<1xsi32>) {
      %0 = affine.load %arg0[0] : memref<1xsi32>
      %1 = "accv.cast"(%0) : (si32) -> si32
      affine.store %1, %arg1[0] : memref<1xsi32>
      return
    }

    // CHECK-LABEL: func @test_si64_to_si64(%arg0: memref<1xsi64>, %arg1: memref<1xsi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xsi64>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xsi64>
    builtin.func @test_si64_to_si64(%arg0: memref<1xsi64>, %arg1: memref<1xsi64>) {
      %0 = affine.load %arg0[0] : memref<1xsi64>
      %1 = "accv.cast"(%0) : (si64) -> si64
      affine.store %1, %arg1[0] : memref<1xsi64>
      return
    }

    // CHECK-LABEL: func @test_ui8_to_ui8(%arg0: memref<1xui8>, %arg1: memref<1xui8>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui8>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xui8>
    builtin.func @test_ui8_to_ui8(%arg0: memref<1xui8>, %arg1: memref<1xui8>) {
      %0 = affine.load %arg0[0] : memref<1xui8>
      %1 = "accv.cast"(%0) : (ui8) -> ui8
      affine.store %1, %arg1[0] : memref<1xui8>
      return
    }

    // CHECK-LABEL: func @test_ui16_to_ui16(%arg0: memref<1xui16>, %arg1: memref<1xui16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui16>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xui16>
    builtin.func @test_ui16_to_ui16(%arg0: memref<1xui16>, %arg1: memref<1xui16>) {
      %0 = affine.load %arg0[0] : memref<1xui16>
      %1 = "accv.cast"(%0) : (ui16) -> ui16
      affine.store %1, %arg1[0] : memref<1xui16>
      return
    }

    // CHECK-LABEL: func @test_ui32_to_ui32(%arg0: memref<1xui32>, %arg1: memref<1xui32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui32>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xui32>
    builtin.func @test_ui32_to_ui32(%arg0: memref<1xui32>, %arg1: memref<1xui32>) {
      %0 = affine.load %arg0[0] : memref<1xui32>
      %1 = "accv.cast"(%0) : (ui32) -> ui32
      affine.store %1, %arg1[0] : memref<1xui32>
      return
    }

    // CHECK-LABEL: func @test_ui64_to_ui64(%arg0: memref<1xui64>, %arg1: memref<1xui64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xui64>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xui64>
    builtin.func @test_ui64_to_ui64(%arg0: memref<1xui64>, %arg1: memref<1xui64>) {
      %0 = affine.load %arg0[0] : memref<1xui64>
      %1 = "accv.cast"(%0) : (ui64) -> ui64
      affine.store %1, %arg1[0] : memref<1xui64>
      return
    }

    // CHECK-LABEL: func @test_index_to_index(%arg0: memref<1xindex>, %arg1: memref<1xindex>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xindex>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xindex>
    builtin.func @test_index_to_index(%arg0: memref<1xindex>, %arg1: memref<1xindex>) {
      %0 = affine.load %arg0[0] : memref<1xindex>
      %1 = "accv.cast"(%0) : (index) -> index
      affine.store %1, %arg1[0] : memref<1xindex>
      return
    }

    // CHECK-LABEL: func @test_f16_to_f16(%arg0: memref<1xf16>, %arg1: memref<1xf16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf16>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xf16>
    builtin.func @test_f16_to_f16(%arg0: memref<1xf16>, %arg1: memref<1xf16>) {
      %0 = affine.load %arg0[0] : memref<1xf16>
      %1 = "accv.cast"(%0) : (f16) -> f16
      affine.store %1, %arg1[0] : memref<1xf16>
      return
    }

    // CHECK-LABEL: func @test_f32_to_f32(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf32>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xf32>
    builtin.func @test_f32_to_f32(%arg0: memref<1xf32>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xf32>
      %1 = "accv.cast"(%0) : (f32) -> f32
      affine.store %1, %arg1[0] : memref<1xf32>
      return
    }

    // CHECK-LABEL: func @test_f64_to_f64(%arg0: memref<1xf64>, %arg1: memref<1xf64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf64>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xf64>
    builtin.func @test_f64_to_f64(%arg0: memref<1xf64>, %arg1: memref<1xf64>) {
      %0 = affine.load %arg0[0] : memref<1xf64>
      %1 = "accv.cast"(%0) : (f64) -> f64
      affine.store %1, %arg1[0] : memref<1xf64>
      return
    }

    // CHECK-LABEL: func @dont_fold_redundant_cast(%arg0: memref<1xi64>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i64 to i32
    // CHECK-NEXT:    %2 = arith.extsi %1 : i32 to i64
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xi64>
    builtin.func @dont_fold_redundant_cast(%arg0: memref<1xi64>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) : (i64) -> i32
      %2 = "accv.cast"(%1) : (i32) -> i64
      affine.store %2, %arg1[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @fold_redundant_internal_cast(%arg0: memref<1xi64>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    affine.store %0, %arg1[0] : memref<1xi64>
    builtin.func @fold_redundant_internal_cast(%arg0: memref<1xi64>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) {internal} : (i64) -> i32
      %2 = "accv.cast"(%1) {internal} : (i32) -> i64
      affine.store %2, %arg1[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @fold_cast_internal_cast(%arg0: memref<1xi16>, %arg1: memref<1xf32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi16>
    // CHECK-NEXT:    %1 = arith.sitofp %0 : i16 to f32
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xf32>
    builtin.func @fold_cast_internal_cast(%arg0: memref<1xi16>, %arg1: memref<1xf32>) {
      %0 = affine.load %arg0[0] : memref<1xi16>
      %1 = "accv.cast"(%0) : (i16) -> i32
      %2 = "accv.cast"(%1) {internal} : (i32) -> f32
      affine.store %2, %arg1[0] : memref<1xf32>
      return
    }

    // CHECK-LABEL: func @dont_fold_cast_internal_cast(%arg0: memref<1xf16>, %arg1: memref<1xi32>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xf16>
    // CHECK-NEXT:    %1 = arith.extf %0 : f16 to f32
    // CHECK-NEXT:    %2 = arith.fptosi %1 : f32 to i32
    // CHECK-NEXT:    affine.store %2, %arg1[0] : memref<1xi32>
    builtin.func @dont_fold_cast_internal_cast(%arg0: memref<1xf16>, %arg1: memref<1xi32>) {
      %0 = affine.load %arg0[0] : memref<1xf16>
      %1 = "accv.cast"(%0) : (f16) -> f32
      %2 = "accv.cast"(%1) {internal} : (f32) -> i32
      affine.store %2, %arg1[0] : memref<1xi32>
      return
    }

    // CHECK-LABEL: func @fold_widening_cast(%arg0: memref<1xi16>, %arg1: memref<1xi64>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi16>
    // CHECK-NEXT:    %1 = arith.extsi %0 : i16 to i64
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi64>
    builtin.func @fold_widening_cast(%arg0: memref<1xi16>, %arg1: memref<1xi64>) {
      %0 = affine.load %arg0[0] : memref<1xi16>
      %1 = "accv.cast"(%0) : (i16) -> i32
      %2 = "accv.cast"(%1) : (i32) -> i64
      affine.store %2, %arg1[0] : memref<1xi64>
      return
    }

    // CHECK-LABEL: func @fold_narrowing_cast(%arg0: memref<1xi64>, %arg1: memref<1xi16>) {
    // CHECK-NEXT:    %0 = affine.load %arg0[0] : memref<1xi64>
    // CHECK-NEXT:    %1 = arith.trunci %0 : i64 to i16
    // CHECK-NEXT:    affine.store %1, %arg1[0] : memref<1xi16>
    builtin.func @fold_narrowing_cast(%arg0: memref<1xi64>, %arg1: memref<1xi16>) {
      %0 = affine.load %arg0[0] : memref<1xi64>
      %1 = "accv.cast"(%0) : (i64) -> i32
      %2 = "accv.cast"(%1) : (i32) -> i16
      affine.store %2, %arg1[0] : memref<1xi16>
      return
    }
  }
}
