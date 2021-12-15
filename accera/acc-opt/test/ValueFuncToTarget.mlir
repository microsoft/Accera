// RUN: acc-opt --canonicalize --value-func-to-target -split-input-file %s | FileCheck %s

// CHECK-LABEL: module @test_cpu_module1
// CHECK-NEXT: accv.module "accera_test"
// CHECK-NEXT: func @foo
// CHECK-SAME: attributes {exec_target = 0 : i64}
// CHECK-NEXT return
module @test_cpu_module1 {
  "accv.module"() ( {
    "accv.func"() ( {
      accv.return
    }) {exec_target = 0 : i64, sym_name = "foo", type = () -> ()} : () -> ()
    "accv.module_terminator"() : () -> ()
  }) {sym_name = "accera_test"} : () -> ()
}

// -----

// CHECK-LABEL: module @test_gpu_module1
// CHECK-NEXT: accv.module "accera_test"
// CHECK-NEXT: func @foo
// CHECK-SAME: attributes {exec_target = 1 : i64}
// CHECK-NEXT return
module @test_gpu_module1 {
  "accv.module"() ( {
    "accv.func"() ( {
      accv.return
    }) {exec_target = 1 : i64, sym_name = "foo", type = () -> ()} : () -> ()
    "accv.module_terminator"() : () -> ()
  }) {sym_name = "accera_test"} : () -> ()
}
