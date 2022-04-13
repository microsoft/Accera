set -e

ACCERA_DIR=../../../..

# Generate the MLIR
rm -rf build/_tmp
python barrier_opt_test_generator.py

# Get the list of test names
tests=`python barrier_opt_test_generator.py list`

echo "// RUN: acc-opt --verify-each=false --optimize-barriers %s | FileCheck %s" > all_barrier_opt_tests.mlir
echo "" >> all_barrier_opt_tests.mlir
for testname in ${tests}; do
    echo "Running acc-opt on ${testname}_pre.mlir"
    (
        ${ACCERA_DIR}/build_cmake/bin/acc-opt --verify-each=false --optimize-barriers build/_tmp/${testname}/11_BarrierOpt.mlir > ${testname}_post.mlir
        # ${ACCERA_DIR}/build_cmake/bin/acc-opt --verify-each=false --acc-to-llvm="runtime=vulkan target=host enable-profiling=false gpu-only=false barrier-opt-dot barrier-opt-dot-filename=${testname}_graph.dot" --print-ir-after=optimize-barriers --mlir-disable-threading -o /dev/null ${testname}.mlir > /dev/null >& ${testname}_post.mlir
    )

    echo "// CHECK-LABEL: module @${testname}" >> all_barrier_opt_tests.mlir
    tail -n +2 ${testname}_post.mlir | sed 's/^[ \t]*/\/\/ CHECK-NEXT: /'>> all_barrier_opt_tests.mlir
    cat ${testname}_post.mlir >> all_barrier_opt_tests.mlir
    echo "\n" >> all_barrier_opt_tests.mlir
done

rm -rf build
rm -rf barrier_*.mlir
rm -rf *.dot
