// matrix size = 32x32
// block size = 4
// uses subview , vector.transfer_read, and linalg.copy
module @gpu_module3 attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 32 : i32, max_compute_workgroup_size = dense<[4, 4, 1]> : vector<3xi32>}>} {
  gpu.module @kernels {
    gpu.func @dot(%a: memref<32x32xf32>,  %b: memref<32x32xf32>, %c: memref<32x32xf32>)
      // does not work to specify the workgroup shared memory 
      // requirements in the function using
      // workgroup(%aSh: memref<4x4xf32, 3>, %bSh: memref<4x4xf32, 3>)
      kernel attributes {spv.entry_point_abi = {local_size = dense<[4, 4, 1]> : vector<3xi32>}} {
      %aSh = alloc() : memref<4x4xf32,3>
      %bSh = alloc() : memref<4x4xf32,3>
      %blkidx = gpu.block_id x
      %blkidy = gpu.block_id y
      %tidx = gpu.thread_id x
      %tidy = gpu.thread_id y
      %bdimx = arith.constant 4 : index
      %bdimy = arith.constant 4 : index

      %0 = arith.muli %blkidx, %bdimx: index
      %i = addi %0, %tidx: index // gidx = threadIdx.x + 4*blockIdx.x
      %1 = arith.muli %blkidy, %bdimy: index
      %j = addi %1, %tidy: index // gidy = threadIdx.y + 4*blockIdx.y


      %accum_0 = arith.constant 0.0 : f32
      %lb = arith.constant 0 : index
      %ub = arith.constant 4 : index
      %niters = arith.constant 8 : index // 32 / 4
      %step = arith.constant 1 : index

      %accum = scf.for %m = %lb to %niters step %step iter_args(%accum_iter = %accum_0) -> (f32) {

        %mx0 = arith.muli %m, %bdimx: index
        %mx = addi %mx0, %tidx: index // m * TILE_SIZE + tx
        %my0 = arith.muli %m, %bdimy: index
        %my = addi %my0, %tidy: index // m * TILE_SIZE + ty

        %ai = load %a[%i, %mx] : memref<32x32xf32>
        %bi = load %b[%my, %j] : memref<32x32xf32>

        store %ai, %aSh[%tidy, %tidx] : memref<4x4xf32, 3>
        store %bi, %bSh[%tidy, %tidx] : memref<4x4xf32, 3>

        // TODO: Need to study the different arguments to control barier and why a gpu.barrier does 
        // not legalize
        spv.ControlBarrier "Workgroup", "Workgroup", "AcquireRelease" // gpu.barrier

        %accumi = scf.for %k = %lb to %ub step %step iter_args(%accum_iter0 = %accum_iter) -> (f32) {

          %asi = load %aSh[%tidy, %k] : memref<4x4xf32, 3>
          %bsi = load %bSh[%k, %tidx] : memref<4x4xf32, 3>
          %t = mulf %asi, %bsi : f32
          %accum_next = addf %accum_iter0, %t : f32

          scf.yield %accum_next : f32
        }
        
        spv.ControlBarrier "Workgroup", "Workgroup", "AcquireRelease" // gpu.barrier

        scf.yield %accumi : f32
      }

      store %accum, %c[%i, %j] : memref<32x32xf32>
      dealloc %aSh: memref<4x4xf32, 3>
      dealloc %bSh: memref<4x4xf32, 3>
      gpu.return
    }
  }
  func @main() {
    %onef = arith.constant 1.0 : f32


    %gdimx = arith.constant 8 : index // 32/4
    %gdimy = arith.constant 8 : index // 32/4
    %bdimx = arith.constant 4 : index
    %bdimy = arith.constant 4 : index
    %one = arith.constant 1 : index

    // Allocate the matricies and fill then with all ones
    %a = alloc() : memref<32x32xf32>
    %b = alloc() : memref<32x32xf32>
    %c = alloc() : memref<32x32xf32>
    %a0 = memref_cast %a : memref<32x32xf32> to memref<?x?xf32>
    call @fillResource2DFloat(%a0, %onef) : (memref<?x?xf32>, f32) -> ()
    %b0 = memref_cast %b : memref<32x32xf32> to memref<?x?xf32>
    call @fillResource2DFloat(%b0, %onef) : (memref<?x?xf32>, f32) -> ()
    %c0 = memref_cast %c : memref<32x32xf32> to memref<?x?xf32>
    call @fillResource2DFloat(%c0, %onef) : (memref<?x?xf32>, f32) -> ()

    // launch gpu kernel
    "gpu.launch_func"(%gdimx, %gdimy, %one, %bdimx, %bdimy, %one, %a, %b, %c) {kernel = @kernels::@dot} : (index, index, index, index, index, index, memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) -> ()

    %c2 = memref_cast %c : memref<32x32xf32> to memref<*xf32>
    call @print_memref_f32(%c2) : (memref<*xf32>) -> ()
    return
  }
  func @fillResource2DFloat(memref<?x?xf32>, f32)
  func @print_memref_f32(memref<*xf32>)
}
