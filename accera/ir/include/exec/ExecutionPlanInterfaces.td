#ifndef EXECUTION_PLAN_INTERFACES
#define EXECUTION_PLAN_INTERFACES

include "mlir/IR/OpBase.td"

//
// Cache region common interface
//

def accxp_BeginCacheRegionOpInterface : OpInterface<"BeginCacheRegion"> {
  let description = [{
    Interface for the cache region begin ops
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the corresponding end op for this cache region
      }],
      "mlir::Operation*", "getEndOp", (ins)
    >,
    InterfaceMethod<[{
        Returns the ID for this cache region
      }],
      "int64_t", "getId", (ins)
    >
  ];
}

def accxp_EndCacheRegionOpInterface : OpInterface<"EndCacheRegion"> {
  let description = [{
    Interface for the cache region end ops
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the corresponding begin op for this cache region
      }],
      "mlir::Operation*", "getBeginOp", (ins)
    >
  ];
}


#endif // EXECUTION_PLAN_INTERFACES