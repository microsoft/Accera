////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ACCERA_COMMON_TD
#define ACCERA_COMMON_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class rc_HasShape<list<int> shape> :
    CPred<"$_self.cast<ShapedType>().hasStaticShape({" # !interleave(shape, ",") # "})">;

class rc_HasNumElements<int length> :
    CPred<"$_self.cast<ShapedType>().getNumElements() == " # length>;

class rc_MemRefOfTypeWithShape<list<Type> allowedTypes, list<int> shape> :
    Type<And<[MemRefOf<allowedTypes>.predicate, rc_HasShape<shape>]>,
        MemRefOf<allowedTypes>.description # " with shape { " #
        !interleave(shape, ",") # " }">;

class rc_MemRefOfTypeWithNumElements<list<Type> allowedTypes, int length> :
    Type<And<[MemRefOf<allowedTypes>.predicate, rc_HasNumElements<length>]>,
        MemRefOf<allowedTypes>.description # " with " # length # " elements">;

class rc_MemRefWithShape<list<int> shape> :
    Type<And<[AnyStaticShapeMemRef.predicate, rc_HasShape<shape>]>,
        AnyStaticShapeMemRef.description # " with shape { " # !interleave(shape, ",") # " }">;

class rc_MemRefWithNumElements<int length> :
    Type<And<[AnyStaticShapeMemRef.predicate, rc_HasNumElements<length>]>,
        AnyStaticShapeMemRef.description # " with " # length # " elements">;

class rc_TensorOfTypeWithShape<list<Type> allowedTypes, list<int> shape> :
    Type<And<[TensorOf<allowedTypes>.predicate, rc_HasShape<shape>]>,
        TensorOf<allowedTypes>.description # " with shape { " #
        !interleave(shape, ",") # " }">;

class rc_TensorOfTypeWithNumElements<list<Type> allowedTypes, int length> :
    Type<And<[TensorOf<allowedTypes>.predicate, rc_HasNumElements<length>]>,
        TensorOf<allowedTypes>.description # " with " # length # " elements">;


class rc_TensorWithShape<list<int> shape> :
    Type<And<[AnyStaticShapeTensor.predicate, rc_HasShape<shape>]>,
        AnyStaticShapeTensor.description # " with shape { " # !interleave(shape, ",") # " }">;

class rc_TensorWithNumElements<int length> :
    Type<And<[AnyStaticShapeTensor.predicate, rc_HasNumElements<length>]>,
        AnyStaticShapeTensor.description # " with " # length # " elements">;


class rc_ContainerOfTypeWithShape<list<Type> allowedTypes, list<int> shape> :
    Type<Or<[rc_MemRefOfTypeWithShape<allowedTypes, shape>.predicate,
        rc_TensorOfTypeWithShape<allowedTypes, shape>.predicate]>,
        MemRefOf<allowedTypes>.description # " or " #
        TensorOf<allowedTypes>.description # " with shape { " # !interleave(shape, ",") # " }">;

class rc_ContainerOfTypeWithNumElements<list<Type> allowedTypes, int length> :
    Type<Or<[rc_MemRefOfTypeWithNumElements<allowedTypes, length>.predicate,
        rc_TensorOfTypeWithNumElements<allowedTypes, length>.predicate]>,
        MemRefOf<allowedTypes>.description # " or " #
        TensorOf<allowedTypes>.description # " with " # length # " elements">;

class rc_ContainerWithShape<list<int> shape> :
    Type<Or<[rc_MemRefWithShape<shape>.predicate, rc_TensorWithShape<shape>.predicate]>,
        rc_MemRefWithShape<shape>.description # " or " #
        rc_TensorWithShape<shape>.description>;

class rc_ContainerWithNumElements<int length> :
    Type<Or<[rc_MemRefWithNumElements<length>.predicate, rc_TensorWithNumElements<length>.predicate]>,
        rc_MemRefWithNumElements<length>.description # " or " #
        rc_TensorWithNumElements<length>.description>;

def rc_NumericType :
    Type<Or<[AnySignlessInteger.predicate, AnyFloat.predicate, Index.predicate]>, "Arithmetic type">;

def rc_ScalarOrVectorNumericType :
    AnyTypeOf<[rc_NumericType, VectorOf<[rc_NumericType]>]>;

class rc_Scalarlike<Type type> :
    AnyTypeOf<[type, rc_ContainerOfTypeWithNumElements<[type], 1>]>;

def rc_Indexlike : AnyTypeOf<[Index, rc_Scalarlike<AnySignlessInteger>]>;

def rc_BoolType : Type<I1.predicate, "Boolean type">;

def rc_ScalarOrVectorBoolType :
    AnyTypeOf<[rc_BoolType, VectorOf<[rc_BoolType]>]>;

#endif // ACCERA_COMMON_TD
