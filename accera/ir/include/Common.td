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

class acc_HasShape<list<int> shape> :
    CPred<"$_self.cast<ShapedType>().hasStaticShape({" # !interleave(shape, ",") # "})">;

class acc_HasNumElements<int length> :
    CPred<"$_self.cast<ShapedType>().getNumElements() == " # length>;

class acc_MemRefOfTypeWithShape<list<Type> allowedTypes, list<int> shape> :
    Type<And<[MemRefOf<allowedTypes>.predicate, acc_HasShape<shape>]>,
        MemRefOf<allowedTypes>.description # " with shape { " #
        !interleave(shape, ",") # " }">;

class acc_MemRefOfTypeWithNumElements<list<Type> allowedTypes, int length> :
    Type<And<[MemRefOf<allowedTypes>.predicate, acc_HasNumElements<length>]>,
        MemRefOf<allowedTypes>.description # " with " # length # " elements">;

class acc_MemRefWithShape<list<int> shape> :
    Type<And<[AnyStaticShapeMemRef.predicate, acc_HasShape<shape>]>,
        AnyStaticShapeMemRef.description # " with shape { " # !interleave(shape, ",") # " }">;

class acc_MemRefWithNumElements<int length> :
    Type<And<[AnyStaticShapeMemRef.predicate, acc_HasNumElements<length>]>,
        AnyStaticShapeMemRef.description # " with " # length # " elements">;

class acc_TensorOfTypeWithShape<list<Type> allowedTypes, list<int> shape> :
    Type<And<[TensorOf<allowedTypes>.predicate, acc_HasShape<shape>]>,
        TensorOf<allowedTypes>.description # " with shape { " #
        !interleave(shape, ",") # " }">;

class acc_TensorOfTypeWithNumElements<list<Type> allowedTypes, int length> :
    Type<And<[TensorOf<allowedTypes>.predicate, acc_HasNumElements<length>]>,
        TensorOf<allowedTypes>.description # " with " # length # " elements">;


class acc_TensorWithShape<list<int> shape> :
    Type<And<[AnyStaticShapeTensor.predicate, acc_HasShape<shape>]>,
        AnyStaticShapeTensor.description # " with shape { " # !interleave(shape, ",") # " }">;

class acc_TensorWithNumElements<int length> :
    Type<And<[AnyStaticShapeTensor.predicate, acc_HasNumElements<length>]>,
        AnyStaticShapeTensor.description # " with " # length # " elements">;


class acc_ContainerOfTypeWithShape<list<Type> allowedTypes, list<int> shape> :
    Type<Or<[acc_MemRefOfTypeWithShape<allowedTypes, shape>.predicate,
        acc_TensorOfTypeWithShape<allowedTypes, shape>.predicate]>,
        MemRefOf<allowedTypes>.description # " or " #
        TensorOf<allowedTypes>.description # " with shape { " # !interleave(shape, ",") # " }">;

class acc_ContainerOfTypeWithNumElements<list<Type> allowedTypes, int length> :
    Type<Or<[acc_MemRefOfTypeWithNumElements<allowedTypes, length>.predicate,
        acc_TensorOfTypeWithNumElements<allowedTypes, length>.predicate]>,
        MemRefOf<allowedTypes>.description # " or " #
        TensorOf<allowedTypes>.description # " with " # length # " elements">;

class acc_ContainerWithShape<list<int> shape> :
    Type<Or<[acc_MemRefWithShape<shape>.predicate, acc_TensorWithShape<shape>.predicate]>,
        acc_MemRefWithShape<shape>.description # " or " #
        acc_TensorWithShape<shape>.description>;

class acc_ContainerWithNumElements<int length> :
    Type<Or<[acc_MemRefWithNumElements<length>.predicate, acc_TensorWithNumElements<length>.predicate]>,
        acc_MemRefWithNumElements<length>.description # " or " #
        acc_TensorWithNumElements<length>.description>;

def acc_NumericType :
    Type<Or<[AnyInteger.predicate, AnyFloat.predicate, Index.predicate]>, "Arithmetic type">;

def acc_ScalarOrVectorNumericType :
    AnyTypeOf<[acc_NumericType, VectorOf<[acc_NumericType]>]>;

def acc_IntegerOrIntegerVectorNumericType :
    AnyTypeOf<[AnyInteger, VectorOf<[AnyInteger]>]>;

def acc_FloatOrFloatVectorNumericType :
    AnyTypeOf<[AnyFloat, VectorOf<[AnyFloat]>]>;

class acc_Scalarlike<Type type> :
    AnyTypeOf<[type, acc_ContainerOfTypeWithNumElements<[type], 1>]>;

def acc_Indexlike : AnyTypeOf<[Index, acc_Scalarlike<AnySignlessInteger>]>;

def acc_BoolType : Type<I1.predicate, "Boolean type">;

def acc_ScalarOrVectorBoolType :
    AnyTypeOf<[acc_BoolType, VectorOf<[acc_BoolType]>]>;

#endif // ACCERA_COMMON_TD
