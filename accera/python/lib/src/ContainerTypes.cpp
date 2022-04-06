////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraTypes.h"

#include <value/include/FastMath.h>
#include <value/include/ScalarOperations.h>

namespace py = pybind11;
namespace value = accera::value;
namespace util = accera::utilities;

using namespace pybind11::literals;

namespace accera::python::lang
{
namespace
{
    void DefineContainerEnums(py::module& module, py::module& subModule)
    {
        py::enum_<value::ValueType>(module, "ScalarType", "An enumeration of primitive types")
            .value("undefined", value::ValueType::Undefined, "undefined type")
            .value("void", value::ValueType::Void, "void type")
            .value("bool", value::ValueType::Boolean, "1 byte boolean")
            .value("int8", value::ValueType::Int8, "1 byte signed integer")
            .value("int16", value::ValueType::Int16, "2 byte signed integer")
            .value("int32", value::ValueType::Int32, "4 byte signed integer")
            .value("int64", value::ValueType::Int64, "8 byte signed integer")
            .value("uint8", value::ValueType::Byte, "1 byte unsigned integer")
            .value("uint16", value::ValueType::Uint16, "2 byte unsigned integer")
            .value("uint32", value::ValueType::Uint32, "4 byte unsigned integer")
            .value("uint64", value::ValueType::Uint64, "8 byte unsigned integer")
            .value("index", value::ValueType::Index, "index type")
            .value("float16", value::ValueType::Float16, "2 byte floating point")
            .value("float32", value::ValueType::Float, "4 byte floating point")
            .value("float64", value::ValueType::Double, "8 byte floating point");

        py::enum_<value::AllocateFlags>(subModule, "AllocateFlags", "An enumeration of allocation flags")
            .value("NONE", value::AllocateFlags::None)
            .value("THREAD_LOCAL", value::AllocateFlags::ThreadLocal);
    }

    void DefineContainerStructs(py::module& module, py::module& /*subModule*/)
    {
        // Docstring format: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
        // (Google style with Python 3 type annotations, type annotations inserted by pybind11)
        // Sphinx requires the indentation to be consistent, see the pattern below.

        py::class_<util::MemoryLayout>(module, "_MemoryLayout", R"pbdoc(
A class representing layout of a block of data in memory where the block can also contain padding
such that a certain offset is required to access the active memory inside the padded block.
)pbdoc")
            .def(py::init())
            .def(
                py::init([](std::vector<int64_t> size,
                            std::optional<std::vector<int64_t>> extent,
                            std::optional<std::vector<int64_t>> offset,
                            std::optional<std::vector<int64_t>> order) {
                    if (order && offset && extent)
                    {
                        return util::MemoryLayout(
                            util::MemoryShape(size),
                            util::MemoryShape(*extent),
                            util::MemoryShape(*offset),
                            util::DimensionOrder(*order));
                    }
                    else if (order && !offset && !extent)
                    {
                        return util::MemoryLayout(
                            util::MemoryShape(size),
                            util::DimensionOrder(*order));
                    }
                    else if (!order && offset && extent)
                    {
                        return util::MemoryLayout(
                            util::MemoryShape(size),
                            util::MemoryShape(*extent),
                            util::MemoryShape(*offset));
                    }
                    else if (!order && !offset && !extent)
                    {
                        return util::MemoryLayout(util::MemoryShape(size));
                    }
                    else
                    {
                        throw std::logic_error("Unsupported argument combination!");
                    }
                }),
                "size"_a,
                "extent"_a.none() = py::none(),
                "offset"_a.none() = py::none(),
                "order"_a.none() = py::none(),
                R"pbdoc(
General constructor.

Args:
    size: The extent of the active area of the memory region, expressed in logical dimensions
        (`MemoryLayout({M, N}, {0, 1})` creates a row-major array with M rows and N columns,
        and `MemoryLayout({M, N}, {1, 0})` creates a column-major array with M rows and N columns)
    extent: The extent of the allocated memory of the memory region
    offset: The offset into memory to the active area of the memory region
    order: The ordering of the logical dimensions in memory (e.g., [0, 1] for the canonical row-major ordering
        of 2D arrays, and [1, 0] for column-major.
)pbdoc")
            .def("__repr__", [](const util::MemoryLayout& layout) {
                return layout.ToString();
            })
            .def("set_memory_space", [](util::MemoryLayout& layout, util::MemorySpace space) {
                return layout.SetMemorySpace(space);
            })
            .def(py::self == py::self)
            .def_static("get_subarray_layout", [](const util::MemoryLayout& originalLayout, std::vector<int64_t> size) {
                return util::MemoryLayout(
                    util::MemoryShape(size),
                    originalLayout.GetExtent(),
                    originalLayout.GetOffset());
            });
    }

    void DefineContainerFunctions(py::module& module)
    {
        module.def("abs", &value::Abs);
        module.def("max", [](value::Scalar s1, value::Scalar s2) {
            return value::Max(s1, s2); // compiler-assist
        });
        module.def("min", &value::Min);
        module.def("ceil", &value::Ceil);
        module.def("floor", &value::Floor);
        module.def("sqrt", &value::Sqrt);
        module.def("exp", &value::Exp);
        module.def("fast_exp", &value::FastExp);
        module.def("fast_exp_mlas", &value::FastExpMlas);
        module.def("log", &value::Log);
        module.def("log10", &value::Log10);
        module.def("log2", &value::Log2);
        module.def("sin", &value::Sin);
        module.def("cos", &value::Cos);
        module.def("tan", &value::Tan);
        module.def("sinh", &value::Sinh);
        module.def("cosh", &value::Cosh);
        module.def("tanh", &value::Tanh);
        module.def("logical_not", [](value::Scalar s) {
            return value::Cast(value::LogicalNot(s), s.GetType());
        });
        module.def("logical_and", [](value::Scalar s1, value::Scalar s2) {
            return value::Cast(s1 && s2, s1.GetType());
        });
        module.def("logical_or", [](value::Scalar s1, value::Scalar s2) {
            return value::Cast(s1 || s2, s1.GetType());
        });
        module.def("_cast", [](value::Scalar s, value::ValueType type) {
            return value::Cast(s, type);
        });
        module.def("_unsigned_cast", [](value::Scalar s, value::ValueType type) {
            return value::UnsignedCast(s, type);
        });
    }

    void DefineArrayClass(py::module& module)
    {
        // gcc8 can't properly parse raw strings in macros, workaround by using a variable
        auto CTOR_DOCSTRING = R"pbdoc(
Constructs an instance from a 1D list reshaped into the given array shape

Args:
    data: The data represented as a 1D list, in canonical row-major layout
    memory_layout:  The memory layout to apply
    name: An optional name for the emitted construct
)pbdoc";

#define ADD_CTOR(TYPE) \
    def(py::init<const std::vector<TYPE>&, const std::optional<util::MemoryLayout>, const std::string&>(), "data"_a, "memory_layout"_a, "name"_a = "", CTOR_DOCSTRING)

        py::class_<value::Array>(module, "Array", "A View type that wraps a _Valor instance and enforces a memory layout that represents a multidimensional array")
            .def(py::init<value::Array>())
            .def(py::init([](value::ValueType type, const util::MemoryLayout& layout, const std::string& name) {
                     return value::Array(value::Value(type, layout), name);
                 }),
                 "element_type"_a,
                 "memory_layout"_a,
                 "name"_a = "")
            .def(py::init<value::Value, const std::string&>(),
                 "value"_a,
                 "name"_a = "")
            .def(py::init([](const py::buffer buffer, const std::optional<util::MemoryLayout> layout, const std::string& name) {
                     // constructor for np.float32 python buffers, a special case because python floats are 64-bit
                     py::buffer_info info = buffer.request();
                     if (info.format != py::format_descriptor<float>::format())
                     {
                         throw std::runtime_error("Unsupported buffer format");
                     }
                     assert(info.itemsize == sizeof(float));
                     std::vector<float> v(static_cast<float*>(info.ptr), static_cast<float*>(info.ptr) + info.size);
                     return value::Array(v, layout, name);
                 }),
                 "buffer"_a,
                 "memory_layout"_a,
                 "name"_a = "")
            // .ADD_CTOR(bool) // BUG: bool requires std::vector nonsense
            .ADD_CTOR(int8_t)
            .ADD_CTOR(int16_t)
            .ADD_CTOR(int32_t)
            .ADD_CTOR(int64_t)
            .ADD_CTOR(uint8_t)
            .ADD_CTOR(uint16_t)
            .ADD_CTOR(uint32_t)
            .ADD_CTOR(uint64_t)
            .ADD_CTOR(float)
            // .ADD_CTOR(float16_t) // no built-in type
            .ADD_CTOR(double)
            .def(
                "__getitem__", [](value::Array& array, const std::vector<value::Scalar>& indices) {
                    return array(indices);
                },
                "Array element access operator. Returns the Scalar value wrapping the value that is at the specified indices within the array")
            .def(
                "__getitem__", [](value::Array& array, const value::Scalar& index) {
                    std::vector<value::Scalar> indices{ index };
                    return array(indices);
                },
                "Array element access operator. Returns the Scalar value wrapping the value that is at the specified index within the array")
            .def(
                "__setitem__", [](value::Array& array, const std::vector<value::Scalar>& indices, value::Scalar new_value) {
                    array(indices) = new_value;
                },
                "Array element access operator. Sets the Scalar value that is at the specified indices within the array")
            .def(
                "__setitem__", [](value::Array& array, const value::Scalar& index, value::Scalar new_value) {
                    std::vector<value::Scalar> indices{ index };
                    array(indices) = new_value;
                },
                "Array element access operator. Sets the Scalar value that is at the specified index within the array")
            .def("_copy", &value::Array::Copy)
            .def(
                "sub_array", [](value::Array& arr, const std::vector<value::Scalar>& offsets, const std::vector<int64_t>& shape, std::optional<std::vector<int64_t>> strides) {
                    return arr.SubArray(offsets, shape, strides);
                },
                "offsets"_a,
                "shape"_a,
                "strides"_a = std::nullopt)
            .def("_slice", &value::Array::Slice)
// TODO: Enable when functionality is needed and semantics are fully cleared
#if 0
            .def("_merge_dimensions", &value::Array::MergeDimensions, "dim1"_a, "dim2"_a)
            .def("_split_dimension", &value::Array::SplitDimension, "dim"_a, "size"_a)
            .def("_reshape", &value::Array::Reshape, "layout"_a)
#endif // 0
            .def(
                "_reorder", [](const value::Array& arr, const std::vector<int64_t>& order) {
                    return arr.Reorder(order);
                },
                "order"_a)
            .def_property_readonly("shape", [](const value::Array& arr) -> std::vector<int64_t> {
                return arr.Shape().ToVector();
            })
            .def_property_readonly("element_type", &value::Array::GetType, "The element type, e.g. accera.Type.float32")
            .def_property_readonly("layout", &value::Array::GetLayout, "The memory layout")
            .def_property_readonly("_rank", &value::Array::Rank)
            .def_property_readonly("_size", &value::Array::Size)
            .def_property("name", &value::Array::GetName, &value::Array::SetName, "The optional name")
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def("__iadd__", py::overload_cast<value::Scalar>(&value::Array::operator+=))
            .def("__isub__", py::overload_cast<value::Scalar>(&value::Array::operator-=))
            .def("__imul__", py::overload_cast<value::Scalar>(&value::Array::operator*=))
            .def("__idiv__", py::overload_cast<value::Scalar>(&value::Array::operator/=))
            .def_property_readonly("_value", &value::Array::GetValue);

        py::implicitly_convertible<value::Value, value::Array>();
#undef ADD_CTOR
    }

    void DefineValueClass(py::module& module)
    {
#define ADD_CTOR(TYPE) \
    def(py::init<std::vector<TYPE>, const util::MemoryLayout&>(), "data"_a.noconvert(), "layout"_a)

        py::class_<value::Value>(module, "_Valor", R"pbdoc(
The basic type upon which most operations are based. Wraps either C++ data (constant data) or data that is
specific to the EmitterContext, specified by the Emittable type.
)pbdoc")
            .def(py::init<value::Value>())
            .def(py::init([](value::ValueType type, const util::MemoryLayout& layout) {
                     return value::Value(type, layout);
                 }),
                 "type"_a,
                 "layout"_a)
            // .ADD_CTOR(bool) // BUG: bool requires std::vector nonsense
            .ADD_CTOR(int8_t)
            .ADD_CTOR(int16_t)
            .ADD_CTOR(int32_t)
            .ADD_CTOR(int64_t)
            .ADD_CTOR(uint8_t)
            .ADD_CTOR(uint16_t)
            .ADD_CTOR(uint32_t)
            .ADD_CTOR(uint64_t)
            .ADD_CTOR(float)
            // .ADD_CTOR(float16_t) // no built-in type
            .ADD_CTOR(double)
            .def(py::hash(py::self))
            .def_property("layout", &value::Value::GetLayout, &value::Value::SetLayout)
            .def_property("name", &value::Value::GetName, &value::Value::SetName)
            .def_property_readonly("is_empty", &value::Value::IsEmpty);

        py::implicitly_convertible<value::ViewAdapter, value::Value>();

#undef ADD_CTOR
    }

    void DefineScalarClass(py::module& module)
    {
#define ADD_CTOR(TYPE) \
    def(py::init<TYPE>(), "data"_a.noconvert())

        py::class_<value::Scalar>(module, "Scalar", "A View type that wraps a _Valor instance and enforces a memory layout that represents a single value")
            .def(py::init<value::Scalar>())
            .def(py::init<value::Value, const std::string&>(), "value"_a, "name"_a = "")
            .def(py::init([](value::Array arr) {
                return value::Scalar(arr.GetValue());
            }))
            .def(py::init([](value::ValueType type) {
                     return value::MakeScalar(type, "");
                 }),
                 "type"_a)
            // .ADD_CTOR(bool) // BUG: bool requires std::vector nonsense
            .ADD_CTOR(int8_t)
            .ADD_CTOR(int16_t)
            .ADD_CTOR(int32_t)
            .ADD_CTOR(int64_t)
            .ADD_CTOR(uint8_t)
            .ADD_CTOR(uint16_t)
            .ADD_CTOR(uint32_t)
            .ADD_CTOR(uint64_t)
            // .ADD_CTOR(float16_t) // no built-in type
            .ADD_CTOR(float)
            .ADD_CTOR(double)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def(py::self % py::self)
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def(py::self *= py::self)
            .def(py::self /= py::self)
            .def(py::self %= py::self)
            .def(-py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self <= py::self)
            .def(py::self >= py::self)
            .def(py::self < py::self)
            .def(py::self > py::self)
            .def(py::self * float())
            .def(float() * py::self)
            .def(py::self * int())
            .def(int() * py::self)
            .def("__rshift__", [](value::Scalar& s, value::Scalar& shift) {
                return value::UnsignedShiftRight(s, value::Cast(shift, s.GetType()));
            })
            .def("__lshift__", [](value::Scalar& s, value::Scalar& shift) {
                return value::ShiftLeft(s, value::Cast(shift, s.GetType()));
            })
            .def("__floordiv__", [](value::Scalar& a, value::Scalar& b) {
                // Floor is limited to floating point types
                return a.GetValue().IsFloatingPoint() ? value::Floor(value::Divide(a, b)) : value::Divide(a, b);
            })
            .def("__and__", &value::BitwiseAnd)
            .def("__or__", &value::BitwiseOr)
            .def("__invert__", &value::BitwiseNot)
            .def("__xor__", &value::BitwiseXOr)
            .def("__pow__", &value::Pow)
            .def("copy", &value::Scalar::Copy)
            .def_property("name", &value::Scalar::GetName, &value::Scalar::SetName)
            .def_property_readonly("type", &value::Scalar::GetType)
            .def_property_readonly("_value", &value::Scalar::GetValue);

        py::implicitly_convertible<value::Value, value::Scalar>();
        py::implicitly_convertible<value::Array, value::Scalar>();
        py::implicitly_convertible<py::int_, value::Scalar>();
        py::implicitly_convertible<py::float_, value::Scalar>();

#undef ADD_CTOR
    }

    void DefineViewAdapterClass(py::module& module)
    {
        py::class_<value::ViewAdapter>(module, "ViewAdapter", "A helper type that can hold any View type")
            .def(py::init<value::Value>())
            .def(py::init<value::Scalar>())
            .def(py::init<value::Array>())
            .def("get_value", py::overload_cast<>(&value::ViewAdapter::GetValue))
            .def_property_readonly("_value", py::overload_cast<>(&value::ViewAdapter::GetValue));

        py::implicitly_convertible<value::Value, value::ViewAdapter>();
        py::implicitly_convertible<value::Scalar, value::ViewAdapter>();
        py::implicitly_convertible<value::Array, value::ViewAdapter>();
    }
} // namespace

void DefineContainerTypes(py::module& module, py::module& subModule)
{
    DefineContainerEnums(module, subModule);
    DefineContainerStructs(module, subModule);
    DefineContainerFunctions(module);

    // for large classes, use class-specific "Define" methods for readability
    DefineArrayClass(subModule);
    DefineValueClass(subModule);
    DefineScalarClass(subModule);
    DefineViewAdapterClass(subModule);
}
} // namespace accera::python::lang
