#include "ScalarDimension.h"

namespace accera
{
namespace value
{
    ScalarDimension::ScalarDimension(Role role) :
        Scalar{ MakeScalar(ValueType::Index, "", role) }
    {
    }

    ScalarDimension::ScalarDimension(const std::string& name, Role role) :
        Scalar{ MakeScalar(ValueType::Index, name, role) }
    {
    }

    ScalarDimension::ScalarDimension(Value value, const std::string& name, Role role) :
        Scalar{ value, name, role }
    {
    }

    void ScalarDimension::SetValue(Value value)
    {
        Scalar::SetValue(value);
    }

    ScalarDimension::~ScalarDimension() = default;

} // namespace value
} // namespace accera