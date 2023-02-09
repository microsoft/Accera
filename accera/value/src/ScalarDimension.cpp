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
        ScalarDimension{ role }
    {
        SetName(name);
    }

    ScalarDimension::ScalarDimension(Value value, const std::string& name, Role role) :
        Scalar{ value, "", role }
    {
        SetName(name);
    }

    void ScalarDimension::SetName(const std::string& name)
    {
        _name = name;
    }

    std::string ScalarDimension::GetName() const
    {
        return _name;
    }

    void ScalarDimension::SetValue(Value value)
    {
        Scalar::SetValue(value);
    }
} // namespace value
} // namespace accera