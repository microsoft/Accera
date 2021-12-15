Some suggestions to debug crashes while running the generated IR.

## Switch to the Debug cmake configuration.

Run `accc.py` with `-c Debug` to enable debug symbols and asserts.

## Enable Pass Dumping

Run `accc.py` with the `--dump_all_passes` flag to dump the result of each intermediate IR level. You can then inspect higher level IR to get more context around the crash. The IR is saved to the `all_passes` directory.

You may find it useful to also enable Location Tracking (next section).

## Enable Location Tracking

Use `LocationGuard` to tag your DSL calls. These will percolate through the different levels of IR and can be used to narrow down the problematic DSL call(s).

```cpp
#include <value/include/Debugging.h>

void Accera_Sample_Function(...)
{
    // You can tag anywhere in the DSL.
    // GET_LOCATION() will indicate the line number for this file.
    value::LocationGuard myOuterRegion(GET_LOCATION());

    nest.Set([&] {
        // within a lambda...
        value::LocationGuard myNestRegion(GET_LOCATION());

        ...
    });

    {
        // within a scope...
        value::LocationGuard myScope(GET_LOCATION());

        ...
    }
}
```
The tags should show up in the intermediate IR levels. See the previous section for how to enable pass dumping.

```
    %19 = "accv.bin_op"(%14, %c1_i64_0) {predicate = 2 : i64} : (index, i64) -> index loc("/path/to/Accera_Sample.cpp":76:0)
    ...
```

## Logging

Use `Print` to emit logging statements.

```cpp
using namespace std::string_literals;

void Accera_Sample_Function(...)
{
    auto indices = nest.GetIndices();
    auto i = indices[0];
    ...

    nest.Set([&] {
        Print(i); // print a Scalar
        Print("\n"s); // print a string

        ...
    });
}
```

## Step-by-step Debugging (Advanced)

If the above do not help isolate the crash, you can create a test case that reproduces your issue, and use the debugger (e.g. `gdb`) to step through the Accera pipeline. Refer to examples in `accera/tests/ir/ir_tests.cpp`.
