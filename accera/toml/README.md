# Accera TOML Data

## Parsing TOML data from headers
The `toml_header_parser.py` python utility defines utilities for parsing the TOML data from a Accera library header file, extracting the well-known-named entries and collecting custom parameter values.

This script requires that you have the `tomlkit` python package installed via
```sh
pip install tomlkit
```

## Emitting TOML Data
Accera-generated library headers have TOML metadata embedded in them. The header file contains both the C++ function declarations for the library as well as the TOML metadata for those functions giving:
- The library name
- A list of the module names
- Prologue/Epilogue C++ code declarations for the header file as a string
- Utility module declarations
- TOML data per Accera funtion variant

Each variant of a Accera function will produce a module with TOML data giving:
- The variant module name
- The variant function name
- The variant module initialize function name
- The variant module de-initialize function name
- The variant domain
- Any custom parameters for that variant
- The C++ function declarations as a string

Example function variant data interwoven with C++ header declarations:
```c++
#ifdef TOML
[modules.MLASValueGEMM_256_256_256_module]
module_name = 'MLASValueGEMM_256_256_256_module'

    [modules.MLASValueGEMM_256_256_256_module.code]
    cpp = '''
#endif // TOML
void MLASValueGEMM_256_256_256(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t);

void _mlir_ciface_MLASValueGEMM_256_256_256(void*, void*, void*);

void MLASValueGEMM_256_256_256_module_initialize();

void _mlir_ciface_MLASValueGEMM_256_256_256_module_initialize();

void MLASValueGEMM_256_256_256_module_deinitialize();

void _mlir_ciface_MLASValueGEMM_256_256_256_module_deinitialize();


#ifdef TOML
'''

    [modules.MLASValueGEMM_256_256_256_module.metadata]
    BMatrixTileSize = [ 512, 128 ]
    _deinitialize_function = 'MLASValueGEMM_256_256_256_module_deinitialize'
    _function = 'MLASValueGEMM_256_256_256'
    _initialize_function = 'MLASValueGEMM_256_256_256_module_initialize'
    domain = [ 256, 256, 256 ]
    forceCacheBMatrix = false
    kUnroll = 4
    numColumnsInKernelScaleFactor = 2
    numRowsInKernel = 6
    simdVectorSize = 8
    simdVectorUnits = 16
#endif // TOML
```
