[//]: # (Project: Accera)
[//]: # (Version: <<VERSION>>)

# Section 10: Building Packages
The Accera `Package` class represents a collection of Accera-generated functions. When a package is built, it creates a stand-alone function library that can be used by other pieces of software. Accera currently supports two package formats: HAT and MLIR.

## HAT package format
[HAT](https://github.com/microsoft/hat) is a format for packaging compiled libraries in the C programming language. HAT stands for "Header Annotated with TOML", which implies that a standard C header is decorated with useful metadata in the TOML markup language.

Say that `nest` contains some loop-nest logic. To build a HAT package that contains a function with this logic for the Windows operating system, we do the following:
```python
package = acc.Package()
package.add(nest, base_name="func1")
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", platform=acc.Package.Platform.WINDOWS)
```

The result is two files: `MyPackage.hat` and `MyPackage.lib`. The output directory defaults to the current working directory. To change the output directory:

```python
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", platform=acc.Package.Platform.WINDOWS, output_dir="hat_packages")
```

## MLIR package format
MLIR format is mainly used for debugging purposes, and as a way to follow the multiple lowering MLIR stages, from Accera DSL all the way to runnable code.
```python
package.build(format=acc.Package.Format.MLIR, name="myPackage")
```

## Function names in packages
When a function is added to a package, the user specifies its base name. The full function name is the base name followed by an automatically generated unique identifier. For example, if the base name is "myFunc" then the function name could be "myFunc_8f24bef5". If no base name is given, the function name is just the automatically-generated unique identifier.

The unique identifier ensures that no two functions share the same name, but also makes it harder to call the function from client code. Specifically, each time that the Accera package is updated and rebuilt, the function name could change. Therefore, the HAT file also includes the client code to call the function without the unique identifier. Concretely, if the function signature in C is
```
void myFunc_8f24bef5(const float* A, float* B);
```
then the HAT file also contains the line:
```
void (*myFunc)(const float* A, float* B) = myFunc_8f24bef5;
```
The above basically makes the abbreviated name `myFunc` a synonym of the full function name `myFunc_8f24bef5`. If multiple functions share the same base name, an arbitrary one of them gets the abbreviation.

## Debug mode
A package can be built with the option `mode=acc.Package.Mode.DEBUG`. This creates a special version of each function that checks its own correctness each time the function is called. From the outside, a debugging package looks identical to a standard package. However, each of its functions actually contains two different implementations: the RoboCode implementation (with all of the fancy scheduling and planning) and the trivial default implementation (without any of the scheduling or planning). When called, the function runs both implementations and asserts that their outputs are within some predefined tolerance. If the outputs don't match, the function prints error messages to `stderr`.
```python
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", mode=acc.Package.Mode.DEBUG, tolerance=1.0e-6)
```

## Adding descriptions
Accera allows us to specify some standard descriptive fields in a package:
```python
package.add_description(version​​​​​​​​​​​​​​​​="1.0", license="https://mit-license.org/", author="Microsoft Research")​​​​​​​​​​
```
Additionally, we can add arbitrary metadata to the package description as follows:
```python
package.add_description(other={​​​​​​​​​​​​​​​​"title": "My Package Title", "source": "https://github.com/", "citations": ["https://arxiv.org/2021.12345/", "https://arxiv.org/2021.56789/"]}​​​​​​​​​​​​​​​​)
```


<div style="page-break-after: always;"></div>
