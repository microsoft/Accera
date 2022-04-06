[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Section 10: Building Packages
The `Package` class represents a collection of Accera-generated functions. Whenever a package is built, it creates a stand-alone function library that other pieces of software can use. Currently, Accera supports two package formats: HAT and MLIR.

## HAT package format
[HAT](https://github.com/microsoft/hat) "Header Annotated with TOML" is a format for packaging compiled libraries in the C programming language. HAT implies that a standard C header is styled with useful metadata in the TOML markup language.

Consider a nest that holds some loop-nest logic. To build a HAT package containing a function with this logic for the Windows operating system, we write the following lines of code: 
```python
package = acc.Package()
package.add(nest, base_name="func1")
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", platform=acc.Packag+-e.Platform.WINDOWS)
```

The result is two files: `myPackage.hat` and `myPackage.dll`. The output directory defaults to current working directory. We can change the output directory with `output_dir` set to a relative or absolute path:

```python
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", platform=acc.Package.Platform.WINDOWS, output_dir="hat_packages")
```

## MLIR package format
MLIR format is used for debugging the multiple stages of MLIR lowering, from the Accera DSL all the way to runnable code.
```python
package.build(format=acc.Package.Format.MLIR, name="myPackage")
```

## Function names in packages
We can specify the base name of a function when it is added to a package. The full function name is the base name followed by an automatically generated unique identifier. For example, if the base name is "myFunc" then the function name could be "myFunc_8f24bef5". If no base name is defined, the automatically-generated unique identifier becomes the function name.

The unique identifier ensures that no two functions share the same name. However, invoking the function from the client code becomes cumbersome because the function name changes each time the Accera package is updated and rebuilt. Therefore, the HAT file includes the client code to call the function without the unique identifier. Concretely, if the function signature in C is:
```
void myFunc_8f24bef5(const float* A, float* B);
```
then the HAT file also contains the line:
```
void (*myFunc)(const float* A, float* B) = myFunc_8f24bef5;
```
The above code makes the abbreviated name `myFunc` an alias of the full function name `myFunc_8f24bef5`. If multiple functions share the same base name, the first function in the HAT file gets the alias.

## Debug mode
A package can be built with` mode=acc.Package.Mode.DEBUG`. Doing so creates a special version of each function that validates its own correctness every time the function is called. From the outside, a debugging package looks identical to a standard package. However, each of its functions actually contains two different implementations: the Accera implementation (with all of the fancy scheduling and planning) and the trivial default implementation (without any scheduling or planning). When called, the function runs both implementations and asserts that their outputs are within the predefined tolerance. If the outputs don't match, the function prints error messages to `stderr`.
```python
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", mode=acc.Package.Mode.DEBUG, tolerance=1.0e-6)
```

## Adding descriptions
Accera allows us to specify some standard descriptive fields in a package:
```python
package.add_description(version="1.0", license="https://mit-license.org/", author="Microsoft Research")
```
Additionally, we can add arbitrary metadata to the package description as follows:
```python
package.add_description(other={"title": "My Package Title", "source": "https://github.com/", "citations": ["https://arxiv.org/2021.12345/", "https://arxiv.org/2021.56789/"]})
```


<div style="page-break-after: always;"></div>
