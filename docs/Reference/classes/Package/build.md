[//]: # (Project: Accera)
[//]: # (Version: v1.2.3)

# Accera v1.2.3 Reference

## `accera.Package.build(name[, format, mode, platform, tolerance, output_dir])`
Builds a HAT package.

## Arguments

argument | description | type/default
--- | --- | ---
`name` | The package name. | string
`format` | The format of the package. | `accera.Package.Format`, defaults to `Package.Format.HAT_STATIC`
`mode` | The package mode, such as whether it is optimized or used for debugging. | `robopy.Package.Mode`, defaults to `Package.Mode.Release`
`platform` | The platform where the package runs. | `accera.Package.Platform`
`tolerance` | The tolerance for correctness checking when `mode = Package.Mode.Debug`. | float, defaults to 1e-5
`output_dir` | The path to an output directory. Defaults to the current directory if unspecified. | string

## Examples

Build a Dynamically-linked HAT package called `myPackage` containing `func1` for the host platform in the current directory:

```python
package = acc.Package()
package.add(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage")
```

Build a statically-linked HAT package called `myPackage` containing `func1` for the host platform in the `hat_packages` subdirectory:

```python
package = acc.Package()
package.add(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT_STATIC, name="myPackage", output_dir="hat_packages")
```

Build a statically-linked `myPackage` with additional intermediate MLIR files for debugging purposes. To build a dynamically-linked package, use `acc.Package.Format.MLIR_DYNAMIC`:

```python
package = acc.Package()
package.add(plan, base_name="func1")
package.build(format=acc.Package.Format.MLIR_STATIC, name="myPackage")
```

Build a package with error checking for `func1`, outputing error messages to `stderr` if the default implementation and the Accera implementation do not match within a tolerance of `1.0e-6`:

```python
package = acc.Package()
package.add(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT_DYNAMIC, name="myPackage", mode=acc.Package.Mode.DEBUG, tolerance=1.0e-6)
```

Cross-compile a statically-linked HAT package called `myPackage` containing `func1` for the Raspberry Pi 3. Note that dynamically-linked HAT packages are not supported for cross-compilation:

```python
pi3 = Target("Raspberry Pi 3B", category=Target.Category.CPU)
plan = schedule.create_plan(target=pi3)
package = acc.Package()
package.add(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT_STATIC, name="myPackagePi3", platform=acc.Package.Platform.RASPBIAN)
```

<div style="page-break-after: always;"></div>


