[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.Package.build(name[, format, mode, platform, tolerance, output_dir])`
Builds a HAT package.

## Arguments

argument | description | type/default
--- | --- | ---
`name` | The package name. | string
`format` | The format of the package. | `accera.Package.Format`
`mode` | The package mode, such as whether it is optimized or used for debugging. | `robopy.Package.Mode`, defaults to `Package.Mode.Release`
`platform` | The platform where the package will run. | `accera.Package.Platform`
`tolerance` | The tolerance for correctness checking when `mode = Package.Mode.Debug`. | float, defaults to 1e-5
`output_dir` | The path to an output directory. Defaults to the current directory if unspecified. | string

## Examples

Build a HAT package called `myPackage` containing `func1` for the host platform in the current directory:

```python
package = acc.Package()
package.add_function(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT, name="myPackage")
```

Build a HAT package called `myPackage` containing `func1` for the host platform in the `hat_packages` subdirectory:

```python
package = acc.Package()
package.add_function(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT, name="myPackage", output_dir="hat_packages")
```


Builds `myPackage` with additional intermediate MLIR files for debugging purposes:

```python
package = acc.Package()
package.add_function(plan, base_name="func1")
package.build(format=acc.Package.Format.MLIR, name="myPackage")
```

Build a package with error checking for `func1`, outputing error messages to `stderr` if the default implementation and the Accera implementation do not match within a tolerance of `1.0e-6`:

```python
package = acc.Package()
package.add_function(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT, name="myPackage", mode=acc.Package.Mode.DEBUG, tolerance=1.0e-6)
```

Cross-compile a HAT package called `myPackage` containing `func1` for the Raspberry Pi 3:

```python
pi3 = Target(model=Target.Model.RASPBERRY_PI3)
plan = schedule.create_action_plan(target=pi3)
package = acc.Package()
package.add_function(plan, base_name="func1")
package.build(format=acc.Package.Format.HAT, name="myPackagePi3", platform=acc.Package.Platform.RASPBIAN)
```

<div style="page-break-after: always;"></div>
