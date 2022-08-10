[//]: # (Project: Accera)
[//]: # (Version: v1.2.8)

# Accera v1.2.8 Reference

## `accera.Package.add_description([author, license, other, version])`
Adds descriptive metadata to the HAT package.
## Arguments

argument | description | type/default
--- | --- | ---
`author` | Name of the individual or group that authored the package. | string
`license` | The internet URL of the license used to release the package. | string
`other` | User-specific descriptive metadata | dictionary
`version` | The package version. | string

## Examples

Adds the standard version, license, and author description fields to the package:
```python
package.add_description(version​​​​​​​​​​​​​​​​="1.0", license="https://mit-license.org/", author="Microsoft Research")​​​​​​​​​​
```

Adds arbitrary user-defined metadata to describe the package:
```python
package.add_description(other={​​​​​​​​​​​​​​​​"title": "My Package Title", "source": "https://github.com/", "citations": ["https://arxiv.org/2021.12345/", "https://arxiv.org/2021.56789/"]}​​​​​​​​​​​​​​​​)
```


<div style="page-break-after: always;"></div>
