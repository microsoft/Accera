![Accera logo](https://raw.githubusercontent.com/microsoft/Accera/main/docs/assets/Accera_darktext.png)
<div style="margin-bottom:30px"></div>

<a href="https://pypi.org/accera"><img src="https://badge.fury.io/py/accera.svg" alt="PyPI package version"/></a> <a href="https://pypi.org/accera"><img src="https://img.shields.io/pypi/pyversions/accera" alt="Python versions"/></a> ![MIT License](https://img.shields.io/github/license/microsoft/Accera)

Accera is a programming model, a domain-specific programming language embedded in Python (eDSL), and an optimizing cross-compiler for compute-intensive code. Accera currently supports CPU and GPU targets and focuses on optimization of nested for-loops.

Writing highly optimized compute-intensive code in a traditional programming language is a difficult and time-consuming process. It requires special engineering skills, such as fluency in Assembly language and a deep understanding of computer architecture. Manually optimizing the simplest numerical algorithms already requires a significant engineering effort. Moreover, highly optimized numerical code is prone to bugs, is often hard to read and maintain, and needs to be reimplemented every time a new target architecture is introduced. Accera aims to solve these problems.

Accera has three goals:

* Performance: generate the fastest implementation of any compute-intensive algorithm.
* Readability: do so without sacrificing code readability and maintainability.
* Writability: a user-friendly programming model, designed for agility.

## Installation
Read the [Install](https://raw.githubusercontent.com/microsoft/Accera/main/docs/Install/README.md) instructions for how to build Accera from source or install pre-built packages.

## Documentation
Get to know Accera by reading the [Documentation](https://raw.githubusercontent.com/microsoft/Accera/main/docs/README.md).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
