[//]: # (Project: Accera)
[//]: # (Version: v1.2.14)

# Contributing Guide

Thank you for investing your time contributing a community case study!

In this guide, you will get an overview of the contribution workflow.

## Getting started

Read our [Code of Conduct](https://github.com/microsoft/Accera/raw/main/CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

Refer to the [Manual](../Manual/00%20Introduction.md) and [Tutorials](../Tutorials/README.md) to familiarize yourself with the Accera language and programming model.

## Components of a good case study

A good case study should have these components and characteristics:

1. Solves *one* specific task, such as matrix multiplication, matrix convolution, vector addition. If you have a series of tasks to solve, break them up into multiple case studies that reference one another.

2. Includes working Accera Python code implementing that task. At the end of the case study, the code should produce a HAT package using [`accera.Package.build()`](../Manual/10%20Packages.md).

3. Describes the thought process, considerations, pros and cons of your implementation in a `README.md`.

4. If the case study generates several implementations (for example, using [Parameter Grids](../Manual/09%20Parameters.md)), include the following:
  - Benchmark results on a target machine (for example, your laptop). You can run `hatlib.run_benchmark` on your HAT package.
  - A description of the make and model of that target machine you used (for example, Intel Xeon E5). If you are unsure, you can use the output of this command:

    ```shell
    python -m cpuinfo
    ```

For some examples, refer to the published case studies in the [Table of Contents](README.md).

## Publishing your case study

All community case studies are published directly from the author's GitHub repository and linked to from the Accera GitHub repository.

Once you are ready to publish your case study:
1. Make your case study GitHub repository public (if you haven't done so already).

2. Edit [Case Studies/README.md](https://github.com/microsoft/Accera/blob/main/docs/Case%20Studies/README.md) to add your case study to the Table of Contents. The link should point to the git SHA for your latest commit. The format to use is: https://github.com/*user*/*repo*/blob/*git_sha*/*path_to_case_study*/README.md.

3. Create a [Pull Request](https://github.com/microsoft/Accera/compare) to submit your edits to [Case Studies/README.md](https://github.com/microsoft/Accera/blob/main/docs/Case%20Studies/README.md).
