# Cross compile HAT and test ONNX transformer model on Raspberry Pi 3


## This includes two parts: cross compile HAT package for the target on your host machine, then copy the package to the target machine and run the test.

Please do not try to build onnxruntime natively on Raspberry Pi 3, it's too slow considering you also have to build/install up-to-date cmake, python and other dependencies. 

Please always use cross compilation (we use docker in this case), please do not use WSL as your devbox to build docker image, it doesn't work nicely for the Dockerfile we use. Grab an Azure VM (18.04.1-Ubuntu) instead.


## Step 1. Create a folder named `onnx-build` on your devbox(18.04.1-Ubuntu)

You can use whatever name, this folder is where to place all the packages, binaries, model files and scripts we need to run Transformer model on Raspberry Pi 3, including:

* onnxruntime .whl file
* gpt2 model .onnx file
* HAT package emitted for Pi 3
* scripts to drive the cross compilation and test

## Step 2. Download the ONNX transformer model gpt2_small.onnx 

Create a subfolder named `testdata` in `onnx-build`, download the model to that folder.


## Step 3. Enlist onnxruntime source code

Clone onnxruntime repo, get your own credential, check out this branch `dev/kerha/robocode_ep` where we add accera as a provider to ORT. We clone this repro for two reasons, one is that the scripts used to emit and run reside in this branch, second is that your devbox needs onnxruntime installed for cross compilition, you can build and install from the source code (or just pip install the public version)

 
## Step 4. Build onnxruntime for Raspberry Pi 3

Follow the instruction in this link to build onnxruntime for Raspberry Pi 3 using docker, you need to modify the corresponding lines as marked yellow in the docker file `Dockerfile.arm32v7`:

```dockerfile
ONNXRUNTIME_REPO=https://PAT@aiinfra.visualstudio.com/Lotus/_git/onnxruntime
ARG ONNXRUNTIME_SERVER_BRANCH=dev/kerha/robocode_ep
ARG BUILDARGS="--config ${BUILDTYPE} --arm64"
RUN ./build.sh --use_openmp ${BUILDARGS} --update --build --build_shared_lib --build_wheel --enable_developer_mode
```

After you've built the onnxruntime .whl file, e.g.  `onnxruntime-1.7.0-cp38-cp38-linux_armv7l.whl`, copy it to `onnx-build` folder.

 
## Step 5. Emit(cross compile) HAT package for the Transformer model for the target - Raspberry Pi 3

* Go to onnxruntime repo(`dev/kerha/robocode_ep`), copy the following scripts to `onnx-build` folder. We just attach them here for easy access.

  * emit_hat_package.py - emits (cross compiles) HAT package of the Transformer model for the target you specify
  * run_model_test.py - drives the performance test for CPU and Accera execution providers

* Go to Accera repo, copy the following scripts to `onnx-build` folder. Attach them here for easy access.

    * hat_file.py
    * hat_package.py
    * onnx_hat_package.py

* Build onnxruntime package and accera package, then pip install both of them on your devbox.

For accera, run `python setup.py build -b build -t build bdist_wheel -d build/dist`
For onnxruntime, run `build.bat --config Release --parallel --build_wheel --enable_developer_mode --cmake_generator Ninja --skip_tests`

* Go to `onnx-build` folder, run `python emit_hat_package.py -t pi3`, this will generate HAT package for the model in `testdata` folder for Raspberry pi3 target. 

* Thus far we've got all the things we need to test ONNX Transformer model on Pi 3, all in `onnx-build` folder.

## Step 6. Test ONNX Transformer model on Raspberry Pi 3 with Accera provider

* Copy `onnx-build` folder to Pi 3, you can delete all the IR files from this folder to save space before copy.

* Install onnxruntime .whl file built from docker file on the pi3, e.g.
Run `python -m pip install onnxruntime-1.7.0-cp38-cp38-linux_armv7l.whl`

* Now you are ready to run the test script on the Pi 3
Run `python run_model_test.py`

* If you encounter the following error when running the test, set this env variable: `export ALLOW_RELEASED_ONNX_OPSET_ONLY=0`

ONNX Runtime only *guarantees* support for models stamped with official released ONNX opset versions. Opset 14 is under development and support for this is limited. The operator schemas and or other functionality could possibly change before next ONNX release and in this case ONNX Runtime will not guarantee backward compatibility. Current official support for domain ai.ONNX is till opset 13.

Please note: due to the constrained resource on pi 3, you might have to modify `run_model_test.py` to use smaller numbers of input sets, and smaller batch size, so that you don't have to wait for hours to get the result.

Please note: there was an issue being investigated that onnxruntime fails to reserve(allocate) memory for the output shapes in Accera provider, the error is shown as below, to solve this problem, we need to separate CPU EP test from Accera EP test to avoid allocating large amounts of memory in one method:

```
2021-05-22 04: 32•.00. 333316844 [I: onnxruntime:Defau1t, bfc_arena.cc: 280 AllocateRawInterna1J Extending BFCArena for Accera. rounded_bytes: 7077888 2021-05-22 04: 32 :02.615942979 [E: onnxruntime: , inference_session. cc :1300 operator Exception during initialization: /code/onnxruntime/onnxruntime/core/framework/bfc_arena.cc: 305 void* onnxruntime: : BFCArena: : AllocateRawInterna1 (size_t, bool) Fai led to allocate memory for requested buffer of size 7077888 expection <class 'onnxruntime. capi . RuntimeException occurred.
```