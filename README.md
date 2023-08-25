# `OceanFFT` Sample

The `OceanFFT` sample demonstrates the use of SYCL queues for Ocean height field on GPU devices. It is implemented using SYCL by migrating code from original CUDA source code and offloading computations to a GPU or CPU.

| Area                   | Description
|:---                    |:---
| What you will learn    | How to begin migrating CUDA to SYCL
| Time to complete       | 15 minutes
| Category               | Concepts and Functionality

>**Note**: This sample is migrated from NVIDIA CUDA sample. See the [oceanFFT](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/oceanFFT) sample in the NVIDIA/cuda-samples GitHub.

## Purpose

The `oceanFFT` sample shows the Ocean height field on the device at the same time.

> **Note**: The sample used the open-source SYCLomatic tool that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the Intel® DPC++ Compatibility Tool available to augment Base Toolkit.

This sample contains two versions of the code in the following folders:

| Folder Name          | Description
|:---                  |:---
|`01_dpct_output`      | Contains output of SYCLomatic Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
|`02_sycl_migrated`    | Contains manually migrated SYCL code from CUDA code.

## Prerequisites

| Optimized for         | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | Intel® Gen9 <br> Gen11 <br> Xeon CPU <br> Data Center GPU Max
| Software              | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2023.2.1

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.


## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA feature:

- Streams and Event Management
- Reduction

concurrentKernels involves a kernel that does no real work but runs at least for a specified number of iterations.

This code demonstrates the use of CUDA streams for concurrent execution of multiple kernels. It creates multiple streams, each associated with a separate kernel, and uses cudaStreamWaitEvent to introduce dependencies between the streams. The code measures the elapsed time for both serial and concurrent execution of the kernels. The kernel has a loop that iterates for a specific number of times without performing any actual work. Finally, the code verifies if the concurrent execution of kernels was faster than the serial execution based on the measured times.

>**Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html) for general information about the migration workflow.


## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the Code Using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates 100% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the concurrentKernels sample directory.
   ```
   cd cuda-samples/Samples/4_CUDA_Libraries/oceanFFT/
   ```
3. Generate a compilation database with intercept-build.
   ```
   intercept-build make
   ```
   This step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the Intel® SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../.. --use-custom-helper=api
   ```
## Manual Workarounds
The following manual change has been modified in order to complete the migration.
1. OpenGl feature is not supported in SYCL. Need to comment out the code before migration. 
      ```
      //#include <helper_gl.h>
      //#include <cuda_gl_interop.h>
      //#include <GLUT/glut.h>
      //#else
      //#include <GL/freeglut.h>
      //#endif

      //#include <rendercheck_gl.h>
      ```
   
2. Few CUDA headers are not migrated to SYCL which contain some macros,those are used in code. It has been changed manually.
      ```
      CUDART_SQRT_HALF_F
      ```
      Manually defined as below
      ```
      #define SYCLRT_SQRT_HALF_F 0.707106781f
      ```

## Build and Run the `oceanFFT` Sample

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*.

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D PVC=1 .. )
   $ make
   ```

   By default, this command sequence will build the  `02_sycl_migrated` version of the program.

3. Run `02_sycl_migrated` on GPU.
   ```
   make run_sm
   ```
   Run `02_sycl_migrated` on CPU.
   ```
   export ONEAPI_DEVICE_SELECTOR=opencl:cpu
   make run_sm
   unset ONEAPI_DEVICE_SELECTOR
   ```

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Example Output

The following example is for `02_sycl_migrated` for GPU on **Intel(R) UHD Graphics [0x9a60]**.
```
./a.out qatest
Compute capability 1.3
sdkDumpBin: <spatialDomain.bin>
> compareBin2Bin <float> nelements=65536, epsilon=0.10, threshold=0.15
   src_file <spatialDomain.bin>, size=262144 bytes
   ref_file <./data/ref_spatialDomain.bin>, size=262144 bytes
  OK
sdkDumpBin: <slopeShading.bin>
> compareBin2Bin <float> nelements=131072, epsilon=0.10, threshold=0.15
   src_file <slopeShading.bin>, size=524288 bytes
   ref_file <./data/ref_slopeShading.bin>, size=524288 bytes
  OK
Processing time : 92.435997 (ms)
```

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
