# requirements
cmake >= 3.10.0
opencv >= 3.0.0
openmp

# installation commands for x86 Linux 
<pre>
git clone https://github.com/lqian/light-LPR
cd light-LPR && mkdir build && cd build
cmake ../
make
</pre>

# installation commands for ARM Linux 
<pre>
git clone https://github.com/lqian/light-LPR
cd light-LPR && mkdir build && cd build
cmake ../ -DLIGHT_LPR_ARCH=arm
make
</pre>

# installation commands for Windows
- Download and install cmake 3.10 and above
- Download Visual Studio 2017 or Native Builder Tool for Visual Studio 2017, install c ++ compilation tool
- if you compile for x64, download  [opencv-3.4.2-install-win64.zip](https://pan.baidu.com/s/1CtabojjfEK-bK_XwfG9HTA), x86 archictecture [opencv-3.4.2-install-win32.zip](https://pan.baidu.com/s/1E7zhRsrrpc9JEhB_6gpehg)，and unzip the file.
- clone source code: [MNN](https://github.com/alibaba/MNN)
- download [flatc_windows_exe.zip](https://github.com/google/flatbuffers/releases/download/v1.11.0/flatc_windows_exe.zip)，and put the file to {MNN}/3rd_party/flatbuffers/tmp directory
- run powershell.exe as administrator，and then run the command: set-executionpolicy -executionpolicy unrestricted，select Y from prompt.
- comment the line `COMMAND powershell ${CMAKE_CURRENT_SOURCE_DIR}/schema/generate.ps1 -lazy` in CMakeLists.txt of MNN project，the line number is about 461.
<pre>
> cd MNN
> schema\enerate.ps1
> mkdir build 
> cd build
run x86 native tools command prompt for VS 2017 or x64 native tools command prompt for VS 2017
> cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ../
> nmake 
copy the MNN.dll、MNN.lib into light-LPR/lib directory
> cd light-LPR && mkdir build && cd build
> set OpenCV_DIR=/path/to/opencv-install/directory
> cmake -G "NMake Makefiles" ..
> nmake
</pre>

# test
`./examples/demo ../models/ [/path/to/a/image]`
This project passed the test on Fedora 29, CentOS 7.6, Windows 10 64-bit Home Edition, Ubuntu 18.04 mate for ARM platform


# reference
- [Alibaba MNN](https://github.com/alibaba/MNN)
- [License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9)
- [Caffe_OCR](https://github.com/senlinuc/caffe_ocr)
- [MNN MTCNN CPU OPENCL](https://github.com/liushuan/MNN-MTCNN-CPU-OPENCL)