## 安装依赖
cmake >= 3.10.0
opencv >= 3.0.0
openmp

## x86平台Linux安装指令
<pre>
git clone https://github.com/lqian/light-LPR
cd light-LPR && mkdir build && cd build
cmake ../
make
</pre>

## ARM平台Linux安装指令
<pre>
git clone https://github.com/lqian/light-LPR
cd light-LPR && mkdir build && cd build
cmake ../ -DLIGHT_LPR_ARCH=arm
make
</pre>

## Windows平台上安装指令
- 下载cmake 3.10以上版本并安装
- 首先下载Visual Studio 2017或者 Native Builder Tool for Visual Studio 2017，安装c++编译工具
- 如果编译64位系统，下载64位[opencv-3.4.2-install-win64.zip](https://pan.baidu.com/s/1CtabojjfEK-bK_XwfG9HTA), 32位系统则下载[opencv-3.4.2-install-win32.zip](https://pan.baidu.com/s/1E7zhRsrrpc9JEhB_6gpehg)，解压到任意目录
- 克隆[MNN](https://github.com/alibaba/MNN)的源码
- 下载[flatc_windows_exe.zip](https://github.com/google/flatbuffers/releases/download/v1.11.0/flatc_windows_exe.zip)，把flatc.exe可执行文件复制到{MNN}/3rd_party/flatbuffers/tmp目录下
- 以管理员权限打开powershell.exe，然后执行set-executionpolicy -executionpolicy unrestricted，提示选Y
- 注释掉MNN的源码目录中的CMakelist.txt中的`COMMAND powershell ${CMAKE_CURRENT_SOURCE_DIR}/schema/generate.ps1 -lazy`这行，大约在461行
<pre>
> cd MNN
> schema\enerate.ps1
> mkdir build 
> cd build
按win键，根据需要，搜索x86 native tools command prompt for VS 2017 或者x64 native tools command prompt for VS 2017
> cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ../
> nmake 
把编译成功的MNN.dll、MNN.lib文件复制到light-LPR项目的lib目录下
> cd light-LPR && mkdir build && cd build
> set OpenCV_DIR=/path/to/opencv-install/directory
> cmake -G "NMake Makefiles" ..
> nmake
</pre>

## 运行测试
`./examples/demo ../models/ [/path/to/a/image]`
本项目在Fedora 29，CentOS 7.6, Windows 10 64位家庭版，Ubuntu 18.04 mate for ARM平台测试通过


## 参考和引用
- [Alibaba MNN](https://github.com/alibaba/MNN)
- [License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9)
- [Caffe_OCR](https://github.com/senlinuc/caffe_ocr)
- [MNN MTCNN CPU OPENCL](https://github.com/liushuan/MNN-MTCNN-CPU-OPENCL)