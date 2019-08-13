# ---- 2019-08-13 支持黄牌识别的模型和车牌颜色识别的模型放出 ------
Light-LPR是一个瞄准可以在嵌入式设备、手机端和普通的x86平台上运行的车牌识别开源项目，旨在支持各种场景的车牌识别,车牌字符识别准确率超99.95%，综合识别准确率超过99%，支持目前国内所有的车牌识别，觉得好用的一定要加星哦。200星公布黄牌识别模型，400星公布新能源车牌模型。 
技术上采用MTCNN检测车牌和四个角点精确定位，并进行偏斜纠正，最后进行端到端识别车牌号码，使用MNN作为推理引擎。具有结构简单，灵活部署的特点，适应各类计算平台。

# 支持的车牌

| 车牌 | 检测 | 识别 |
| --------: | :-----: | :----: |
| 蓝   |  Y |  Y |
| 黄   |  Y |  Y |
| 新能源   |  E |  Y |
| 大型新能源   |  E |  Y |
| 教练车牌   |  E |  Y |
| 警牌   |  Y |  Y |
| 军牌   |  E |  Y |
| 双层军牌   |  - |  Y |
| 武警车牌   |  E |  Y |
| 双层武警牌照   |  - |  Y |
| 双层黄牌| - | Y |
| 港澳通行牌 | - | E | 
| 应急车牌 | - | E |
| 民航车牌 | - | E |
| 普通黑牌 | - | E |
| 使、领馆车牌 | - | E |
| 摩托车牌 | - | E |
| 低速农用车牌 | - | E |
| 临牌 | - | E |

备注： Y 支持，- 未知, E有限度支持

# 1080P图片识别基准性能

| 平台      | CPU型号    |  内存  | 平均识别时间(ms)  |
| :-------- | :-----    | :----:  | ----:  |
| X86  | i5-8265   |  -    | 451 |
| ARM  | A53       | 1G    | 1532|

#安装依赖
cmake >= 3.10.0
opencv >= 3.0.0
openmp

# x86平台Linux安装指令
<pre>
git clone https://github.com/lqian/light-LPR
cd light-LPR && mkdir build && cd build
cmake ../
make
</pre>

# ARM平台Linux安装指令
<pre>
git clone https://github.com/lqian/light-LPR
cd light-LPR && mkdir build && cd build
cmake ../ -DLIGHT_LPR_ARCH=arm
make
</pre>

# 运行测试
`./examples/demo ../models/ [/path/to/a/image]`
本项目在Fedora 29，Ubuntu 18.04 mate for ARM平台测试通过

# 未来优化
目前使用车牌的检测使用了[License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9)项目的MTCNN模型，在实际场景中误检测率比较高，抗干扰能力不强，检测耗时长。后期考虑采用MSSD或者YOLOV3等 one stage 的检测算法先检测出车牌，再进行偏斜纠正的方案，提高检测模块的性能。

# 参考和引用
- [Alibaba MNN](https://github.com/alibaba/MNN)
- [License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9](https://github.com/zhubenfu/License-Plate-Detect-Recognition-via-Deep-Neural-Networks-accuracy-up-to-99.9)
- [Caffe_OCR](https://github.com/senlinuc/caffe_ocr)
- [MNN MTCNN CPU OPENCL](https://github.com/liushuan/MNN-MTCNN-CPU-OPENCL)

# 其他
- 技术交流、数据交流和捐赠请联系作者或加QQ群，图像处理分析机器视觉 109128646[已满], light-LPR群号：813505078, 作者微信
- [![](https://raw.githubusercontent.com/lqian/light-LPR/master/109128646.png)](https://raw.githubusercontent.com/lqian/light-LPR/master/109128646.png) [![](https://raw.githubusercontent.com/lqian/light-LPR/master/light-LPR.png)](https://raw.githubusercontent.com/lqian/light-LPR/master/light-LPR.png) [![](https://raw.githubusercontent.com/lqian/light-LPR/master/contact.jpg)](https://raw.githubusercontent.com/lqian/light-LPR/master/contact.jpg)
