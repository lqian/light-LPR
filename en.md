Light-LPR is an open source project aimed at license plate recognition that can run on embedded devices, mobile phones, and x86 platforms. It aims to support license plate recognition in various scenarios. The accuracy rate of license plate character recognition exceeds 99.95%, and the comprehensive recognition accuracy rate exceeds 99%. The project supports multi-lingual and multi-country license plates.

 [ About ](README.md) | [ English ](en.md) | [ 中文 ](cn-zh.md) | [ 中文繁体 ](cn-tw.md)| [ 한국어 ](kr.md) 

## Change Log 
- 2021-06 LLPR-313E is released, supporting scene recognition of charging piles and underground garages, and can recognize up to 3 license plates at the same time. Support license plate recognition in China, Taiwan and South Korea
- 2021-04 LLPR-CN-330E is released to support vehicle-mounted inspection license plate recognition; light-lpr-api 1.4.5 is released, and the device management interface is opened
- 2021-02 The fourth-generation recognition engine light-lpr-pro 4.0 is released, which fully supports the recognition of dual-line license plates in mainland China and improves the recognition of small-size license plates
- 2021-01 hardware recognition engine supports FTP upload and HTTP upload, [light-lpr-httpdemo-0.0.1](https://github.com/lqian/light-lpr-httpdemo) is released.
- 2020-11 replease light-lpr-api 1.2.0 for development with LightLPR device, support c/c++, c#, Java, Android programming.
- 2020-10 release light-lpr-tool for LightLPR device management, and test on road. [test video1](https://pan.baidu.com/s/16D2S6StjKsv879nMFSZAmQ) code: ewqb , [test video1](https://pan.baidu.com/s/1wV_agW71bthTpzhxKLf6cA) code: cun3
- 2020-08 release LightLPR engine 3.0 at Hisilcon 3516CV500 and 3516DV300
- 2020-06 release LightLPR engine 3.0
- 2020-04 full Support for 2019 new license plates of the Republic of South Korea
- 2020-04 LightLPR apk support device which run android 5, android 6, android 7, android 8, android 9, android 10 
- 2020-01 support South Korean License Plate Recognization
- 2019-12 Support Taiwan license plate recognition
- 2019-11 Open source Windows x86 compilation method, providing Opencv-3.4.2 Windows pre-compiled package
- 2019-08 Open source model supporting yellow card recognition and license plate color recognition model
- 2019-07 Open source supports the blue license plate recognition model in the People's Republic of China

## License plate recognization benchmark for 1080P image

|       | CPU     |  Memory  | average cost of community version (ms)   |  average cost of Commercial version(ms) |
| :-------- | :-----    | :----:  | ----:  | ----:  |
| X86  | i5-8265   |  8G    | 451 | < 50  |
| ARM  | A53       | 1G    | 1532| < 160 |
| Huwei P20 pro| ... | 4G | - |  < 100 |
| LLPR-320E | ... |  | - |  < 45 (NPU support) |
| LLPR-310E | ... |  | - | < 85 (NPU support) |

## License
LGPL

## [install](install_en.md)

## Others
- Contact： link.com@yeah.net, WeChat: +86 18010870244, Skype: +86 18010870244