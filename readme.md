# FinTechathon2020微众银行第二届金融科技高校技术大赛——人脸对抗攻击

团队名称：一辆二手迈巴赫+.+

[toc]

## 介绍

文件结构如下：

--attack_code

​		--function.py;  main.py;  demo_test.py;  demo_val.py

​		--backbone

​				--Backbone_IR_152_Epoch_37.pth;  insight-face-v3.pt;  iresnet100.pth;  iresnet50.pth;  iresnet34.pth;  iresnet.py;  model_irse.py;  models_irse101.py

​		--images

​				--input

​						--test

​						--val

​				--output	

​						--test

​						--val

backbone文件夹: 存放backbone的权重与网络结构,百度网盘链接为： https://pan.baidu.com/s/1DLPg98ANMajnKiROgzULFA 提取码: fhaf 

images文件夹: 存放输入图片与经过处理后的攻击图片,百度网盘链接为： https://pan.baidu.com/s/1VPdNjoL5ENvGThEq2IeefA 提取码: p8fa 

main.py: 使用一个类集成了攻击模型与攻击方法

demo_test.py: 攻击test数据集图片的脚本文件

demo_val.py: 攻击val数据集图片的脚本文件



## 环境配置与运行

环境配置：pip -r install requirements.txt

运行：python demo_test.py / demo_val.py



## 选择攻击模型与攻击方法

### 选择攻击模型

打开demo_test.py/demo_val.py,可以修改model_names列表来选择攻击模型，一共有5个模型，分别为"irse101","ir152","ir100","ir50","ir34"，模型精度为irse101>ir100>ir50>ir34>ir152。

### 选择攻击方法

一共四种攻击方式："si","di","ti","mi"，可以改变attack_methods变量的值来选择。经过测试，精度最高的攻击方式是"sidi"方法，这也是我们在test数据集中使用的攻击方法。攻击精度较高、同时攻击速度较快的方式是"timi"方法，由于其攻击速度较快，我们使用这种方法在val数据集中测试模型的精度与并进行超参数的调优。

### 选择数据集与指定攻击对象

input_path变量存放输入数据集的路径，既可以是test数据集的路径，也可以是val数据集的路径，同时需要改变对应dataset变量的值

output_path变量则存放输出攻击样本数据的路径

pair_path变量存放pair.txt的路径，该.txt文件指定了数据集中的图片的攻击目标

### 调整超参数

ts超参数指定攻击的阈值。当目标图片feature与攻击图片feature的cos相似度达到ts时，网络停止攻击

max_step超参数指定攻击的最大步数，考虑到攻击时间的限制，我们限制最大攻击步数为300，超过300步之后cos相似度若仍没有达到ts值，将自动攻击下一张图片。

alpha超参数指定攻击的距离，一般设置为0.5，值太大会使攻击不收敛，值太小则会导致攻击速度较慢

## 复现最佳分数22.64

我们使用5模型叠加跑出了一份Basement，之后利用四模型改变权重、三模型改变权重跑出了共11份结果，我们将剩下的11份结果按照Loss的高低进行了最优结果选择产生Supplement，之后我们找到Basement中未被攻破（没有达到阈值）的图片，观察在Supplement中是否这些原来未被攻破的图片现在被攻破，这样的图片一共有12张，因为运算时间过长（Nvidia-Tesla V100 32小时），我们将这12张图片提供在我们的源代码当中（extra12.zip)，只需要运行demo_test.py之后将这12张图片替换最后的结果里对应的图片，即可复现我们的最佳攻击，实际过程中分数由于DIM的随机性可能会更好或者更差，但误差应该不会超过0.1，也就是最终的分数会在22.54-22.74之间，如果没有达到这个分数请联系我们进行源码的修正，我们的代码经过二次整理，完全按照Github开源的风格，所有的攻击方法集成到了类中，方便评委们复现与评测，如果使用过程中出现无法复现的情况，请及时联系我们！

## 联系方式

我们是哈尔滨工业大学（深圳）的三名研二学生，如果想了解具体的实验数据与实现细节，请联系我们的邮箱：

左琪：hit_zq@qq.com

乔长浩：383794668@qq.com

瞿豪豪：1063510670@qq.com