## coffeeCup Demo

coffeeCup是一个基于点云的马克杯识别的演示程序，能够从kinect v2传感器获取RGBD信息，并计算为点云数据，通过对点云数据的处理和分割，从一堆物体中识别出马克杯对应的点云簇(point cloud cluster);

### Setup
硬件的setup包括点云采集传感器：kinect V2 （若采用其他深度摄像头，需要替换相关驱动和`kinect2grabber.cpp`采集程序）; 除了深度摄像头，还需要一个马克杯作为被测对象；

### Pipeline
* realtime recognize pipeline:

  `grab pointCloud` --> `filter` --> `segment` --> `computeFeature` --> `search`

* training pipeline:

  `prepare_data` --> `build_tree`

### source files
* realtime_recoginze_demo -- 实时识别马克杯
  * realtime_recognize_demo.cpp  
  * kinect2grabber.cpp


* get_train_samples -- 采集模型点云，采集的点云放在models文件夹中，源代码中可以定义模型的颜色过滤和命名格式，采集不同模型时需要修改；
  * get_train_samples.cpp
  * kinect2grabber.cpp


* prepare_data -- 数据准备的可执行文件, 主要完成训练数据加载、VFH特征计算、并将特征数据保存为pcd文件；在training_data的models文件夹中提供了3种模型点云，也可以用于训练，不必重新采集；
  * prepare_data.cpp


* build_tree -- 构建搜索树，主要完成加载模型数据直方图、数据转换、数据保存，以及构建tree index;
  * build_tree.cpp


* nearest_neighbors -- K邻近搜索，用于测试；
  * nearest_neighbors.cpp

### Test Data
* **training_data** -- 用于训练的数据

    * models 模型
        * bowl
        * cup
        * stair

    * features 特征
        * bowl
        * cup
        * stair

    * coffee_cup

* **test_data** -- 测试数据

### Install & Compile

#### Dependencies

* PCL 1.8

	安装PCL 1.8之前，首先需要安装PCL自身的一些依赖库：
	* boost

	* Eigen 3.0

	* Flann 1.7.1

	* VTK 5.6


```
sudo apt-get install libboost-all-dev libeigen3-dev libflann-dev
```

VTK的安装稍麻烦，请参考[https://blog.csdn.net/sinat_28752257/article/details/79169647?utm_source=blogkpcl7](#https://blog.csdn.net/sinat_28752257/article/details/79169647?utm_source=blogkpcl7)

开始安装PCL1.8

```
git clone https://github.com/PointCloudLibrary/pcl pcl-trunk
cd pcl-trunk && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
sudo make -j2 install
```
* freenect2

  参考:[ https://github.com/code-iai/iai_kinect2#install](https://github.com/code-iai/iai_kinect2#install)
  
* hdf5
	
	参考：[https://www.hdfgroup.org/downloads/](https://www.hdfgroup.org/downloads/)
	下载source code 完成编译安装
	
* opencv-2.4.13
	
	参考：[https://blog.csdn.net/u011557212/article/details/54706966?utm_source=itdadao&utm_medium=referral](https://blog.csdn.net/u011557212/article/details/54706966?utm_source=itdadao&utm_medium=referral)

#### Compile
```
git clone https://github.com/windz0629/coffeeCup.git
cd coffeeCup
mkdir build && cd build
cmake ..
make
```
#### run
train:
```
sh train.sh
```

test:
```
sh test.sh
```

首先运行train.sh脚本训练数据，然后运行test.sh运行测试demo

运行realtime\_recoginze\_demo：

```
cd build
./realtime_recoginze_demo
```
