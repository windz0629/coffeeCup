## coffeeCup Demo

coffeeCup是一个基于点云的马克杯识别的演示程序，能够从kinect v2传感器获取RGBD信息，并计算为点云数据，通过对点云数据的处理和分割，识别出马克杯对应的点云簇(point cloud cluster);

### Setup
硬件的setup包括点云采集传感器：kinect V2 （若采用其他深度摄像头，需要替换相关驱动和`kinect2grabber.cpp`采集程序）; 除了深度摄像头，还需要一个马克杯作为被测对象；

### Pipeline
`grab pointCloud` --> `filter` --> `segment` --> `computeFeature` --> `search`

### source files architecture
* realtime_recoginze_demo -- 实时识别马克杯的可执行文件，由
realtime_recognize_demo.cpp 和 kinect2grabber.cpp编译；

* get_train_samples是 -- 采集杯子点云的可执行文件，由get_train_samples.cpp和kinect2grabber.cpp编译；

* prepare_data -- 数据准备的可执行文件，由prepare_data.cpp编译；主要完成训练数据加载、VFH特征计算、并将特征数据保存为pcd文件；

* build_tree -- 构建搜索树，主要完成加载模型数据直方图、数据转换、数据保存，以及构建tree index;

* nearest_neighbors -- K邻近搜索，用于测试；

### Test Data
**training_data** -- 用于训练的数据

    * models 模型
        * bowl
        * cup
        * stair

    * features 特征
        * bowl
        * cup
        * stair

    * coffee_cup

**test_data** -- 测试数据

### Install & Compile

#### Dependencies

* PCL 1.8
```
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl
sudo apt-get update
sudo apt-get install libpcl-all
```
* freenect2

  [install ref: https://github.com/code-iai/iai_kinect2#install](#https://github.com/code-iai/iai_kinect2#install)
* hdf5
* flann
* opencv-2.4.13
* boost

#### Compile
```
cd coffeeCup
mkdir build && cd build
cmake ..
make
```
#### run
```
sh train.sh
sh test.sh
```
首先运行train.sh脚本训练数据，然后运行test.sh运行测试demo
