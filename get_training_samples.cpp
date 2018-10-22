#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include "kinect2grabber.h"
#include <stdio.h>
#include <linux/input.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#define DEV_PATH "/dev/input/event4"   //difference is possible

/**
// This programe aims to acquire the training
// point cloud (limited to the same target object)
**/
typedef pcl::PointXYZRGB pointT;
typedef pcl::PointCloud<pointT> pointCloudT;
typedef pointCloudT::Ptr pointCloudPtr;

/**
 * @brief filterCloud--limit the range of point cloud and remove outliers
 * @param cloud
 */
int filterCloud(pointCloudPtr & cloud)
{
  if(cloud->points.size()==0){
      std::cout<<"the input cloud is empty"<<std::endl;
      return -1;
    }

  std::cout<<"the point cloud size is: "<<cloud->points.size()<<std::endl;

  //remove NANs
//  for(pointCloudT::iterator it=cloud->begin();it<cloud->end();++it){
//      if(!pcl::isFinite(*it))
//        cloud->erase(it);
//    }

  //filter by passthrough
  const double zmin=0.8, zmax=1.1;
  const double xmin=-0.4,xmax=0.4;
  const double ymin=-0.4,ymax=0.4;
  pcl::PassThrough<pointT> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(zmin,zmax);
  pass.filter(*cloud);

  pass.setInputCloud(cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(xmin,xmax);
  pass.filter(*cloud);

  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(ymin,ymax);
  pass.filter(*cloud);

  float leafsize=0.001;
  pcl::VoxelGrid<pointT> grid;
  grid.setInputCloud(cloud);
  grid.setLeafSize(leafsize,leafsize,leafsize);
  grid.filter(*cloud);

  //outlier removal
  pcl::StatisticalOutlierRemoval<pointT> outlierRemv;
  outlierRemv.setInputCloud(cloud);
  outlierRemv.setMeanK(50);
  outlierRemv.setStddevMulThresh(0.8);
  outlierRemv.filter(*cloud);

  std::cout<<"after filtered, the point cloud size is: "<<cloud->points.size()<<std::endl;
  return 1;
}

/**
 * @brief segmentCloud--segment the cloud into clusters,
 *                      the plane will be removed
 * @param cloud
 * @param clusters
 */
int segmentCloud(pointCloudPtr & cloud,std::vector<pointCloudPtr> & clusters)
{
  //plane segment
  pcl::SACSegmentation<pointT> plane_seg;
  pcl::ModelCoefficients::Ptr coeffPtr(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  plane_seg.setOptimizeCoefficients(true);
  plane_seg.setModelType(pcl::SACMODEL_PLANE);
  plane_seg.setMethodType(pcl::SAC_RANSAC);
  plane_seg.setInputCloud(cloud);
  plane_seg.setDistanceThreshold(0.01);
  plane_seg.segment(*inliers,*coeffPtr);
  if(inliers->indices.size()==0)
    {
      std::cerr<<"WARNING: no plane extracted"<<std::endl;
    }
  else
    std::cout<<"plane extracted, point size: "<<inliers->indices.size()<<std::endl;

  //extract plane and scene-without-plane
  pointCloudT::Ptr scene_no_plane(new pointCloudT);
  pcl::ExtractIndices<pointT> extractor;
  extractor.setInputCloud(cloud);
  extractor.setIndices(inliers);
  extractor.setNegative(true);
  extractor.filter(*scene_no_plane);
  std::cout<<"scene extracted, point size: "<<scene_no_plane->points.size()<<std::endl;

  //euclidean cluster
  pcl::search::KdTree<pointT>::Ptr tree(new pcl::search::KdTree<pointT>);
  tree->setInputCloud(scene_no_plane);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pointT> clusterExtrac;
  clusterExtrac.setInputCloud(scene_no_plane);
  clusterExtrac.setSearchMethod(tree);
  clusterExtrac.setClusterTolerance(0.01);
  clusterExtrac.setMinClusterSize(600);
  clusterExtrac.setMaxClusterSize(25000);
  clusterExtrac.extract(cluster_indices);
  if(cluster_indices.size()==0)
    {
      std::cerr<<"ERROR: no cluster extracted"<<std::endl;
      return -1;
    }
  else
    std::cout<<"extracted "<<cluster_indices.size()<<" clusters"<<std::endl;

  //extract the clusters
  pcl::ExtractIndices<pointT> extc;
  extc.setInputCloud(scene_no_plane);
  extc.setNegative(false);

  clusters.clear();
  std::vector<pcl::PointIndices>::iterator iter;
  int idx=0;

  for(iter=cluster_indices.begin();iter!=cluster_indices.end();++iter)
    {
      pcl::PointIndices _indices=*iter;
      pcl::PointIndices::Ptr cluster=boost::make_shared<pcl::PointIndices>(_indices);
      //std::cout<<"    cluster #"<<++idx<<" size: "<<cluster->indices.size()<<std::endl;
      pointCloudT::Ptr tmpCloud(new pointCloudT);
      extc.setIndices(cluster);
      extc.filter(*tmpCloud);
      clusters.push_back(tmpCloud);
    }

  return 1;
}

bool key_pressed=false;

void * keyEventListener(void *)
{
    //get key char
    int keys_fd;
    char ret[2];
    struct input_event t;
    keys_fd=open(DEV_PATH, O_RDONLY);
    if(keys_fd <= 0)
    {
        printf("open /dev/input/event4 device error!\n");
        return NULL;
    }

    while(1)
    {
        if(read(keys_fd, &t, sizeof(t)) == sizeof(t))
        {
            if(t.type==EV_KEY)
            {
                if(t.code==31 && t.value==1)
                {
                    key_pressed=true;
                    if(t.code == KEY_ESC)
                        break;
                }
            }
        }
    }
    close(keys_fd);
}

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
    pointCloudPtr cloud(new pointCloudT);
    sensor::Kinect2Grabber grabber;
    grabber.start();
    grabber.getPointCloud(cloud);

    filterCloud(cloud);

    std::vector<pointCloudPtr> clusters;
    segmentCloud(cloud,clusters);


    //initialize key listening thread
    pthread_t keyEventThreadID;
    int ret=pthread_create(&keyEventThreadID,NULL,keyEventListener,NULL);
    if(ret!=0){
      std::cerr<<"create key listening thread "<<keyEventThreadID<<" failed!"<<std::endl;
      return -1;
    }
    pthread_detach(keyEventThreadID);

    std::stringstream ss;
    int saveFileIndex=0;
    std::string savePCDDir="get_train_sample//";
    pcl::visualization::PCLVisualizer viewer;
    viewer.addPointCloud(cloud,"object");
    while(!viewer.wasStopped())
    {
        //acquire and preprocess
        grabber.getPointCloud(cloud);
        filterCloud(cloud);
        segmentCloud(cloud,clusters);

        //filter by color red
        pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
        for(int i=0;i<clusters.size();++i)
        {
            pointCloudPtr cluster=clusters.at(i);
            target->clear();
            target->height=1;
            target->width=cluster->points.size();
            target->points.resize(target->width);

            long color_r=0;
            long color_g=0;
            long color_b=0;
            for(int j=0;j<cluster->points.size();++j)
            {
                target->points[j].x=cluster->points[j].x;
                target->points[j].y=cluster->points[j].y;
                target->points[j].z=cluster->points[j].z;
                color_r+=cluster->points[j].r;
                color_g+=cluster->points[j].g;
                color_b+=cluster->points[j].b;
            }
            if(color_g/cluster->points.size()>150)
                break;
        }

        if(target->points.size()==0)
            continue;

        if(key_pressed)
        {
            ss.clear();
            ss<<savePCDDir<<"stair_sample_"<<saveFileIndex++<<".pcd";
            std::string path;
            ss>>path;
            pcl::io::savePCDFileASCII(path,*target);//save training sample
            key_pressed=false;
        }

        //visualize
        viewer.updatePointCloud(target,"object");
        viewer.spinOnce();


    }

}
