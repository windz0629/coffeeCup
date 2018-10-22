#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/correspondence.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features//fpfh_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <boost/algorithm/string/replace.hpp>

using namespace std;
typedef pcl::PointXYZ pointT;
typedef pcl::PointCloud<pointT> pointCloudT;
typedef pointCloudT::Ptr pointCloudPtr;
typedef pcl::Normal normalT;
typedef pcl::PointCloud<normalT> normalCloudT;

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
  const double zmin=0, zmax=1.0;
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
int segmentCloud(pointCloudPtr & cloud,vector<pointCloudPtr> & clusters)
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

/**
 * @brief computeVFHFeatures
 * @param clusters
 * @param features
 */
void computeVFHFeatures(vector<pointCloudPtr> & clusters,
                        vector<pcl::PointCloud<pcl::VFHSignature308> > & features)
{
    for(int i=0;i<clusters.size();++i)
    {
        pointCloudPtr cloud=clusters.at(i);

        //calculate normals
        pcl::PointCloud<pcl::PointNormal>::Ptr point_normals(new pcl::PointCloud<pcl::PointNormal>);
        pcl::NormalEstimationOMP<pcl::PointXYZ,pcl::PointNormal> normalEst_;
        normalEst_.setInputCloud(cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_norm (new pcl::search::KdTree<pcl::PointXYZ> ());
        normalEst_.setSearchMethod(tree_norm);
        normalEst_.setKSearch(10);
        normalEst_.compute(*point_normals);

        pcl::VFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::VFHSignature308> vfh;
        vfh.setInputCloud (cloud);
        vfh.setInputNormals (point_normals);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr _tree_vfh (new pcl::search::KdTree<pcl::PointXYZ> ());
        vfh.setSearchMethod (_tree_vfh);
        pcl::PointCloud<pcl::VFHSignature308>::Ptr _vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
        vfh.compute (*_vfhs);

        pcl::CVFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::VFHSignature308> cvfh;
        cvfh.setInputCloud(cloud);
        cvfh.setInputNormals(point_normals);
        cvfh.setSearchSurface(cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_vfh (new pcl::search::KdTree<pcl::PointXYZ> ());
        cvfh.setSearchMethod(tree_vfh);
        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
        cvfh.setEPSAngleThreshold(10.0/180*M_PI);
        cvfh.setCurvatureThreshold(0.003);
        cvfh.setNormalizeBins(true);
        cvfh.compute(*vfhs);

        features.push_back(*_vfhs);
    }
}

int main(int argc, char** argv)
{
    pointCloudPtr cloud(new pointCloudT);
    vector<pointCloudPtr> clusters;
    vector<pcl::PointCloud<pcl::VFHSignature308> > features;
    std::string fileName;
    if(argc<2)
        fileName="//home//wangy//develop//perception_ws//projects//build//coffeeCup//test_data//cup_test_sample.pcd";
    else
        fileName=argv[1];

    pcl::io::loadPCDFile(fileName,*cloud);
    filterCloud(cloud);
//    segmentCloud(cloud,clusters);
    clusters.push_back(cloud);
    computeVFHFeatures(clusters,features);

    std::stringstream ss;
    string path="//home//wangy//develop//perception_ws//projects//build//coffeeCup//test_data//";
    for(int i=0;i<clusters.size();++i)
    {
        ss.clear();
        ss<<path<<"cluster_"<<i<<".pcd";;
        string cluster_name;
        ss>>cluster_name;
        std::cout<<"saved "<<cluster_name<<std::endl;
        pcl::io::savePCDFileASCII(cluster_name,*(clusters.at(i)));

        boost::replace_last(cluster_name,".pcd","_vfh.pcd");
        pcl::io::savePCDFileASCII(cluster_name,features.at(i));
        std::cout<<"saved "<<cluster_name<<std::endl;
    }

    pcl::visualization::PCLVisualizer viewer("point cloud");
    viewer.addPointCloud(cloud,"scene_cloud");
    pcl::visualization::PCLVisualizer viewer2("segment clusters");

    while(!viewer.wasStopped()){

        viewer.updatePointCloud(cloud,"scene_cloud");

        std::stringstream ss;
        std::string cloud_name;

        viewer2.removeAllPointClouds();
        for(int i=0;i<clusters.size();++i){
            ss.clear();
            ss<<"cluster_"<<i;
            ss>>cloud_name;
            pcl::visualization::PointCloudColorHandlerCustom<pointT> tmp_colorH(clusters.at(i),
                                                                                rand()%255,
                                                                                rand()%255,
                                                                                rand()%255);
            viewer2.addPointCloud(
                  clusters.at(i),
                  tmp_colorH,
                  cloud_name);
          }

        viewer.spinOnce();
        viewer2.spinOnce();
      }
}
