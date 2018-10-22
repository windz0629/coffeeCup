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
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include "kinect2grabber.h"
#define WITH_KINECT false //switch between test mode and online mode

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
  clusterExtrac.setMinClusterSize(200);
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
void computeVFHFeatures(pointCloudPtr & cloud,
                        pcl::PointCloud<pcl::VFHSignature308> & feature)
{
    //calculate normals
    pcl::PointCloud<pcl::PointNormal>::Ptr point_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::NormalEstimationOMP<pointT,pcl::PointNormal> normalEst_;
    normalEst_.setInputCloud(cloud);
    pcl::search::KdTree<pointT>::Ptr tree_norm (new pcl::search::KdTree<pointT> ());
    normalEst_.setSearchMethod(tree_norm);
    normalEst_.setKSearch(10);
    normalEst_.compute(*point_normals);

    pcl::VFHEstimation<pointT, pcl::PointNormal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud (cloud);
    vfh.setInputNormals (point_normals);
    pcl::search::KdTree<pointT>::Ptr _tree_vfh (new pcl::search::KdTree<pointT> ());
    vfh.setSearchMethod (_tree_vfh);
//    pcl::PointCloud<pcl::VFHSignature308>::Ptr _vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
    vfh.compute (feature);

//        pcl::CVFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::VFHSignature308> cvfh;
//        cvfh.setInputCloud(cloud);
//        cvfh.setInputNormals(point_normals);
//        cvfh.setSearchSurface(cloud);
//        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_vfh (new pcl::search::KdTree<pcl::PointXYZ> ());
//        cvfh.setSearchMethod(tree_vfh);
//        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
//        cvfh.setEPSAngleThreshold(10.0/180*M_PI);
//        cvfh.setCurvatureThreshold(0.003);
//        cvfh.setNormalizeBins(true);
//        cvfh.compute(*vfhs);
}

typedef std::pair<std::string, std::vector<float> > vfh_model;

/** \brief Loads an n-D histogram file as a VFH signature
  * \param path the input file name
  * \param vfh the resultant VFH model
  */
bool
loadHist (const boost::filesystem::path &path, vfh_model &vfh)
{
  int vfh_idx;
  // Load the file as a PCD
  try
  {
    pcl::PCLPointCloud2 cloud;
    int version;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    pcl::PCDReader r;
    int type; unsigned int idx;
    r.readHeader (path.string (), cloud, origin, orientation, version, type, idx);

    vfh_idx = pcl::getFieldIndex (cloud, "vfh");
    if (vfh_idx == -1)
      return (false);
    if ((int)cloud.width * cloud.height != 1)
      return (false);
  }
  catch (const pcl::InvalidConversionException&)
  {
    return (false);
  }

  // Treat the VFH signature as a single Point Cloud
  pcl::PointCloud <pcl::VFHSignature308> point;
  pcl::io::loadPCDFile (path.string (), point);
  vfh.second.resize (308);

  std::vector <pcl::PCLPointField> fields;
  getFieldIndex (point, "vfh", fields);

  for (size_t i = 0; i < fields[vfh_idx].count; ++i)
  {
    vfh.second[i] = point.points[0].histogram[i];
  }
  vfh.first = path.string ();
  return (true);
}

bool
loadHist (const pcl::PointCloud<pcl::VFHSignature308> feature, vfh_model &vfh)
{
  int vfh_idx=0;

  // Treat the VFH signature as a single Point Cloud
  vfh.second.resize (308);
  std::vector <pcl::PCLPointField> fields;
  getFieldIndex (feature, "vfh", fields);

  for (size_t i = 0; i < fields[vfh_idx].count; ++i)
  {
    vfh.second[i] = feature.points[0].histogram[i];
  }
  vfh.first = "";
  return (true);
}

/** \brief Search for the closest k neighbors
  * \param index the tree
  * \param model the query model
  * \param k the number of neighbors to search for
  * \param indices the resultant neighbor indices
  * \param distances the resultant neighbor distances
  */
inline void
nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model,
                int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
  // Query point
  flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
  memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));

  indices = flann::Matrix<int>(new int[k], 1, k);
  distances = flann::Matrix<float>(new float[k], 1, k);
  index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
  delete[] p.ptr ();
}

/** \brief Load the list of file model names from an ASCII file
  * \param models the resultant list of model name
  * \param filename the input file name
  */
bool
loadFileList (std::vector<vfh_model> &models, const std::string &filename)
{
  ifstream fs;
  fs.open (filename.c_str ());
  if (!fs.is_open () || fs.fail ())
    return (false);

  std::string line;
  while (!fs.eof ())
  {
    getline (fs, line);
    if (line.empty ())
      continue;
    vfh_model m;
    m.first = line;
    models.push_back (m);
  }
  fs.close ();
  return (true);
}

/**
 * @brief searchMatchedHist
 * @param [in]feature
 * @param [out]distance
 * @param [out]object_type
 * @return
 */
int searchMatchedHist(pcl::PointCloud<pcl::VFHSignature308> feature, float & distance, std::string & object_type)
{
    std::string extension (".pcd");
    std::transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);

    // Load the test histogram
    vfh_model histogram;
    if (!loadHist (feature,histogram))
    {
      pcl::console::print_error ("Cannot load feature\n");
      return -1;
    }

    float thresh=80;
    int k=13;

    std::string kdtree_idx_file_name = "kdtree.idx";
    std::string training_data_h5_file_name = "training_data.h5";
    std::string training_data_list_file_name = "training_data.list";

    std::vector<vfh_model> models;
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;
    flann::Matrix<float> data;
    // Check if the data has already been saved to disk
    if (!boost::filesystem::exists ("training_data.h5") || !boost::filesystem::exists ("training_data.list"))
    {
      pcl::console::print_error ("Could not find training data models files %s and %s!\n",
          training_data_h5_file_name.c_str (), training_data_list_file_name.c_str ());
      return -1;
    }
    else
    {
      loadFileList (models, training_data_list_file_name);
      flann::load_from_file (data, training_data_h5_file_name, "training_data");
      pcl::console::print_highlight ("Training data found. Loaded %d VFH models from %s/%s.\n",
          (int)data.rows, training_data_h5_file_name.c_str (), training_data_list_file_name.c_str ());
    }

    // Check if the tree index has already been saved to disk
    if (!boost::filesystem::exists (kdtree_idx_file_name))
    {
      pcl::console::print_error ("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str ());
      return -1;
    }
    else
    {
      flann::Index<flann::ChiSquareDistance<float> > index (data, flann::SavedIndexParams ("kdtree.idx"));
      index.buildIndex ();
      nearestKSearch (index, histogram, k, k_indices, k_distances);
    }

    // Output the results on screen
    std::string matchedFeatureName=models.at (k_indices[0][0]).first;
    distance=k_distances[0][0];
    if(distance>thresh)
    {
        object_type="unknown";
        return 1;
    }
    size_t pos=matchedFeatureName.find_last_of("/");
    std::string name=matchedFeatureName.substr(pos+1);
    std::vector<std::string> name_elements;
    boost::split( name_elements, name, boost::is_any_of( "_" ), boost::token_compress_on );
    if(name_elements.size()>0)
    {
        object_type=name_elements[0];
        return 1;
    }
    else
        return -1;
}

static inline bool cluster_comparator(pointCloudPtr p1, pointCloudPtr p2)
{
    return p1->points.size()>p2->points.size();
}

sensor::Kinect2Grabber grabber;
std::string offline_scene_path="//home//wangy//develop//perception_ws//projects//build//coffeeCup//test_data//cup_test_sample.pcd";
void getCloud(pointCloudPtr & cloud)
{
    if(WITH_KINECT){
        grabber.getPointCloud(cloud);
    }else{
        pcl::io::loadPCDFile(offline_scene_path,*cloud);
    }
}

int main(int argc, char** argv)
{
    std::string search_type="cup";
    if(argc>1)
      search_type=argv[1];

    pointCloudPtr scene(new pointCloudT);

    if(WITH_KINECT)
        grabber.start();
    getCloud(scene);
    filterCloud(scene);
    std::vector<pointCloudPtr> clusters;
    segmentCloud(scene,clusters);

    int maxClusterNo=6;
    int matchedIndex=-1;
    float distance;
    std::string object_type="null";

    pcl::visualization::PCLVisualizer viewer("scene");
    viewer.addPointCloud(scene,"scene");
    pcl::visualization::PCLVisualizer viewer2("clusters");

    while(!viewer.wasStopped())
    {
        getCloud(scene);
        filterCloud(scene);
        std::vector<pointCloudPtr> clusters;
        segmentCloud(scene,clusters);

        //choose the largest at most 5 clusters
        std::sort(clusters.begin(),clusters.end(),cluster_comparator);
        if(clusters.size()<maxClusterNo)
            maxClusterNo=clusters.size();

        for(int i=0;i<maxClusterNo;++i)
        {
            pointCloudPtr object=clusters.at(i);
            //compute the vfh of object
            pcl::PointCloud<pcl::VFHSignature308> feature;
            computeVFHFeatures(object,feature);

            //nearest search the target
            if(-1==searchMatchedHist(feature,distance,object_type))
            {
                std::cerr<<"search match object failed with errors"<<std::endl;
            }

            //if found, break;
            if(object_type==search_type)
            {
                matchedIndex=i;
                std::cerr<<"====found match object: "<<object_type<<"===="<<std::endl;
                std::cerr<<"distance: "<<distance<<std::endl;
                std::cerr<<"============================================="<<std::endl;
                break;
            }
        }

        std::stringstream ss;
        std::string cloud_name;

        viewer2.removeAllPointClouds();
        viewer2.removeAllShapes();
        for(int i=0;i<maxClusterNo;++i){
            ss.clear();
            ss<<"cluster_"<<i;
            ss>>cloud_name;
            pcl::visualization::PointCloudColorHandlerCustom<pointT> tmp_colorH(clusters.at(i),
                                                                                rand()%255,
                                                                                rand()%255,
                                                                                rand()%255);
            viewer2.addPointCloud(clusters.at(i),tmp_colorH,cloud_name);
        }

        std::string displayInfo="";
        if(matchedIndex!=-1)
        {
            displayInfo="found match objects: "+search_type+"\r\n distance: "+boost::lexical_cast<std::string>(distance);
            viewer2.addText(displayInfo,30,80,30,0,1,0,"displayInfo",0);
            pointT minP,maxP;
            pcl::getMinMax3D(*(clusters.at(matchedIndex)),minP,maxP);
            pointT p1,p2,p3,p4,p5,p6;
            p1.x=minP.x;p1.y=minP.y;p1.z=maxP.z;
            p2.x=minP.x;p2.y=maxP.y;p2.z=maxP.z;
            p3.x=minP.x;p3.y=maxP.y;p3.z=minP.z;
            p4.x=maxP.x;p4.y=minP.y;p4.z=maxP.z;
            p5.x=maxP.x;p5.y=minP.y;p5.z=minP.z;
            p6.x=maxP.x;p6.y=maxP.y;p6.z=minP.z;
            viewer2.addLine(minP,p1,0,1,0,"l1",0);
            viewer2.addLine(minP,p3,0,1,0,"l2",0);
            viewer2.addLine(minP,p5,0,1,0,"l3",0);
            viewer2.addLine(p1,p2,0,1,0,"l4",0);
            viewer2.addLine(p1,p4,0,1,0,"l5",0);
            viewer2.addLine(p2,p3,0,1,0,"l6",0);
            viewer2.addLine(p2,maxP,0,1,0,"l7",0);
            viewer2.addLine(p3,p6,0,1,0,"l8",0);
            viewer2.addLine(p6,maxP,0,1,0,"l9",0);
            viewer2.addLine(p6,p5,0,1,0,"l10",0);
            viewer2.addLine(p5,p4,0,1,0,"l11",0);
            viewer2.addLine(p4,maxP,0,1,0,"l12",0);
        }
        else
        {
            displayInfo="failed to match target: "+search_type;
            viewer2.addText(displayInfo,30,80,30,1,0,0,"displayInfo",0);
        }
        matchedIndex=-1;

        viewer2.spinOnce();

        viewer.updatePointCloud(scene,"scene");
        viewer.spinOnce();

        usleep(1000*20);
    }

}
