#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/features/vfh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/find.hpp>
#include <fstream>

typedef std::pair<std::string, pcl::PointCloud<pcl::PointXYZ> > pcd_model;

/**
 * @brief loadTrainingData
 * @param base_dir
 * @param extension
 * @param models
 * @return
 */
int loadTrainingData(const boost::filesystem::path &base_dir, const std::string &extension,
                     std::vector<pcd_model> &models)
{
    if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir))
      return -1;

    for (boost::filesystem::directory_iterator it (base_dir); it != boost::filesystem::directory_iterator (); ++it)
    {
      if (boost::filesystem::is_directory (it->status ()))
      {
        std::stringstream ss;
        ss << it->path ();
        //pcl::console::print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str ().c_str (), (unsigned long)models.size ());
        loadTrainingData (it->path (), extension, models);
      }
      if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == extension)
      {
        pcd_model m;
        m.first = (base_dir / it->path ().filename ()).string();
        pcl::io::loadPCDFile(m.first,m.second);
        models.push_back(m);
      }
    }

    return 1;
}

/**
 * @brief computeVFH
 * @param models
 * @param features
 * @return
 */
int computeVFH(std::vector<pcd_model> &models,
               std::vector<pcl::PointCloud<pcl::VFHSignature308> > &features)
{
    for(int i=0;i<models.size();++i)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        *cloud=models.at(i).second;

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
//        std::cout<<"cvfh computing finished"<<std::endl;

        features.push_back(*_vfhs);
    }

    return 1;

}

/**
 * @brief saveFeaturesToPCD
 * @param models
 * @param features
 */
void saveFeaturesToPCD(std::vector<pcd_model> &models,
                       std::vector<pcl::PointCloud<pcl::VFHSignature308> > &features)
{
    for(int i=0;i<features.size();++i)
    {
        pcl::PointCloud<pcl::VFHSignature308> vfhs=features.at(i);
        std::string pcdName=models.at(i).first;
        boost::replace_last(pcdName,".pcd","_vfh.pcd");
        boost::replace_first(pcdName,"models","features");
        pcl::io::savePCDFileASCII(pcdName,vfhs);
    }
}

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
    boost::filesystem::path base_model_dir;
    if (argc < 2)
    {
      // PCL_ERROR ("Need at least two parameters! Syntax is: %s [model_directory] [options]\n", argv[0]);
      // return (-1);
      base_model_dir = "..//training_data//models";
    } else {
      base_model_dir = argv[1];
    }

    std::string extension (".pcd");
    transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);

    std::vector<pcd_model> models;
    if(-1==loadTrainingData(argv[1],extension,models))
    {
        PCL_ERROR("Load training models error!\n");
    }
    pcl::console::print_highlight("Load %d training models!\n", models.size());

    std::vector<pcl::PointCloud<pcl::VFHSignature308> > features;

    computeVFH(models,features);
    pcl::console::print_highlight("Compute vfh feature finished!\n", models.size());

    saveFeaturesToPCD(models,features);
    pcl::console::print_highlight("Save vfh features to pcd file!\n", models.size());

    return 1;
}
