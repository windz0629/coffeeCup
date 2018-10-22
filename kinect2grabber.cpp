#include "kinect2grabber.h"

namespace sensor {

  Kinect2Grabber::Kinect2Grabber (ProcessorType processor_type, bool mirror)
      : processor_type_(processor_type), mirror_(mirror),
        listener_(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth),
        undistorted_(512, 424, 4), registered_(512, 424, 4), big_depth_(1920, 1082, 4),
        qnan_(std::numeric_limits<float>::quiet_NaN())
  { }

  Kinect2Grabber::~Kinect2Grabber ()
  {
    shutDown();
  }


  IrCameraParams
  Kinect2Grabber::getIrParameters ()
  {
    return (device_->getIrCameraParams ());
  }

  ColorCameraParams
  Kinect2Grabber::getRgbParameters ()
  {
    return (device_->getColorCameraParams ());
  }

  void
  Kinect2Grabber::disableLog ()
  {
    logger_ = libfreenect2::getGlobalLogger();
    libfreenect2::setGlobalLogger(nullptr);
  }

  void
  Kinect2Grabber::enableLog ()
  {
    libfreenect2::setGlobalLogger(logger_);
  }

  void
  Kinect2Grabber::printParameters ()
  {
    libfreenect2::Freenect2Device::ColorCameraParams color_camera = getRgbParameters();
    std::cout << "Coloar Camera Params: fx=" << color_camera.fx << ", fy=" << color_camera.fy
              << ", cx=" << color_camera.cx << ", cy=" << color_camera.cy << std::endl;

    libfreenect2::Freenect2Device::IrCameraParams ir_camera = getIrParameters();
    std::cout << "IR Camera Params: fx=" << ir_camera.fx << ", fy=" << ir_camera.fy
              << ", cx=" << ir_camera.cx << ", cy=" << ir_camera.cy
              << ", k1=" << ir_camera.k1 << ", k2=" << ir_camera.k2 << ", k3=" << ir_camera.k3
              << ", p1=" << ir_camera.p1 << ", p2=" << ir_camera.p2 << std::endl;
  }

  void
  Kinect2Grabber::storeParameters ()
  {
    ColorCameraParams color_camera = getRgbParameters();
    IrCameraParams ir_camera = getIrParameters();

    cv::Mat rgb = (cv::Mat_<float>(3,3) << color_camera.fx, 0, color_camera.cx, 0, color_camera.fy, color_camera.cy, 0, 0, 1);
    cv::Mat depth = (cv::Mat_<float>(3,3) << ir_camera.fx, 0, ir_camera.cx, 0, ir_camera.fy, ir_camera.cy, 0, 0, 1);
    cv::Mat depth_dist = (cv::Mat_<float>(1,5) << ir_camera.k1, ir_camera.k2, ir_camera.p1, ir_camera.p2, ir_camera.k3);
    std::cout << "Storing " << serial_number_ << std::endl;
    cv::FileStorage file_storage ("calib_" + serial_number_ + ".yml", cv::FileStorage::WRITE);

    file_storage << "CcameraMatrix" << rgb;
    file_storage << "DcameraMatrix" << depth << "distCoeffs" << depth_dist;

    file_storage.release();
  }

  void
  Kinect2Grabber::start (std::string serial_number)
  {
    if (freenect2_.enumerateDevices() == 0)
    {
      std::cout << "Warning! No Kinect2 device is connected!" << std::endl;
      return;
    }

    switch (processor_type_)
    {
      case CPU:
        std::cout << "With CPU depth processing." << std::endl;
        if (serial_number.empty())
          device_ = freenect2_.openDefaultDevice (new libfreenect2::CpuPacketPipeline ());
        else
          device_ = freenect2_.openDevice (serial_number, new libfreenect2::CpuPacketPipeline ());

        break;

#ifdef WITH_OPENCL
      case OPENCL:
        std::cout << "With OpenCL depth processing." << std::endl;
        if (serial_number.empty ())
          device_ = freenect2_.openDefaultDevice(new libfreenect2::OpenCLPacketPipeline());
        else
          device_ = freenect2_.openDevice(serial_number, new libfreenect2::OpenCLPacketPipeline());
        break;
#endif

      case OPENGL:
        std::cout << "With OpenGL depth processing." << std::endl;
        if (serial_number.empty())
          device_ = freenect2_.openDefaultDevice (new libfreenect2::OpenGLPacketPipeline ());
        else
          device_ = freenect2_.openDevice (serial_number, new libfreenect2::OpenGLPacketPipeline ());
        break;

#ifdef WITH_CUDA
      case CUDA:
        std::cout << "With CUDA depth processing." << std::endl;
        if(serial_number.empty())
          device_ = freenect2_.openDefaultDevice(new libfreenect2::CudaPacketPipeline());
        else
          device_ = freenect2_.openDevice(serial_number, new libfreenect2::CudaPacketPipeline());
        break;
#endif

      default:
        std::cout << "With OpenGL depth processing." << std::endl;
        if (serial_number.empty())
          device_ = freenect2_.openDefaultDevice (new libfreenect2::OpenGLPacketPipeline ());
        else
          device_ = freenect2_.openDevice (serial_number, new libfreenect2::OpenGLPacketPipeline ());
        break;
    }

    if (!device_)
    {
      std::cout << "Warning! The device is not opened." << std::endl;
      return;
    }

    serial_number_ = device_->getSerialNumber ();

    device_->setColorFrameListener (&listener_);
    device_->setIrAndDepthFrameListener (&listener_);
    device_->start();
    std::cout << "[Info] Device with serial number " << serial_number_ << " has started." << std::endl;

    logger_ = libfreenect2::getGlobalLogger();

    registration_ = new libfreenect2::Registration(device_->getIrCameraParams(), device_->getColorCameraParams());

    prepareMake3D (device_->getIrCameraParams());
  }

  bool
  Kinect2Grabber::isOpen ()
  {
    if (device_)
      return (true);
    else
      return (false);
  }

  void
  Kinect2Grabber::shutDown ()
  {
    if (device_)
    {
      device_->stop();
      device_->close();
    }
  }

  void
  Kinect2Grabber::getPointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud)
  {
    listener_.waitForNewFrame(frames_);
    libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
    libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
    registration_->apply (rgb, depth, &undistorted_, &registered_, true, &big_depth_, color_depth_map_);

    const std::size_t width = undistorted_.width;
    const std::size_t height = undistorted_.height;
    if (point_cloud->size() != width * height)
      point_cloud->resize(width * height);

    updatePointCloud(point_cloud);

    listener_.release(frames_);
  }

  void
  Kinect2Grabber::updatePointCloud (pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud)
  {
    const std::size_t width = undistorted_.width;
    const std::size_t height = undistorted_.height;
    cv::Mat tmp_it_depth_0(height, width, CV_8UC4, undistorted_.data);
    cv::Mat tmp_it_rgb_0(height, width, CV_8UC4, registered_.data);

    if (mirror_ == true)
    {
      cv::flip (tmp_it_depth_0,tmp_it_depth_0,1);
      cv::flip (tmp_it_rgb_0,tmp_it_rgb_0,1);
    }

    const float * it_depth_0 = (float *) tmp_it_depth_0.ptr();
    const char * it_rgb_0 = (char *) tmp_it_rgb_0.ptr();

    pcl::PointXYZRGB * it_point_cloud = &point_cloud->points[0];
    bool is_dense = true;

    for (std::size_t idx_row = 0; idx_row < height; ++idx_row)
    {
      const unsigned int offset = idx_row * width;
      const float * it_depth = it_depth_0 + offset;
      const char * it_rgb = it_rgb_0 + offset * 4;
      const float dy = rowmap(idx_row);

      for(std::size_t x = 0; x < width; ++x, ++it_point_cloud, ++it_depth, it_rgb += 4 )
      {
        const float depth_value = *it_depth / 1000.0f;

        if (!std::isnan(depth_value) && !(std::abs(depth_value) < 0.0001))
        {
          const float rx = colmap(x) * depth_value;
          const float ry = dy * depth_value;
          it_point_cloud->z = depth_value;
          it_point_cloud->x = rx;
          it_point_cloud->y = ry;

          it_point_cloud->b = it_rgb[0];
          it_point_cloud->g = it_rgb[1];
          it_point_cloud->r = it_rgb[2];
        }
        else
        {
          it_point_cloud->z = qnan_;
          it_point_cloud->x = qnan_;
          it_point_cloud->y = qnan_;

          it_point_cloud->b = qnan_;
          it_point_cloud->g = qnan_;
          it_point_cloud->r = qnan_;
          is_dense = false;
        }
      }
    }
    point_cloud->is_dense = is_dense;
  }

  // Use only if you want only depth, else use get(cv::Mat, cv::Mat) to have the images aligned
  void
  Kinect2Grabber::getDepth (cv::Mat depth_mat)
  {
    listener_.waitForNewFrame(frames_);

    libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
    cv::Mat depth_tmp(depth->height, depth->width, CV_32FC1, depth->data);
    if (mirror_ == true)
      cv::flip(depth_tmp, depth_mat, 1);
    else
      depth_mat = depth_tmp.clone();

    listener_.release(frames_);
  }

  void
  Kinect2Grabber::getIr (cv::Mat ir_mat)
  {
    listener_.waitForNewFrame(frames_);

    libfreenect2::Frame * ir = frames_[libfreenect2::Frame::Ir];
    cv::Mat ir_tmp(ir->height, ir->width, CV_32FC1, ir->data);
    if (mirror_ == true)
      cv::flip(ir_tmp, ir_mat, 1);
    else
      ir_mat = ir_tmp.clone();

    listener_.release(frames_);
  }

  // Use only if you want only color, else use get(cv::Mat, cv::Mat) to have the images aligned
  void
  Kinect2Grabber::getColor (cv::Mat & color_mat)
  {
    listener_.waitForNewFrame(frames_);

    libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
    cv::Mat tmp_color(rgb->height, rgb->width, CV_8UC4, rgb->data);
    if (mirror_ == true)
      cv::flip(tmp_color, color_mat, 1);
    else
      color_mat = tmp_color.clone();

    listener_.release(frames_);
  }

  // Depth and color are aligned and registered
  void
  Kinect2Grabber::get (cv::Mat & color_mat, cv::Mat & depth_mat, const bool full_hd, const bool remove_points)
  {
    listener_.waitForNewFrame(frames_);

    libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
    libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
    registration_->apply(rgb, depth, &undistorted_, &registered_, remove_points, &big_depth_, color_depth_map_);

    cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
    cv::Mat tmp_color;
    if (full_hd)
      tmp_color = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    else
      tmp_color = cv::Mat(registered_.height, registered_.width, CV_8UC4, registered_.data);

    if (mirror_ == true)
    {
      cv::flip(tmp_depth, depth_mat, 1);
      cv::flip(tmp_color, color_mat, 1);
    }
    else
    {
      color_mat = tmp_color.clone();
      depth_mat = tmp_depth.clone();
    }

    listener_.release(frames_);
  }

  // Depth and color are aligned and registered
  void
  Kinect2Grabber::get (cv::Mat & color_mat, cv::Mat & depth_mat, cv::Mat & ir_mat, const bool full_hd, const bool remove_points)
  {
    listener_.waitForNewFrame(frames_);
    libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
    libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
    libfreenect2::Frame * ir = frames_[libfreenect2::Frame::Ir];

    registration_->apply(rgb, depth, &undistorted_, &registered_, remove_points, &big_depth_, color_depth_map_);

    cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
    cv::Mat tmp_color;
    cv::Mat ir_tmp(ir->height, ir->width, CV_32FC1, ir->data);

    if (full_hd)
      tmp_color = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    else
      tmp_color = cv::Mat(registered_.height, registered_.width, CV_8UC4, registered_.data);

    if (mirror_ == true)
    {
      cv::flip(tmp_depth, depth_mat, 1);
      cv::flip(tmp_color, color_mat, 1);
      cv::flip(ir_tmp, ir_mat, 1);
    }
    else
    {
      color_mat = tmp_color.clone();
      depth_mat = tmp_depth.clone();
      ir_mat = ir_tmp.clone();
    }

    listener_.release(frames_);
  }

  void
  Kinect2Grabber::get (cv::Mat & color_mat, cv::Mat & depth_mat, pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
      const bool full_hd, const bool remove_points)
  {
    listener_.waitForNewFrame(frames_);
    libfreenect2::Frame * rgb = frames_[libfreenect2::Frame::Color];
    libfreenect2::Frame * depth = frames_[libfreenect2::Frame::Depth];
    registration_->apply(rgb, depth, &undistorted_, &registered_, remove_points, &big_depth_, color_depth_map_);

    cv::Mat tmp_depth(undistorted_.height, undistorted_.width, CV_32FC1, undistorted_.data);
    cv::Mat tmp_color;

    if(full_hd)
      tmp_color = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
    else
      tmp_color = cv::Mat(registered_.height, registered_.width, CV_8UC4, registered_.data);

    if (mirror_ == true)
    {
      cv::flip(tmp_depth, depth_mat, 1);
      cv::flip(tmp_color, color_mat, 1);
    }
    else
    {
      color_mat = tmp_color.clone();
      depth_mat = tmp_depth.clone();
    }

    updatePointCloud(point_cloud);

    listener_.release(frames_);
  }

  void
  Kinect2Grabber::prepareMake3D (const IrCameraParams & depth_p)
  {
    const int width = 512;
    const int height = 424;
    float * pm1 = colmap.data();
    float * pm2 = rowmap.data();
    for(int i = 0; i < width; i++)
    {
        *pm1++ = (i-depth_p.cx + 0.5) / depth_p.fx;
    }
    for (int i = 0; i < height; i++)
    {
        *pm2++ = (i-depth_p.cy + 0.5) / depth_p.fy;
    }
  }

} // namespace radi
