cmake_minimum_required(VERSION 2.8)

project(coffeeCup)
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/freenect2/)
find_package(PCL 1.8 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
find_package(freenect2 REQUIRED)
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${freenect2_INCLUDE_DIR})
link_directories("/opt/freenect2/lib/")

find_package(HDF5)
if(HDF5_FOUND)

  find_package(FLANN)
  include_directories(${FLANN_INCLUDE_DIRS})
  include_directories(${HDF5_INCLUDE_DIR})

  add_executable(realtime_recognize_demo realtime_recognize_demo.cpp kinect2grabber.cpp)
  target_link_libraries(realtime_recognize_demo ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} freenect2
    ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY})

  add_executable(get_training_samples get_training_samples.cpp kinect2grabber.cpp)
  target_link_libraries(get_training_samples ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} freenect2)

  add_executable(scene_test scene_test.cpp)
  target_link_libraries(scene_test ${PCL_LIBRARIES} ${Boost_LIBRARIES})

  add_executable(prepare_data prepare_data.cpp)
  target_link_libraries(prepare_data ${PCL_LIBRARIES} ${Boost_LIBRARIES})

  add_executable(build_tree build_tree.cpp)
  target_link_libraries(build_tree ${PCL_LIBRARIES} ${Boost_LIBRARIES}
                                 ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY})

  add_executable(nearest_neighbors nearest_neighbors.cpp)
  target_link_libraries(nearest_neighbors ${PCL_LIBRARIES}
                                        ${Boost_LIBRARIES} ${FLANN_LIBRARIES} ${HDF5_hdf5_LIBRARY}
                                        )
endif(HDF5_FOUND)
