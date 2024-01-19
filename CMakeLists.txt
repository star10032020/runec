cmake_minimum_required(VERSION 3.8)
project(runec)

# 默认设置
add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

#设置显卡cuda架构，根据https://developer.nvidia.com/cuda-gpus查询并把小数点去掉
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

set(CMAKE_CUDA_ARCHITECTURES 86)

endif()


#Tensorrt路径
set(TensorRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu/") # Adjust this path
set(TensorRT_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu/")     # Adjust this path
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)

# 包含头文件
set(YOLO
#src/yololayer.cu#单独注入依赖#不单独了
src/block.cpp
src/BufDetect.cpp
src/calibrator.cpp
src/process.cu              
src/model.cpp  
src/utils.cpp                
src/YoloTRT.cpp
)
include_directories(
  "include"
)

#自定义消息包
find_package(msg_pkg REQUIRED)
#Opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)
message(STATUS "libraries: ${CUDA_LIBRARIES}")
message(STATUS "include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)
#Tensorrt
include_directories(${TensorRT_INCLUDE_DIR})
link_directories(${TensorRT_LIBRARY_DIR})

#生成yolo依赖
add_library(myplugins SHARED src/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

add_executable(testImagePublisher src/testImagePublisher.cpp )
add_executable(testImageSubscriber src/testImageSubscriber.cpp ${YOLO})

ament_target_dependencies(testImagePublisher rclcpp sensor_msgs image_transport  cv_bridge OpenCV msg_pkg)
ament_target_dependencies(testImageSubscriber rclcpp sensor_msgs image_transport  cv_bridge OpenCV msg_pkg)

#注入依赖
target_link_libraries(testImageSubscriber nvinfer)
target_link_libraries(testImageSubscriber cudart)
target_link_libraries(testImageSubscriber myplugins)
target_link_libraries(testImageSubscriber ${OpenCV_LIBS})

set_target_properties(testImageSubscriber PROPERTIES
  INSTALL_RPATH "$ORIGIN"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  myplugins
  testImagePublisher
  testImageSubscriber
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
