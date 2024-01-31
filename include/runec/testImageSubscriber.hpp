#pragma once
#include <memory>
#include<map>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include "YoloTRT.h"
//#include "msg_pkg/msg/rune_corner.hpp" //这里被自动修改了，我们也得改
#include "auto_aim_msg/msg/point2d.hpp"
#include "auto_aim_msg/msg/rune_detection.hpp"
#include "auto_aim_msg/msg/rune_detection_result.hpp"
class BufDetect;

class ImageSubscriber : public rclcpp::Node
{
public:
  ImageSubscriber();
   rclcpp::Logger roslogger;
  void DebugSender(const cv::Mat &input,std::string name_);
private:
  void setParam();
  std::string wtsPath = "/workspace/model/Rune.wts";
  std::string enginePath = "/workspace/model/Rune.engine";
  std::string classPath = "/workspace/model/classes.txt";
  std::string modelType = "n";
  YoloTRT * yolo = nullptr;
  std::vector<cv::Point> ansList;

  std::string InputName = "images", OutputName = "output";
  int NumClass = 3, BatchSize = 8;
  int InputH = 640, InputW = 640;
  float NmsThresh = 0.45f, ConfThresh = 0.5f;
  int MaxInputImageSize = 1920 * 1080, MaxNumOutputBbox = 1000;


  int targetColor = 0;    //1红0蓝

  BufDetect * bufdetect = nullptr;
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<auto_aim_msg::msg::RuneDetectionResult>::SharedPtr publisher_;

  std::map<std::string,rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>debuggers;
 
  std::shared_ptr<auto_aim_msg::msg::RuneDetectionResult> msg_Result;
  std::shared_ptr<auto_aim_msg::msg::RuneDetection> msg_Detection;
  std::shared_ptr<auto_aim_msg::msg::Point2d> msg_Point2d;
   rclcpp::TimerBase::SharedPtr timer_;
   void respond();
};
