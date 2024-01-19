#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include "msg_pkg/msg/rune_corner.hpp" //这里被自动修改了，我们也得改
#include "runec/BufDetect.hpp"

class ImageSubscriber : public rclcpp::Node
{
public:
  ImageSubscriber()
  : Node("testImageSubscriber")
  {
    RCLCPP_INFO(this->get_logger(), "We are trying to build the YOLO...");
    this->yolo = new YoloTRT(wtsPath, enginePath, classPath, modelType);
    yolo->setConfig(
      InputName, OutputName, NumClass, BatchSize, InputH, InputW, NmsThresh, ConfThresh,
      MaxInputImageSize, MaxNumOutputBbox);
    yolo->readyWork();
    RCLCPP_INFO(this->get_logger(), "YOLO架构准备完成...");
    this->bufdetect = new BufDetect(yolo, targetColor);
    RCLCPP_INFO(this->get_logger(), "后处理接口完成对接...");

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "ImageSend", 10, std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));
   publisher_ = this->create_publisher<msg_pkg::msg::RuneCorner>("Receive_corner", 10);
        message_ = std::make_shared<msg_pkg::msg::RuneCorner>();
  
  }

private:
  std::string wtsPath = "/workspaces/vscode_ros2_workspace/model/Rune.wts";
  std::string enginePath = "/workspaces/vscode_ros2_workspace/model/Rune.engine";
  std::string classPath = "/workspaces/vscode_ros2_workspace/model/classes.txt";
  std::string modelType = "n";
  YoloTRT * yolo = nullptr;
  std::vector<cv::Point> ansList;

  std::string InputName = "images", OutputName = "output";
  int NumClass = 3, BatchSize = 8;
  int InputH = 640, InputW = 640;
  float NmsThresh = 0.45f, ConfThresh = 0.5f;
  int MaxInputImageSize = 1920 * 1080, MaxNumOutputBbox = 1000;


  int targetColor = 0;    //1红2蓝

  BufDetect * bufdetect = nullptr;
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    //RCLCPP_INFO(this->get_logger(), "接收到图像: 宽度=%u 高度=%u", msg->width, msg->height);
    // 这里可以添加更多的处理代码
    //cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;

    this->bufdetect->newDetect(
      cv_bridge::toCvCopy(
        msg,
        sensor_msgs::image_encodings::BGR8)->image,
      ansList);
      message_->data.clear();
      for(cv::Point item:ansList)
      {
        message_->data.push_back(item.x);
        message_->data.push_back(item.y);
      }
       publisher_->publish(*message_);

  }



  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<msg_pkg::msg::RuneCorner>::SharedPtr publisher_;
  std::shared_ptr<msg_pkg::msg::RuneCorner> message_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageSubscriber>());
  rclcpp::shutdown();
  return 0;
}
