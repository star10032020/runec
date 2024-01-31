
#include "runec/testImageSubscriber.hpp"
#include "runec/BufDetect.hpp"
void ImageSubscriber::setParam(){
  this->get_parameter("NmsThresh",NmsThresh);//可以及时更改
  this->get_parameter("ConfThresh",ConfThresh);//可以及时更改

  this->get_parameter("targetColor",targetColor);//可以及时更改

  this->yolo->setConfig(InputName, OutputName, NumClass, BatchSize, InputH, InputW,
                  NmsThresh, ConfThresh, MaxInputImageSize, MaxNumOutputBbox);
  this->bufdetect->setParam(this->yolo,targetColor);
  
}
void ImageSubscriber::respond()
{
        RCLCPP_INFO(this->get_logger(), "Hello %f", this->get_parameter("NmsThresh").as_double());
}
void ImageSubscriber::DebugSender(const cv::Mat &input, std::string name_) {


  if (debuggers.find(name_) == debuggers.end()) {
    RCLCPP_INFO(roslogger, "NO the '%s' build in the BuptDebugers!",
                name_.c_str());
    return;
  }
  std::string item = "/debug/DetectRunec_" + name_;
  cv::Mat input0 = input.clone();
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", input0).toImageMsg();
  debuggers[name_]->publish(*msg);
  return;
}
ImageSubscriber::ImageSubscriber(): Node("testImageSubscriber", "/rune", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),roslogger(this->get_logger()) {
  //timer_ = this->create_wall_timer(1000ms, std::bind(&ImageSubscriber::respond, this));
 debuggers.clear();
  std::vector<std::string> debuggersList = {"resultImg", "same_color_overImg",
                                            "littleImg", "thresh_delImg"};
  for (std::string item : debuggersList) {
    debuggers[item] = this->create_publisher<sensor_msgs::msg::Image>(
        "/debug/DetectRunec_" + item, 1);
  }

  //必须先声明参数
 /*
  this-><std::string>("wtsPath","Not Set");
  this->declare_parameter<std::string>("enginePath","Not Set");
  this->declare_parameter<std::string>("classPath","Not Set");
  this->declare_parameter<std::string>("modelType","Not Set");

  this->declare_parameter<int>("NumClass",-1);
  this->declare_parameter<int>("InputH",-1);
  this->declare_parameter<int>("InputW",-1);
  this->declare_parameter<double>("NmsThresh",-0.5f);
  this->declare_parameter<double>("ConfThresh",-0.5f);
  
  this->declare_parameter<int>("targetColor",-1);
  */
  //roslogger = this->get_logger();
  RCLCPP_INFO(roslogger, "We are trying to build the YOLO...");
  
  this->get_parameter("wtsPath",wtsPath);
  //std::cout<<"debug::hi,it is "<<this->get_parameter("wtsPath",wtsPath)<<"\n";
  this->get_parameter("enginePath",enginePath);
  this->get_parameter("classPath",classPath);
  this->get_parameter("modelType",modelType);


  this->get_parameter("NumClass",NumClass);
  this->get_parameter("InputH",InputH);
  this->get_parameter("InputW",InputW);
  this->get_parameter("NmsThresh",NmsThresh);//可以及时更改
  this->get_parameter("ConfThresh",ConfThresh);//可以及时更改
  
  this->get_parameter("targetColor",targetColor);
 // RCLCPP_INFO(roslogger, "modelType=%s",this->get_parameter("modelType").as_string().c_str());
 // RCLCPP_INFO(roslogger, "wtsPath=%s",this->get_parameter("wtsPath").as_string().c_str());
 // RCLCPP_INFO(roslogger, "NmsThresh=%f",this->get_parameter("NmsThresh").as_double());

  this->yolo = new YoloTRT(wtsPath, enginePath, classPath, modelType);
  yolo->setConfig(InputName, OutputName, NumClass, BatchSize, InputH, InputW,
                  NmsThresh, ConfThresh, MaxInputImageSize, MaxNumOutputBbox);
  yolo->readyWork();
  RCLCPP_INFO(roslogger, "YOLO架构准备完成...");

 
  this->bufdetect = new BufDetect(this, yolo, targetColor);
  RCLCPP_INFO(roslogger, "后处理接口完成对接...");

  subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", 10,
      std::bind(&ImageSubscriber::image_callback, this, std::placeholders::_1));
  publisher_ = this->create_publisher<auto_aim_msg::msg::RuneDetectionResult>(
      "detection", 10);
  msg_Result = std::make_shared<auto_aim_msg::msg::RuneDetectionResult>();
  msg_Detection = std::make_shared<auto_aim_msg::msg::RuneDetection>();
  msg_Point2d = std::make_shared<auto_aim_msg::msg::Point2d>();
}
void ImageSubscriber::image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg) {

    this->setParam();//订阅修改参数
  // RCLCPP_INFO(this->get_logger(), "接收到图像: 宽度=%u 高度=%u", msg->width,
  // msg->height);
  //  这里可以添加更多的处理代码
  // cv::Mat img = cv_bridge::toCvCopy(msg,
  // sensor_msgs::image_encodings::BGR8)->image;

  this->bufdetect->newDetect(
      cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image,
      ansList);
  msg_Result->runes.clear();
  msg_Result->header = msg->header; // 相机节点的header
  /*for(cv::Point item:ansList)
  {
    message_->data.push_back(item.x);
    message_->data.push_back(item.y);
  }*/
  cv::Point point_;
  if (ansList.size() >= 5) {
    point_ = ansList[0];
    msg_Detection->outwards_1.x = point_.x;
    msg_Detection->outwards_1.y = point_.y;
    point_ = ansList[1];
    msg_Detection->outwards_2.x = point_.x;
    msg_Detection->outwards_2.y = point_.y;
    point_ = ansList[2];
    msg_Detection->inwards_1.x = point_.x;
    msg_Detection->inwards_1.y = point_.y;
    point_ = ansList[3];
    msg_Detection->inwards_2.x = point_.x;
    msg_Detection->inwards_2.y = point_.y;

    point_ = ansList[4];
    msg_Detection->center_r.x = point_.x;
    msg_Detection->center_r.y = point_.y;
    if (this->get_subscriptions_info_by_topic("detection").empty()) {
      // 输出推理得到的点与画原后的图片
      RCLCPP_INFO(roslogger, "outwards_1:(%f,%f)", msg_Detection->outwards_1.x,
                  msg_Detection->outwards_1.y);
      RCLCPP_INFO(roslogger, "outwards_2:(%f,%f)", msg_Detection->outwards_2.x,
                  msg_Detection->outwards_2.y);
      RCLCPP_INFO(roslogger, "inwards_1:(%f,%f)", msg_Detection->inwards_1.x,
                  msg_Detection->inwards_1.y);
      RCLCPP_INFO(roslogger, "inwards_2:(%f,%f)", msg_Detection->inwards_2.x,
                  msg_Detection->inwards_2.y);
      RCLCPP_INFO(roslogger, "center_r:(%f,%f)", msg_Detection->center_r.x,
                  msg_Detection->center_r.y);
      cv::Mat resultImg =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
      cv::Scalar drawColor, redColor(0, 0, 255), blueColor(255, 0, 0);
      if (this->targetColor == 0)
        drawColor = redColor; // 当前识别蓝色就画红色点
      else
        drawColor = blueColor;
      cv::circle(resultImg,
                 cv::Point((int)(msg_Detection->outwards_1.x),
                           (int)(msg_Detection->outwards_1.y)),
                 5, drawColor, cv::FILLED);
      cv::circle(resultImg,
                 cv::Point((int)(msg_Detection->outwards_2.x),
                           (int)(msg_Detection->outwards_2.y)),
                 7, drawColor, cv::FILLED);
      cv::circle(resultImg,
                 cv::Point((int)(msg_Detection->inwards_1.x),
                           (int)(msg_Detection->inwards_1.y)),
                 9, drawColor, cv::FILLED);
      cv::circle(resultImg,
                 cv::Point((int)(msg_Detection->inwards_2.x),
                           (int)(msg_Detection->inwards_2.y)),
                 11, drawColor, cv::FILLED);
      cv::circle(resultImg,
                 cv::Point((int)(msg_Detection->center_r.x),
                           (int)(msg_Detection->center_r.y)),
                 13, drawColor, cv::FILLED);

      this->DebugSender(resultImg, "resultImg");
    }

    msg_Result->runes.push_back(*msg_Detection);
  }

  publisher_->publish(*msg_Result);
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageSubscriber>());
  rclcpp::shutdown();
  return 0;
}
