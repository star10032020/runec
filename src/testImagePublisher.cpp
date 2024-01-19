#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class ImagePublisher : public rclcpp::Node
{
public:
  ImagePublisher() : Node("image_publisher")
  {
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&ImagePublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = sensor_msgs::msg::Image();
    message.header.stamp = this->now();
    // 例如，创建一个简单的100x100灰度图像
    message.height = 100;
    message.width = 100;
    message.encoding = "mono8";
    message.step = 100; // 8位单通道，步长等于宽度
    message.data.resize(100 * 100, 0); // 使用0填充

    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImagePublisher>());
  rclcpp::shutdown();
  return 0;
}