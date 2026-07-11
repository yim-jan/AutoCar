#include "sensor_processor/thread_pool.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

class ImageProcessorNode : public rclcpp::Node {
public:
    ImageProcessorNode() : Node("image_processor"), pool_(4) {
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10,
            [this](sensor_msgs::msg::Image::SharedPtr msg) {
                pool_.enqueue([this, msg] { processImage(msg); });
            });
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("/image_processed", 10);
        timer_ = this->create_wall_timer(std::chrono::seconds(1), [this] {
            RCLCPP_INFO(this->get_logger(), "队列积压: %zu", pool_.pendingTasks());
        });
    }

private:
    void processImage(sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);
        auto out_msg = cv_bridge::CvImage(msg->header, "mono8", edges).toImageMsg();
        pub_->publish(*out_msg);
    }

    SimpleThreadPool pool_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProcessorNode>());
    rclcpp::shutdown();
    return 0;
}