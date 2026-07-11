#include <iostream>
#include <vector>
#include <memory>

class Sensor {
public:
    virtual void processdata() = 0; // 纯虚函数，读取传感

    virtual ~Sensor() {
        std::cout << "Sensor基类析构" << std::endl;
    }; // 虚析构函数
};

class CameraSensor : public Sensor {
public:
    void processdata() override {
        std::cout << ">>> [摄像头] 正在处理图像数据，检测车道线和障碍物..." << std::endl;
    }
    ~CameraSensor() {
        std::cout << "CameraSensor析构" << std::endl;
    }
};

class LidarSensor : public Sensor {
public:
    void processdata() override {   
        std::cout << ">>> [激光雷达] 正在处理点云数据，进行3D目标检测... "<< std::endl;
    }
    ~LidarSensor() {
        std::cout << "LidarSensor析构" << std::endl;
    }
};

class RadarSensor : public Sensor {
public:
    void processdata() override {
        std::cout << ">>> [毫米波雷达] 正在处理目标列表，计算相对速度..." << std::endl;
    }
    ~RadarSensor() {
        std::cout << "RadarSensor析构" << std::endl;
    }
};

int main() {
    std::vector<std::shared_ptr<Sensor>> sensor_list;
    sensor_list.push_back(std::make_shared<CameraSensor>());
    sensor_list.push_back(std::make_shared<LidarSensor>());
    sensor_list.push_back(std::make_shared<RadarSensor>());

    std::cout << "=== 开始处理所有传感器数据 ===" << std::endl;

    for (auto& sensor : sensor_list) {
        sensor->processdata();
    }

    std::cout << "\n=== 程序结束，传感器析构 ===" << std::endl;

    return 0;
}