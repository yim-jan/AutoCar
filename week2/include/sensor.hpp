#ifndef SENSOR_HPP
#define SENSOR_HPP

#include <random>
#include <memory>

// 传感器基类
class Sensor {
public:
    virtual double read() = 0;
    virtual ~Sensor() = default;
};

// 激光雷达模拟
class LidarSensor : public Sensor {
private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution{0.0, 10.0};
public:
    double read() override {
        return distribution(generator);
    }
};

#endif // SENSOR_HPP
