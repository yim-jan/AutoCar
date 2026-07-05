#include <iostream>
#include <memory>
#include <random>
#include <chrono>

class Sensor {
public:
    virtual double read() = 0;
    virtual ~Sensor() = default;
};

class LidarSensor : public Sensor {
private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution{0.0, 10.0};
public:
    double read() override {
        return distribution(generator);
    }
};

int main() {
    std::unique_ptr<Sensor> sensor = std::make_unique<LidarSensor>();
    for (int i = 0; i < 10; ++i) {
        std::cout << "距离: " << sensor->read() << " 米" << std::endl;
    }
    return 0;
}
