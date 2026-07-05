#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include "sensor.hpp"

std::queue<double> data_queue;
std::mutex mtx;
std::atomic<bool> running(true);

void producer() {
    LidarSensor sensor;
    while (running) {
        double val = sensor.read();
        {
            std::lock_guard<std::mutex> lock(mtx);
            data_queue.push(val);
            std::cout << "[生产] " << val << " 米" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void consumer() {
    while (running) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!data_queue.empty()) {
                double val = data_queue.front();
                data_queue.pop();
                std::cout << "[消费] 已处理: " << val << " 米" << std::endl;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main() {
    std::cout << "=== 多线程传感器系统启动 ===" << std::endl;
    std::thread t1(producer);
    std::thread t2(consumer);

    std::this_thread::sleep_for(std::chrono::seconds(3));
    running = false;

    t1.join();
    t2.join();
    std::cout << "=== 系统停止 ===" << std::endl;
    return 0;
}
