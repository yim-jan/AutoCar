#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include "sensor.hpp"

std::queue<double> data_queue;
std::mutex mtx;
std::condition_variable cv;
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
        cv.notify_one();  // 唤醒一个等待的消费者
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void consumer() {
    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !data_queue.empty() || !running; });  // 有数据或被唤醒才继续
        
        if (!running && data_queue.empty()) break;
        
        double val = data_queue.front();
        data_queue.pop();
        std::cout << "[消费] 已处理: " << val << " 米" << std::endl;
    }
}

int main() {
    std::cout << "=== 多线程传感器系统启动 (优化版) ===" << std::endl;
    std::thread t1(producer);
    std::thread t2(consumer);

    std::this_thread::sleep_for(std::chrono::seconds(3));
    running = false;
    cv.notify_all();  // 唤醒所有线程，让它们检查退出条件

    t1.join();
    t2.join();
    std::cout << "=== 系统停止 ===" << std::endl;
    return 0;
}
