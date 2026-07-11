#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <memory>


class SimpleThreadPool {
private:
    std::vector<std::thread> workers;         // 工人线程们
    std::queue<std::shared_ptr<std::function<void()>>> tasks;  // 任务队列

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;  // 关门标志

public:
    // 构造函数：招几个工人？
    SimpleThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            // 每个工人线程都在干同一件事：循环从队列里抢任务
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task; 
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        // 队列空且没关门，就睡觉
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        // 如果关门了且队列也空了，工人就下班
                        if (this->stop && this->tasks.empty()) return;
                        // 抢到一个任务
                        task = std::move(*this->tasks.front());
                        this->tasks.pop();
                    }
                    // 放手干活，不加锁
                    task();
                }
            });
        }
    }

    // 给线程池派任务
    template<typename F>
    void enqueue(F&& task) { 
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::make_shared<std::function<void()>>(std::move(task)));
        }
        condition.notify_one();  // 叫醒一个睡觉的工人
    }
    size_t pendingTasks() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        return tasks.size();
    }
     void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    // 析构函数：关门，等所有工人下班
    ~SimpleThreadPool() {
        if (!stop) shutdown();
    }
};