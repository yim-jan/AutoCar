#include <iostream>
#include <thread>
#include <vector> 
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional> 
#include <chrono>
#include <random>
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
void producer(SimpleThreadPool& pool, bool& running) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(15.0, 35.0);
    int task_id = 0;

    while (running) {
        double temperature = distribution(generator);
        auto start_time = std::chrono::steady_clock::now(); 

        pool.enqueue([task_id, temperature, start_time] {
            auto queue_delay = std::chrono::steady_clock::now() - start_time; 
            // 模拟实际处理工作（50~150ms，模拟算法推理耗时）
            std::this_thread::sleep_for(std::chrono::milliseconds(50 + rand() % 100));
            
            printf("[工人 %d] 任务#%d 温度:%.1f℃ | 排队耗时:%lldms\n",
                   std::this_thread::get_id(), task_id, temperature,
                   std::chrono::duration_cast<std::chrono::milliseconds>(queue_delay).count());
        });

        task_id++;
        // 模拟传感器采样频率（100ms一帧，即10Hz）
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "[生产者] 已停止生产。" << std::endl;
}

int main() {
    SimpleThreadPool pool(3);
    bool running = true;

    // 把随机数生成器移到 main 里，这样突发部分也能用
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(15.0, 35.0);

    std::cout << "=== 模拟真实传感器数据处理（10Hz输入，3工人） ===" << std::endl;
    std::thread producer_thread(producer, std::ref(pool), std::ref(running));

    // 运行5秒，每秒报告一次状态
    for (int t = 0; t < 5; ++t) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "[监控] 运行" << (t+1) << "秒 | 队列积压: " 
                  << pool.pendingTasks() << " 个任务" << std::endl;
    }

    // 模拟突发流量
    std::cout << "=== 模拟突发流量：瞬间涌入20个任务 ===" << std::endl;
    for (int i = 0; i < 20; ++i) {
        double temperature = distribution(generator);
        pool.enqueue([i, temperature] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50 + rand() % 100));
            printf("[工人 %d] 突发任务#%d 温度:%.1f℃\n", 
                   std::this_thread::get_id(), i, temperature);
        });
    }
    std::cout << "突发投放完毕，队列积压: " << pool.pendingTasks() << std::endl;

    // 发出停止信号
    running = false;
    producer_thread.join();
    std::cout << "=== 等待剩余任务处理完毕 ===" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    pool.shutdown();
    std::cout << "=== 系统正常关闭 ===" << std::endl;
    return 0;
}