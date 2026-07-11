//对应的头文件的说明
#include <iostream> //输入输出
#include <thread> //多线程
#include <mutex> //互斥锁防止线程竞争
#include <condition_variable> //条件变量用于线程间的同步
#include <chrono> //时间处理
#include <random> //随机数生成

// 共享数据和同步机制
std::mutex mtx; //调用互斥锁
std::condition_variable cv; //调用条件变量定义同步
double shared_data = 0.0; //共享数据
bool ready = false; //数据准备状态

void producer() {
    std::default_random_engine generator; //调用随机数生成器
    std::uniform_real_distribution<double> distribution(15.0, 35.0); //定义温度范围为15到35摄氏度

    while (true) {
        double new_data = distribution(generator); //生成新的温度数据
        std::this_thread::sleep_for(std::chrono::seconds(2)); //模拟生产数据的时间间隔

        std::unique_lock<std::mutex> lock(mtx); //锁定互斥锁，确保线程安全
        shared_data = new_data; //更新共享数据
        ready = true; //标记数据已准备好
        std::cout << "[生成者] 生产了数据: " << shared_data << " ℃" << std::endl; //输出生成的数据
        cv.notify_one(); //通知等待的线程数据已准备好
    }
}

// 处理者：负责消费数据
void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx); //锁定互斥锁，确保线程安全
        cv.wait(lock, []{ return ready; }); //等待条件变量，直到数据准备好

        double data = shared_data; //读取共享数据
        ready = false; //重置数据准备状态
        std::cout << "[处理者] 处理了数据: " << data << " ℃" << std::endl;
        
        if (data > 30.0){
            std::cout << "⚠️警告：当前室温过高！" << std::endl; //输出警告信息
        } else if (data < 15.0){
            std::cout << "⚠️警告：当前室温过低！" << std::endl; //输出警告信息
        }else{
            std::cout << "✅当前室温正常。" << std::endl; //输出正常信息
        }
        std::cout << "---------------------" << std::endl; //输出分隔线
    }
}

// 主函数
int main() {
    std::thread t1(producer); //创建生产者线程
    std::thread t2(consumer); //创建消费者线程

    t1.join(); //等待生产者线程结束
    t2.join(); //等待消费者线程结束

    return 0; //程序结束
}
    
  