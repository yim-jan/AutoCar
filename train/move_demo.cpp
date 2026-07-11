#include <iostream>
#include <vector>

class SensorData {
private:
    double* data;
    size_t size;

public:
    // 构造函数
    SensorData(size_t n) : size(n) {
        data = new double[n];
        std::cout << "构造函数：分配了 " << n << " 个 double 的内存" << std::endl;
    }

    // 拷贝构造函数（深拷贝，代价高）
    SensorData(const SensorData& other) : size(other.size) {
        data = new double[other.size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
        std::cout << "拷贝构造函数：深拷贝了 " << size << " 个元素（很慢！）" << std::endl;
    }

    // 移动构造函数（偷资源，代价低）
    SensorData(SensorData&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;  // 把原对象的指针置空
        other.size = 0;
        std::cout << "移动构造函数：直接偷走了资源（超快！）" << std::endl;
    }

    // 析构函数
    ~SensorData() {
    if (data) {
        delete[] data;
        std::cout << "析构函数：释放了内存" << std::endl;
    } else {
        std::cout << "析构函数：空壳对象，无需释放" << std::endl;
    }
    }
};

int main() {
    std::cout << "=== 创建数据1 ===" << std::endl;
    SensorData data1(1000000);  // 100万个double，约8MB

    std::cout << "\n=== 拷贝数据1到数据2（慢） ===" << std::endl;
    SensorData data2 = data1;  // 调用拷贝构造

    std::cout << "\n=== 移动数据1到数据3（快） ===" << std::endl;
    SensorData data3 = std::move(data1);  // 调用移动构造，data1 被掏空

    std::cout << "\n=== 程序结束 ===" << std::endl;
    return 0;
}