// MegaLCSPerfMain.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include "Mega.h"

using namespace std;
using namespace std::chrono;

int main() {
    // 测试参数
    vector<int> sizes = {65536, 1048576, 2097152, 4194304};
    int STEP = 256;

    cout << "MegaLCS Performance Test" << endl;
    cout << "========================" << endl;

    for (int MAX: sizes) {
        cout << "\nTesting size: " << MAX << endl;

        // 准备测试数据
        vector<int> inputArray(MAX);
        vector<int> expectArray(MAX);
        for (int i = 0; i < MAX; i++) {
            inputArray[i] = i;
            expectArray[i] = i + 1;
        }

        // 获取GPU设备
        auto devicePair = Mega::GetFirstGpuDevice();
        cl_platform_id platformId = devicePair.first;
        cl_device_id deviceId = devicePair.second;

        if (platformId == nullptr || deviceId == nullptr) {
            cout << "  No GPU device found, skipping..." << endl;
            continue;
        }

        // 准备权重数组
        vector<int> verWeights = inputArray;
        vector<int> horWeights = inputArray;

        // 执行性能测试
        auto start = high_resolution_clock::now();

        Mega::HostLCS_WaveFront(
                platformId,
                deviceId,
                inputArray,
                inputArray,
                verWeights,
                horWeights,
                true,
                STEP,
                false
        );

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << "  Execution time: " << duration.count() << " ms" << endl;
        cout << "  Result: " << horWeights.back() << endl;
    }

    return 0;
}
