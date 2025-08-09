#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <random>
#include "Mega.h"

using namespace std;

class Test_MegaLCSFusion_Value : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// STEP = 2, base = latest = 1 2 3 都相等
TEST_F(Test_MegaLCSFusion_Value, TestCase1_EqualArrays) {
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3};
    vector<int> latest = {1, 2, 3};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto &verWeights = get<1>(result);
    auto &horWeights = get<2>(result);

    EXPECT_FALSE(processByCpu);
    EXPECT_EQ(verWeights.back(), 3);
    EXPECT_EQ(horWeights.back(), 3);
}

// STEP = 2, base = 1 2 3 latest = 5 6 7 都不相等
TEST_F(Test_MegaLCSFusion_Value, TestCase2_DifferentArrays) {
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3};
    vector<int> latest = {5, 6, 7};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto &verWeights = get<1>(result);
    auto &horWeights = get<2>(result);

    EXPECT_FALSE(processByCpu);
    EXPECT_EQ(verWeights.back(), 0);
    EXPECT_EQ(horWeights.back(), 0);
}

// STEP = 2, 随机数组 + 验证
TEST_F(Test_MegaLCSFusion_Value, TestCase3_RandomWithValidation) {
    // 测试足够多的次数，数组足够长且随机，值随机
    for (int j = 0; j < 10; j++) {
        mt19937 rand(j);
        const int MAX = 512;

        int baseLen = rand() % (MAX - 4) + 4;
        vector<int> baseVals(baseLen);
        for (int i = 0; i < baseLen; i++) {
            baseVals[i] = rand() % (MAX - 1) + 1;
        }

        int latestLen = rand() % (MAX - 4) + 4;
        vector<int> latestVals(latestLen);
        for (int i = 0; i < latestLen; i++) {
            latestVals[i] = rand() % (MAX - 1) + 1;
        }

        // 使用经典LCS验证
        auto classicResult = Mega::CpuLCS_DPMatrix(baseVals, latestVals);
        cout << "Classic: " << baseLen << " vs " << latestLen << " => LCS Result: " << classicResult.second.back()
             << endl;

        // 测试目标函数
        auto devicePair = Mega::GetFirstGpuDevice();
        cl_platform_id platformId = devicePair.first;
        cl_device_id deviceId = devicePair.second;
        auto result = Mega::MegaLCS_Fusion(platformId, deviceId, baseVals, latestVals, 2);
        bool processByCpu = get<0>(result);
        auto &verWeights = get<1>(result);
        auto &horWeights = get<2>(result);

        EXPECT_FALSE(processByCpu);
        EXPECT_EQ(verWeights, classicResult.first);
        EXPECT_EQ(horWeights, classicResult.second);
    }
}
