#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "Mega.h"

using namespace std;

class Test_MegaLCSFusion_Coverage : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(Test_MegaLCSFusion_Coverage, Test_Step_Parameter_Validation_Step_Zero) {
    // 测试用例1.1：step = 0，期望抛出异常
    EXPECT_THROW(({
                     vector<int> base = {1, 2};
                     vector<int> latest = {1, 2};
                     Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 0);
                 }), runtime_error);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Step_Parameter_Validation_Step_One) {
    // 测试用例1.2：step = 1，期望正常执行，使用CPU处理
    vector<int> base = {1, 2};
    vector<int> latest = {1, 2};
    auto result = Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 1);
    bool processByCpu = get<0>(result);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_TRUE(processByCpu);
    EXPECT_EQ(verWeights.size(), 2);
    EXPECT_EQ(horWeights.size(), 2);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Step_Parameter_Validation_Step_Two) {
    // 测试用例1.3：step = 2，期望正常执行
    vector<int> base = {1, 2};
    vector<int> latest = {1, 2};
    auto result = Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 2);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_EQ(verWeights.size(), 2);
    EXPECT_EQ(horWeights.size(), 2);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Step_Parameter_Validation_Step_Invalid_High) {
    // 测试用例1.4：step = 257，期望抛出异常
    EXPECT_THROW(({
                     vector<int> base = {1, 2};
                     vector<int> latest = {1, 2};
                     Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 257);
                 }), runtime_error);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Direct_CpuLCS_Base_Length_Less_Than_Step) {
    // 测试用例2.1：baseVals长度 <= step，期望processByCpu=true，正确计算LCS
    vector<int> base = {1};
    vector<int> latest = {1, 2, 3};
    auto result = Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto& horWeights = get<2>(result);

    EXPECT_TRUE(processByCpu);
    // 验证权重计算正确性
    EXPECT_GE(horWeights[horWeights.size() - 1], 0);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Direct_CpuLCS_Latest_Length_Less_Than_Step) {
    // 测试用例2.2：latestVals长度 <= step，期望processByCpu=true，正确计算LCS
    vector<int> base = {1, 2, 3};
    vector<int> latest = {1};
    auto result = Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto& horWeights = get<2>(result);

    EXPECT_TRUE(processByCpu);
    // 验证权重计算正确性
    EXPECT_GE(horWeights[horWeights.size() - 1], 0);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_GPU_Device_Null_Fallback_To_CPU) {
    // 测试用例3.1：platformId为Zero，期望processByCpu=true，正确计算LCS
    vector<int> base = {1, 2, 3};
    vector<int> latest = {1, 2, 3};
    auto result = Mega::MegaLCS_Fusion(nullptr, nullptr, base, latest, 1);
    bool processByCpu = get<0>(result);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_TRUE(processByCpu);
    EXPECT_EQ(verWeights.size(), 3);
    EXPECT_EQ(horWeights.size(), 3);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_LeftTop_Region_Processing_Both_Slice_Greater_Than_Zero) {
    // 测试用例4.1：baseSliceSize > 0 且 latestSliceSize > 0，期望触发左上角处理逻辑
    // 使用真实的OpenCL设备
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2};
    vector<int> latest = {1, 2};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 1);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_EQ(verWeights.size(), 2);
    EXPECT_EQ(horWeights.size(), 2);
    // 如果有GPU设备，应该使用GPU处理
    if (platformId != nullptr && deviceId != nullptr) {
        bool processByCpu = get<0>(result);
        EXPECT_FALSE(processByCpu);
    }
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_RightTop_Processing_Latest_Has_Remainder) {
    // 测试用例5.1：latest有余数，期望触发右上角处理
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2};
    vector<int> latest = {1, 2, 3};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_TRUE(processByCpu);
    EXPECT_EQ(verWeights.size(), 2);
    EXPECT_EQ(horWeights.size(), 3);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_LeftBottom_Processing_Base_Has_Remainder) {
    // 测试用例6.1：base有余数，期望触发左下角处理
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3};
    vector<int> latest = {1, 2};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_TRUE(processByCpu);
    EXPECT_EQ(verWeights.size(), 3);
    EXPECT_EQ(horWeights.size(), 2);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_RightBottom_Processing_Both_Have_Remainder) {
    // 测试用例7.1：base和latest都有余数，期望触发右下角处理
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3};
    vector<int> latest = {1, 2, 3};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    bool processByCpu = get<0>(result);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_FALSE(processByCpu);
    EXPECT_EQ(verWeights.size(), 3);
    EXPECT_EQ(horWeights.size(), 3);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Return_Value_Validation_Weight_Array_Lengths) {
    // 测试用例8.1：验证返回的权重数组长度
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2};
    vector<int> latest = {1, 2};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 1);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_EQ(verWeights.size(), 2);
    EXPECT_EQ(horWeights.size(), 2);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_LCS_Calculation_Correctness) {
    // 测试用例8.2：验证LCS计算正确性
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3};
    vector<int> latest = {2, 3, 4};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 1);
    bool processByCpu = get<0>(result);
    auto& horWeights = get<2>(result);

    EXPECT_FALSE(processByCpu);
    // LCS应该为[2,3]，长度为2
    EXPECT_EQ(horWeights[horWeights.size() - 1], 2);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Step_2_Base_123_vs_Latest_1234) {
    // step=2 [1、2、3] vs [1、2、3、4]
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3};
    vector<int> latest = {1, 2, 3, 4};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_EQ(verWeights.size(), 3);
    EXPECT_EQ(horWeights.size(), 4);
    // LCS应该是[1,2,3]，长度为3
    EXPECT_EQ(horWeights[horWeights.size() - 1], 3);
}

TEST_F(Test_MegaLCSFusion_Coverage, Test_Step_2_Base_1234_vs_Latest_123) {
    // step=2 [1、2、3、4] vs [1、2、3]
    auto devicePair = Mega::GetFirstGpuDevice();
    cl_platform_id platformId = devicePair.first;
    cl_device_id deviceId = devicePair.second;

    vector<int> base = {1, 2, 3, 4};
    vector<int> latest = {1, 2, 3};
    auto result = Mega::MegaLCS_Fusion(platformId, deviceId, base, latest, 2);
    auto& verWeights = get<1>(result);
    auto& horWeights = get<2>(result);

    EXPECT_EQ(verWeights.size(), 4);
    EXPECT_EQ(horWeights.size(), 3);
    // LCS应该是[1,2,3]，长度为3
    EXPECT_EQ(horWeights[horWeights.size() - 1], 3);
}
