#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "Mega.h"

using namespace std;

class Test_CpuLCSMinMax : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

TEST_F(Test_CpuLCSMinMax, test_ex) {
    // 测试异常情况
    EXPECT_THROW({
                     vector<int> empty1;
                     vector<int> empty2;
                     vector<int> empty3;
                     vector<int> empty4;
                     Mega::CpuLCS_MinMax(empty1.data(), empty1.size(), empty2.data(), empty2.size(),
                                         empty3.data(), empty3.size(), empty4.data(), empty4.size());
                 }, runtime_error);

    EXPECT_THROW({
                     int base[] = {5};
                     int latest[] = {5};
                     int ver[] = {5};
                     vector<int> empty;
                     Mega::CpuLCS_MinMax(base, 1, latest, 1, ver, 1, empty.data(), 0);
                 }, runtime_error);

    EXPECT_THROW({
                     int base[] = {5};
                     int latest[] = {5};
                     vector<int> empty;
                     int hor[] = {5};
                     Mega::CpuLCS_MinMax(base, 1, latest, 1, empty.data(), 0, hor, 1);
                 }, runtime_error);

    EXPECT_THROW({
                     int base[] = {5};
                     vector<int> empty;
                     int ver[] = {5};
                     int hor[] = {5};
                     Mega::CpuLCS_MinMax(base, 1, empty.data(), 0, ver, 1, hor, 1);
                 }, runtime_error);

    EXPECT_THROW({
                     vector<int> empty;
                     int latest[] = {5};
                     int ver[] = {5};
                     int hor[] = {5};
                     Mega::CpuLCS_MinMax(empty.data(), 0, latest, 1, ver, 1, hor, 1);
                 }, runtime_error);
    // 长度不匹配测试
    EXPECT_THROW(({
        int base[] = {5, 6};
        int latest[] = {5};
        int ver[] = {5};
        int hor[] = {5, 6};
        Mega::CpuLCS_MinMax(base, 2, latest, 1, ver, 1, hor, 2);
    }), runtime_error);

    EXPECT_THROW(({
        int base[] = {5};
        int latest[] = {5, 6};
        int ver[] = {5, 6};
        int hor[] = {5};
        Mega::CpuLCS_MinMax(base, 1, latest, 2, ver, 2, hor, 1);
    }), runtime_error);

}

TEST_F(Test_CpuLCSMinMax, test_1x1) {
    // 无基础权重的测试用例 1
    {
        int verWeights[] = {0};
        int horWeights[] = {0};
        int base[] = {5};
        int latest[] = {6};
        Mega::CpuLCS_MinMax(base, 1, latest, 1, verWeights, 1, horWeights, 1);
        EXPECT_EQ(verWeights[0], 0);
        EXPECT_EQ(horWeights[0], 0);
    }

    {
        int verWeights[] = {0};
        int horWeights[] = {0};
        int base[] = {5};
        int latest[] = {5};
        Mega::CpuLCS_MinMax(base, 1, latest, 1, verWeights, 1, horWeights, 1);
        EXPECT_EQ(verWeights[0], 1);
        EXPECT_EQ(horWeights[0], 1);
    }

    // 有基础权重的测试用例 1
    {
        int verWeights[] = {10};
        int horWeights[] = {10};
        int base[] = {5};
        int latest[] = {6};
        Mega::CpuLCS_MinMax(base, 1, latest, 1, verWeights, 1, horWeights, 1);
        EXPECT_EQ(verWeights[0], 10);
        EXPECT_EQ(horWeights[0], 10);
    }

    {
        int verWeights[] = {10};
        int horWeights[] = {10};
        int base[] = {5};
        int latest[] = {5};
        Mega::CpuLCS_MinMax(base, 1, latest, 1, verWeights, 1, horWeights, 1);
        EXPECT_EQ(verWeights[0], 11);
        EXPECT_EQ(horWeights[0], 11);
    }
}

TEST_F(Test_CpuLCSMinMax, test_2x2) {
    // 无基础权重的测试用例 1
    {
        int verWeights[] = {0, 0};
        int horWeights[] = {0, 0};
        int base[] = {5, 6};
        int latest[] = {5, 6};
        Mega::CpuLCS_MinMax(base, 2, latest, 2, verWeights, 2, horWeights, 2);
        EXPECT_EQ(verWeights[0], 1);
        EXPECT_EQ(verWeights[1], 2);
        EXPECT_EQ(horWeights[0], 1);
        EXPECT_EQ(horWeights[1], 2);
    }

    {
        int verWeights[] = {0, 0};
        int horWeights[] = {0, 0};
        int base[] = {5, 6};
        int latest[] = {7, 8};
        Mega::CpuLCS_MinMax(base, 2, latest, 2, verWeights, 2, horWeights, 2);
        EXPECT_EQ(verWeights[0], 0);
        EXPECT_EQ(verWeights[1], 0);
        EXPECT_EQ(horWeights[0], 0);
        EXPECT_EQ(horWeights[1], 0);
    }

    // 有基础权重的测试用例 1
    {
        int verWeights[] = {10, 11};
        int horWeights[] = {10, 12};
        int base[] = {5, 6};
        int latest[] = {5, 6};
        Mega::CpuLCS_MinMax(base, 2, latest, 2, verWeights, 2, horWeights, 2);
        EXPECT_EQ(verWeights[0], 12);
        EXPECT_EQ(verWeights[1], 12);
        EXPECT_EQ(horWeights[0], 11);
        EXPECT_EQ(horWeights[1], 12);
    }

    {
        int verWeights[] = {10, 11};
        int horWeights[] = {10, 12};
        int base[] = {5, 6};
        int latest[] = {7, 8};
        Mega::CpuLCS_MinMax(base, 2, latest, 2, verWeights, 2, horWeights, 2);
        EXPECT_EQ(verWeights[0], 12);
        EXPECT_EQ(verWeights[1], 12);
        EXPECT_EQ(horWeights[0], 11);
        EXPECT_EQ(horWeights[1], 12);
    }
}

TEST_F(Test_CpuLCSMinMax, test_2x3) {
    // 有基础权重的测试用例 1
    {
        int verWeights[] = {10, 11, 12};
        int horWeights[] = {10, 11};
        int base[] = {5, 6, 7};
        int latest[] = {5, 6};
        Mega::CpuLCS_MinMax(base, 3, latest, 2, verWeights, 3, horWeights, 2);
        EXPECT_EQ(verWeights[0], 11);
        EXPECT_EQ(verWeights[1], 12);
        EXPECT_EQ(verWeights[2], 12);
        EXPECT_EQ(horWeights[0], 12);
        EXPECT_EQ(horWeights[1], 12);
    }

    {
        int verWeights[] = {10, 11};
        int horWeights[] = {10, 11, 12};
        int base[] = {5, 6};
        int latest[] = {5, 6, 7};
        Mega::CpuLCS_MinMax(base, 2, latest, 3, verWeights, 2, horWeights, 3);
        EXPECT_EQ(verWeights[0], 12);
        EXPECT_EQ(verWeights[1], 12);
        EXPECT_EQ(horWeights[0], 11);
        EXPECT_EQ(horWeights[1], 12);
        EXPECT_EQ(horWeights[2], 12);
    }
}

TEST_F(Test_CpuLCSMinMax, test_all) {
    char chars[] = {'a', 'b', 'c'};
    vector<vector<int>> combinations;

    // 生成所有可能的2字符组合（笛卡尔积），转换为int数组
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            combinations.push_back({chars[i], chars[j]});
        }
    }

    // 生成81个测试用例
    for (size_t i = 0; i < combinations.size(); i++) {
        for (size_t j = 0; j < combinations.size(); j++) {
            int testCaseIndex = i * combinations.size() + j + 1;

            vector<int> baseVals = combinations[i];
            vector<int> latestVals = combinations[j];
            vector<int> verWeights(baseVals.size(), 0);
            vector<int> horWeights(latestVals.size(), 0);

            // 保存原始数组用于经典算法
            vector<int> baseValsOriginal = combinations[i];
            vector<int> latestValsOriginal = combinations[j];

            // 打印测试用例信息
            cout << "Test Case " << testCaseIndex << ": base=["
                 << (char) baseVals[0] << ", " << (char) baseVals[1]
                 << "], latest=[" << (char) latestVals[0] << ", " << (char) latestVals[1] << "]" << endl;

            Mega::CpuLCS_MinMax(baseVals.data(), baseVals.size(),
                                latestVals.data(), latestVals.size(),
                                verWeights.data(), verWeights.size(),
                                horWeights.data(), horWeights.size());

            auto classic = Mega::CpuLCS_DPMatrix(baseValsOriginal, latestValsOriginal);

            EXPECT_EQ(verWeights, classic.first);
            EXPECT_EQ(horWeights, classic.second);
        }
    }
}
