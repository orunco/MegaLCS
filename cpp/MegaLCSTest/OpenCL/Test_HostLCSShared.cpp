#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <chrono>
#include "Mega.h"

using namespace std;

// Helper function to get all devices
vector<pair<cl_platform_id, cl_device_id>> GetAllDevices() {
    vector<pair<cl_platform_id, cl_device_id>> devices;

    cl_uint platformCount;
    clGetPlatformIDs(0, nullptr, &platformCount);
    vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    for (auto platform : platforms) {
        cl_uint deviceCount;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
        vector<cl_device_id> devs(deviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devs.data(), nullptr);

        for (auto device : devs) {
            devices.emplace_back(platform, device);
        }
    }

    return devices;
}

void Test_HostLCS_Shared(
    vector<int> baseVals,
    vector<int> latestVals,
    vector<int> verWeights,
    vector<int> horWeights,
    const vector<int>& versOutExpect,
    const vector<int>& horsOutExpect,
    int step,
    bool isDebug = false) {

    auto devices = GetAllDevices();
    ASSERT_FALSE(devices.empty()) << "No OpenCL devices found.";

    for (auto& [platformId, deviceId] : devices) {
        if (isDebug) {
            char name[1024];
            clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            cout << "Run on device: " << name << endl;
        }

        auto start = chrono::high_resolution_clock::now();

        vector<int> versOut = verWeights;
        vector<int> horsOut = horWeights;

        Mega::HostLCS_WaveFront(
            platformId,
            deviceId,
            baseVals,
            latestVals,
            versOut,
            horsOut,
            true,
            step,
            isDebug
        );

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Execution time: " << duration << " ms" << endl;
        cout << "Last verWeight=" << versOut.back() << ", Last horWeight=" << horsOut.back() << endl;

        ASSERT_EQ(versOut, versOutExpect);
        ASSERT_EQ(horsOut, horsOutExpect);
    }
}

TEST(Test_HostLCSShared, Test_classic) {
    {
        vector<int> bases = {'A', 'B', 'C', 'B', 'D', 'A', 'B'};
        vector<int> latests = {'B', 'D', 'C', 'A', 'B', 'C'};
        vector<int> inputVerW(bases.size(), 0);
        vector<int> inputHorW(latests.size(), 0);
        vector<int> expectVers = {1, 2, 3, 3, 3, 3, 4};
        vector<int> expectHors = {1, 2, 2, 3, 4, 4};
        Test_HostLCS_Shared(bases, latests, inputVerW, inputHorW, expectVers, expectHors, 1, true);
    }

    {
        vector<int> bases = {'A', 'B', 'C', 'B', 'D', 'A'};
        vector<int> latests = {'B', 'D', 'C', 'A', 'B', 'C'};
        vector<int> inputVerW(bases.size(), 0);
        vector<int> inputHorW(latests.size(), 0);
        vector<int> expectVers = {1, 2, 3, 3, 3, 3};
        vector<int> expectHors = {1, 2, 2, 3, 3, 3};
        Test_HostLCS_Shared(bases, latests, inputVerW, inputHorW, expectVers, expectHors, 2, true);
    }

    {
        vector<int> bases = {'A', 'B', 'C', 'B', 'D', 'A'};
        vector<int> latests = {'B', 'D', 'C', 'A', 'B', 'C'};
        vector<int> inputVerW(bases.size(), 0);
        vector<int> inputHorW(latests.size(), 0);
        vector<int> expectVers = {1, 2, 3, 3, 3, 3};
        vector<int> expectHors = {1, 2, 2, 3, 3, 3};
        Test_HostLCS_Shared(bases, latests, inputVerW, inputHorW, expectVers, expectHors, 3, true);
    }

    {
        vector<int> bases = {'A', 'B', 'C', 'B', 'D', 'A'};
        vector<int> latests = {'B', 'D', 'C', 'A', 'B', 'C'};
        vector<int> inputVerW(bases.size(), 0);
        vector<int> inputHorW(latests.size(), 0);
        vector<int> expectVers = {1, 2, 3, 3, 3, 3};
        vector<int> expectHors = {1, 2, 2, 3, 3, 3};
        Test_HostLCS_Shared(bases, latests, inputVerW, inputHorW, expectVers, expectHors, 6, true);
    }
}

TEST(Test_HostLCSShared, Test_RegularArray1x1) {
    Test_HostLCS_Shared({5}, {5}, {0}, {0}, {1}, {1}, 1);
    Test_HostLCS_Shared({5}, {6}, {0}, {0}, {0}, {0}, 1);
    Test_HostLCS_Shared({5}, {5}, {10}, {10}, {11}, {11}, 1);
    Test_HostLCS_Shared({5}, {6}, {10}, {10}, {10}, {10}, 1);
}

TEST(Test_HostLCSShared, Test_RegularArray2x2_step1) {
    Test_HostLCS_Shared({5, 6}, {5, 6}, {0, 0}, {0, 0}, {1, 2}, {1, 2}, 1);
    Test_HostLCS_Shared({5, 6}, {7, 8}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 1);
    Test_HostLCS_Shared({5, 6}, {5, 6}, {10, 20}, {30, 40}, {40, 21}, {20, 21}, 1);
    Test_HostLCS_Shared({5, 6}, {7, 8}, {10, 20}, {30, 40}, {40, 40}, {30, 40}, 1);
}

TEST(Test_HostLCSShared, Test_Other) {
    Test_HostLCS_Shared({5, 6, 5}, {5, 6}, {30, 40, 50}, {10, 20}, {20, 21, 41}, {41, 41}, 1, true);
    Test_HostLCS_Shared({5, 6}, {5, 6}, {30, 40}, {10, 20}, {20, 21}, {40, 21}, 1, true);
    Test_HostLCS_Shared({5, 6}, {5, 6}, {11, 12}, {10, 11}, {11, 12}, {12, 12}, 1, true);
    Test_HostLCS_Shared({5, 6, 7, 8}, {5, 6, 7, 8}, {11, 12, 13, 14}, {10, 11, 12, 13}, {13, 13, 13, 14}, {14, 14, 14, 14}, 2, true);

    int arrayMax = 65536;
    vector<int> inputArray(arrayMax);
    vector<int> expectArray(arrayMax);
    for (int i = 0; i < arrayMax; ++i) {
        inputArray[i] = 0; //csharp里面默认是0，cpp需要显式赋值
        expectArray[i] = i + 1;
    }

    Test_HostLCS_Shared(inputArray, inputArray, inputArray, inputArray, expectArray, expectArray, 256, false);
}