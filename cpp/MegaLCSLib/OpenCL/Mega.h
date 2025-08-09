// Mega.h
#ifndef CPP_MEGA_H
#define CPP_MEGA_H

// 消除OpenCL版本警告
#define CL_TARGET_OPENCL_VERSION 210

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <memory>
#include <tuple>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

class Mega {
public:

    static const string KernelLCS_Shared;

    // 主要的LCS计算函数
    static void HostLCS_WaveFront(
            cl_platform_id platformId,
            cl_device_id deviceId,
            vector<int>& baseVals,
            vector<int>& latestVals,
            vector<int>& verWeights,
            vector<int>& horWeights,
            bool isSharedVersion,
            int step,
            bool isDebug = false);

    // CPU版本的LCS计算函数
    static void CpuLCS_MinMax(
            int* baseVals, int baseValsLength,
            int* latestVals, int latestValsLength,
            int* verWeights, int verWeightsLength,
            int* horWeights, int horWeightsLength);

    static void CpuLCS_RollLeftTop(
            int* baseVals, int baseValsLength,
            int* latestVals, int latestValsLength,
            int* verWeights, int verWeightsLength,
            int* horWeights, int horWeightsLength);

    static pair<vector<int>, vector<int>> CpuLCS_DPMatrix(
            const vector<int>& baseVals,
            const vector<int>& latestVals);

    // OpenCL设备管理函数
    static vector<tuple<cl_platform_id, cl_device_id, string, cl_device_type>> GetAllDevices();
    static pair<cl_platform_id, cl_device_id> GetFirstGpuDevice();

    // Fusion函数
    static int MegaLCSLen(const vector<int>& baseVals, const vector<int>& latestVals);
    static tuple<bool, vector<int>, vector<int>> MegaLCS_Fusion(
            cl_platform_id platformId,
            cl_device_id deviceId,
            const vector<int>& baseVals,
            const vector<int>& latestVals,
            int step,
            bool isDebug = false);

private:
    // 验证输入参数
    static int Valid(
            const vector<int>& originalValues,
            bool IsSharedVersion,
            int step);

    // 创建内存对象
    static bool CreateMemObjects(
            cl_context context,
            cl_mem memObjects[4],
            cl_command_queue commandQueue,
            const vector<int>& bases,
            const vector<int>& latests,
            vector<int>& verWeights,
            vector<int>& horWeights);

    // 创建程序
    static cl_program CreateProgram(
            cl_context context,
            cl_device_id device,
            bool IsSharedVersion,
            int _step,
            bool isDebug);

    // 清理资源
    static void Cleanup(
            cl_context context,
            cl_command_queue commandQueue,
            cl_program program,
            cl_kernel kernel,
            cl_mem memObjects[4]);

    // 创建命令队列
    static cl_command_queue CreateCommandQueue(
            cl_context context,
            cl_device_id* device);
};

#endif //CPP_MEGA_H
