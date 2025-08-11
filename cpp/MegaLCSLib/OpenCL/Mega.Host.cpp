/*
Copyright (C) 2025 Pete Zhang, rivxer@gmail.com, https://github.com/orunco

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "Mega.h"
#include <algorithm>
#include <sstream>

using namespace std;

void Mega::HostLCS_WaveFront(
        cl_platform_id platformId,
        cl_device_id deviceId,
        vector<int> &baseVals,
        vector<int> &latestVals,
        vector<int> &verWeights,
        vector<int> &horWeights,
        bool isSharedVersion,
        int step,
        bool isDebug) {

    int _baseSliceSize = Valid(baseVals, isSharedVersion, step);
    int _latestSliceSize = Valid(latestVals, isSharedVersion, step);

    cl_context context = nullptr;
    cl_command_queue commandQueue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_device_id device = nullptr;

    cl_mem deviceMemObjects[4] = {nullptr, nullptr, nullptr, nullptr};

    // 创建上下文
    cl_context_properties contextProperties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties) platformId,
            0
    };

    cl_int err;
    context = clCreateContext(
            contextProperties,
            1,
            &deviceId,
            nullptr,
            nullptr,
            &err);

    if (err != CL_SUCCESS || context == nullptr) {
        cerr << "Failed to create OpenCL context for device." << endl;
        return;
    }

    // 创建命令队列
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == nullptr) {
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    // 创建程序
    program = CreateProgram(context, device, isSharedVersion, step, isDebug);
    if (program == nullptr) {
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    // 创建内核
    kernel = clCreateKernel(program, "KernelLCS_MinMax", &err);
    if (err != CL_SUCCESS || kernel == nullptr) {
        cerr << "Failed to create kernel" << endl;
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    // 创建内存对象
    if (!CreateMemObjects(
            context,
            deviceMemObjects,
            commandQueue,
            baseVals,
            latestVals,
            verWeights,
            horWeights)) {
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    // 设置内核参数 (gBases,gLatests,gVerWeights,gHorWeights)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceMemObjects[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceMemObjects[1]);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceMemObjects[2]);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &deviceMemObjects[3]);

    // 有多少slice是确定的
    int baseSliceSize = _baseSliceSize;
    int latestSliceSize = _latestSliceSize;
    err |= clSetKernelArg(kernel, 4, sizeof(int), &baseSliceSize);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &latestSliceSize);

    if (err != CL_SUCCESS) {
        cerr << "Error setting kernel arguments." << endl;
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    // Queue the kernel up for execution across the array
    int totalWave = _baseSliceSize + _latestSliceSize - 1;

    // wavefront算法类似波，沿着对角带的方向前进
    for (int outerWaveFrontBand = 0;
         outerWaveFrontBand < totalWave;
         outerWaveFrontBand++) {

        // 首先：共享内存版本STEP个thread每Block，block内元素处理和线程一一对应
        size_t threadPerBlock = step;
        size_t localWorkSize_ThreadPerBlock[] = {threadPerBlock};

        // latest是X轴/水平方向，sliceID最小值
        int latestSliceIDMin = max(0, outerWaveFrontBand - (_baseSliceSize - 1));

        // latest方向的sliceID的最大值
        int latestSliceIDMax = min(outerWaveFrontBand, _latestSliceSize - 1);

        // 其次：对于当前的wavefront，总共有多少个block?
        int totalBlockInWaveFront = max(0, latestSliceIDMax - latestSliceIDMin + 1);

        // globalWorkSize 决定了内核函数会被执行多少次
        size_t totalThread = totalBlockInWaveFront * threadPerBlock;
        size_t globalWorkSize_AllThreadInOneGrid[] = {totalThread};

        if (isDebug) {
            cout << "\n【Start new kernel】\n"
                 << "outerW=" << outerWaveFrontBand
                 << " blocks=" << totalBlockInWaveFront << "■              "
                 << "totalThread=" << totalThread
                 << " latestSliceID=" << latestSliceIDMin << "->" << latestSliceIDMax
                 << " step=" << step << " (in host)" << endl;
        }

        // 每一次参数是有差异的
        err = clSetKernelArg(kernel, 6, sizeof(int), &outerWaveFrontBand);
        err |= clSetKernelArg(kernel, 7, sizeof(int), &totalThread);

        if (err != CL_SUCCESS) {
            cerr << "Error setting kernel arguments." << endl;
            Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        // 执行内核
        err = clEnqueueNDRangeKernel(
                commandQueue,
                kernel,
                1,
                nullptr,
                globalWorkSize_AllThreadInOneGrid,
                localWorkSize_ThreadPerBlock,
                0,
                nullptr,
                nullptr);

        if (err != CL_SUCCESS) {
            cerr << "Error queuing kernel for execution." << endl;
            Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        err = clFinish(commandQueue);
        if (err != CL_SUCCESS) {
            cerr << "Error queuing kernel for execution Finish." << endl;
            Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        if (isDebug) {
            vector<int> newVerWeights(baseVals.size());
            err = clEnqueueReadBuffer(
                    commandQueue,
                    deviceMemObjects[2],
                    CL_TRUE,
                    0,
                    baseVals.size() * sizeof(int),
                    newVerWeights.data(),
                    0,
                    nullptr,
                    nullptr);

            if (err != CL_SUCCESS) {
                cerr << "Error reading result buffer." << endl;
                Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }

            vector<int> newHorWeights(latestVals.size());
            err = clEnqueueReadBuffer(
                    commandQueue,
                    deviceMemObjects[3],
                    CL_TRUE,
                    0,
                    latestVals.size() * sizeof(int),
                    newHorWeights.data(),
                    0,
                    nullptr,
                    nullptr);

            if (err != CL_SUCCESS) {
                cerr << "Error reading result buffer." << endl;
                Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }

            // 打印结果
            stringstream verStream, horStream;
            for (size_t i = 0; i < newVerWeights.size(); ++i) {
                if (i > 0) verStream << ",";
                verStream << newVerWeights[i];
            }

            for (size_t i = 0; i < newHorWeights.size(); ++i) {
                if (i > 0) horStream << ",";
                horStream << newHorWeights[i];
            }

            cout << "vers=" << verStream.str() << endl;
            cout << "hors=" << horStream.str() << endl;
        } // end of if (isDebug)
    } // end of for

    // 读取最终结果
    err = clEnqueueReadBuffer(
            commandQueue,
            deviceMemObjects[2],
            CL_TRUE,
            0,
            baseVals.size() * sizeof(int),
            verWeights.data(),
            0,
            nullptr,
            nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Error reading result buffer." << endl;
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    err = clEnqueueReadBuffer(
            commandQueue,
            deviceMemObjects[3],
            CL_TRUE,
            0,
            latestVals.size() * sizeof(int),
            horWeights.data(),
            0,
            nullptr,
            nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Error reading result buffer." << endl;
        Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
        return;
    }

    Cleanup(context, commandQueue, program, kernel, deviceMemObjects);
}

bool Mega::CreateMemObjects(
        cl_context context,
        cl_mem memObjects[4],
        cl_command_queue commandQueue,
        const vector<int> &bases,
        const vector<int> &latests,
        vector<int> &verWeights,
        vector<int> &horWeights) {

    cl_int err;
    size_t INT_BASE_AXIS_BYTES = bases.size() * sizeof(int);
    size_t INT_LATEST_AXIS_BYTES = latests.size() * sizeof(int);

    // 创建输入缓冲区
    memObjects[0] = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            INT_BASE_AXIS_BYTES,
            const_cast<int *>(bases.data()),
            &err);

    if (err != CL_SUCCESS) {
        cerr << "Error creating base buffer." << endl;
        return false;
    }

    memObjects[1] = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            INT_LATEST_AXIS_BYTES,
            const_cast<int *>(latests.data()),
            &err);

    if (err != CL_SUCCESS) {
        cerr << "Error creating latest buffer." << endl;
        return false;
    }

    // 创建输出缓冲区
    memObjects[2] = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            INT_BASE_AXIS_BYTES,
            nullptr,
            &err);

    if (err != CL_SUCCESS) {
        cerr << "Error creating verWeights buffer." << endl;
        return false;
    }

    memObjects[3] = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            INT_LATEST_AXIS_BYTES,
            nullptr,
            &err);

    if (err != CL_SUCCESS) {
        cerr << "Error creating horWeights buffer." << endl;
        return false;
    }

    // 将初始权重数据写入设备
    err = clEnqueueWriteBuffer(
            commandQueue,
            memObjects[2],
            CL_TRUE,
            0,
            INT_BASE_AXIS_BYTES,
            verWeights.data(),
            0,
            nullptr,
            nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Error writing verWeights to device." << endl;
        return false;
    }

    err = clEnqueueWriteBuffer(
            commandQueue,
            memObjects[3],
            CL_TRUE,
            0,
            INT_LATEST_AXIS_BYTES,
            horWeights.data(),
            0,
            nullptr,
            nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Error writing horWeights to device." << endl;
        return false;
    }

    return true;
}

cl_program Mega::CreateProgram(
        cl_context context,
        cl_device_id device,
        bool IsSharedVersion,
        int _step,
        bool isDebug) {

    string code = IsSharedVersion ? Mega::KernelLCS_Shared : ""; // 简化处理，实际应包含寄存器版本

    // 替换 __STEP__ 宏
    string stepStr = to_string(_step);
    size_t pos = 0;
    while ((pos = code.find("__STEP__", pos)) != string::npos) {
        code.replace(pos, 8, stepStr);
        pos += stepStr.length();
    }

    const char *source = code.c_str();
    size_t sourceSize = code.length();

    cl_int err;
    cl_program program = clCreateProgramWithSource(
            context,
            1,
            &source,
            &sourceSize,
            &err);

    if (err != CL_SUCCESS || program == nullptr) {
        cerr << "Failed to create CL program from source." << endl;
        return nullptr;
    }

    // 编译选项
    const char *compileOptions = isDebug ? "-DDEBUG" : nullptr;
    err = clBuildProgram(
            program,
            1,
            &device,
            compileOptions,
            nullptr,
            nullptr);

    if (err != CL_SUCCESS) {
        // 获取编译日志
        size_t buildLogSize;
        clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                nullptr,
                &buildLogSize);

        vector<char> buildLog(buildLogSize);
        clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                buildLogSize,
                buildLog.data(),
                nullptr);

        cerr << "=============== OpenCL Program Build Info ================" << endl;
        cerr << buildLog.data() << endl;
        cerr << "==========================================================" << endl;

        clReleaseProgram(program);
        return nullptr;
    }

    return program;
}

void Mega::Cleanup(
        cl_context context,
        cl_command_queue commandQueue,
        cl_program program,
        cl_kernel kernel,
        cl_mem memObjects[4]) {

    for (int i = 0; i < 4; i++) {
        if (memObjects[i] != nullptr)
            clReleaseMemObject(memObjects[i]);
    }

    if (commandQueue != nullptr)
        clReleaseCommandQueue(commandQueue);

    if (kernel != nullptr)
        clReleaseKernel(kernel);

    if (program != nullptr)
        clReleaseProgram(program);

    if (context != nullptr)
        clReleaseContext(context);
}

cl_command_queue Mega::CreateCommandQueue(
        cl_context context,
        cl_device_id *device) {

    cl_int err;
    size_t deviceBufferSize;

    err = clGetContextInfo(
            context,
            CL_CONTEXT_DEVICES,
            0,
            nullptr,
            &deviceBufferSize);

    if (err != CL_SUCCESS) {
        cerr << "Failed call to clGetContextInfo(...,CL_CONTEXT_DEVICES,...)" << endl;
        return nullptr;
    }

    if (deviceBufferSize <= 0) {
        cerr << "No devices available." << endl;
        return nullptr;
    }

    vector<cl_device_id> devices(deviceBufferSize / sizeof(cl_device_id));
    err = clGetContextInfo(
            context,
            CL_CONTEXT_DEVICES,
            deviceBufferSize,
            devices.data(),
            nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Failed to get device IDs" << endl;
        return nullptr;
    }

    // 获取设备名称
    char deviceName[1024];
    err = clGetDeviceInfo(
            devices[0],
            CL_DEVICE_NAME,
            sizeof(deviceName),
            deviceName,
            nullptr);

    if (err != CL_SUCCESS) {
        cerr << "Failed to get device name." << endl;
        return nullptr;
    }

    cl_command_queue commandQueue;

#if CL_VERSION_2_0
    // OpenCL 2.0 及以上版本使用新 API
    commandQueue = clCreateCommandQueueWithProperties(
            context,
            devices[0],
            nullptr,
            &err);
#else
    // OpenCL 1.x 版本使用旧 API
    commandQueue = clCreateCommandQueue(
        context,
        devices[0],
        0,
        &err);
#endif

    if (err != CL_SUCCESS || commandQueue == nullptr) {
        cerr << "Failed to create commandQueue for device 0" << endl;
        return nullptr;
    }

    *device = devices[0];
    return commandQueue;
}

int Mega::Valid(
        const vector<int> &originalValues,
        bool IsSharedVersion,
        int step) {

    if (originalValues.empty()) {
        throw runtime_error("originalValues.Length is invalid.");
    }

    if (IsSharedVersion) {
        // 实际测试256比较合适，再大测试用例错误
        if (!(1 <= step && step <= 256)) {
            throw runtime_error("step is invalid.");
        }
    } else {
        // 寄存器向量化最多是int16
        if (step != 2 &&
            step != 4 &&
            step != 8 &&
            step != 16) {
            throw runtime_error("step is invalid.");
        }
    }

    // 不允许step的值超过_originalArray的长度，没有意义
    if (originalValues.size() < (size_t) step)
        throw invalid_argument("N must be less than or equal to the length of the original array.");

    if (originalValues.size() % step != 0) {
        throw invalid_argument("originalValues.Length % step != 0");
    }

    return originalValues.size() / step;
}
