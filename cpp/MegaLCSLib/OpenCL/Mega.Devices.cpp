#include "Mega.h"

vector<tuple<cl_platform_id, cl_device_id, string, cl_device_type>> Mega::GetAllDevices() {
    vector<tuple<cl_platform_id, cl_device_id, string, cl_device_type>> result;

    // 获取平台数量
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);

    if (err != CL_SUCCESS) {
        return result;
    }

    if (platformCount == 0) {
        return result;
    }

    // 获取平台ID
    vector<cl_platform_id> platformIds(platformCount);
    err = clGetPlatformIDs(platformCount, platformIds.data(), nullptr);

    if (err != CL_SUCCESS) {
        return result;
    }

    // 遍历每个平台
    for (cl_uint p = 0; p < platformCount; p++) {
        cl_platform_id platformId = platformIds[p];

        // 获取设备数量
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

        if (err != CL_SUCCESS || deviceCount == 0) {
            continue;
        }

        // 获取设备ID
        vector<cl_device_id> deviceIds(deviceCount);
        err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr);

        if (err != CL_SUCCESS) {
            continue;
        }

        // 遍历每个设备
        for (cl_uint d = 0; d < deviceCount; d++) {
            cl_device_id deviceId = deviceIds[d];

            // 获取设备名称
            char deviceName[1024];
            size_t deviceNameSize = 0;
            err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(deviceName), deviceName, &deviceNameSize);

            if (err != CL_SUCCESS) {
                continue;
            }

            // 确保字符串正确结束
            deviceName[sizeof(deviceName) - 1] = '\0';
            string deviceNameString(deviceName);

            // 获取设备类型
            cl_device_type deviceType = 0;
            err = clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);

            if (err != CL_SUCCESS) {
                continue;
            }

            result.push_back(make_tuple(platformId, deviceId, deviceNameString, deviceType));
        }
    }

    return result;
}

pair<cl_platform_id, cl_device_id> Mega::GetFirstGpuDevice() {
    // 获取第一个GPU设备，当然如果CPU够强，也可以
    auto allDevices = GetAllDevices();
    cl_platform_id platformId = nullptr;
    cl_device_id deviceId = nullptr;

    for (const auto &device: allDevices) {
        if (get<3>(device) == CL_DEVICE_TYPE_GPU) {
            platformId = get<0>(device);
            deviceId = get<1>(device);
            break;
        }
    }

    return make_pair(platformId, deviceId);
}
