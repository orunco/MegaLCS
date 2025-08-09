#include "Mega.h"
#include <algorithm>

int Mega::MegaLCSLen(const vector<int> &baseVals, const vector<int> &latestVals) {
    // 使用默认最佳值
    const int step = 256;

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

    auto result = MegaLCS_Fusion(platformId, deviceId, baseVals, latestVals, step, false);
    auto &horWeights = get<2>(result);

    if (!horWeights.empty()) {
        return horWeights.back();
    }
    return 0;
}

tuple<bool, vector<int>, vector<int>> Mega::MegaLCS_Fusion(
        cl_platform_id platformId,
        cl_device_id deviceId,
        const vector<int> &baseVals,
        const vector<int> &latestVals,
        int step,
        bool isDebug) {

    if (!(1 <= step && step <= 256)) {
        throw runtime_error("step is invalid.");
    }

    // 初始化权重数组
    vector<int> verWeights(baseVals.size(), 0);
    vector<int> horWeights(latestVals.size(), 0);

    // 首先检查是否可以直接使用CpuLCS
    // 如果任意一个序列长度小于等于step，直接使用CPU版本
    if (baseVals.size() <= (size_t) step || latestVals.size() <= (size_t) step) {
        CpuLCS_MinMax(const_cast<int *>(baseVals.data()), baseVals.size(),
                      const_cast<int *>(latestVals.data()), latestVals.size(),
                      verWeights.data(), verWeights.size(),
                      horWeights.data(), horWeights.size());
        return make_tuple(true, verWeights, horWeights);
    }

    // 如果没有找到GPU设备，则全部使用CPU处理
    if (platformId == nullptr || deviceId == nullptr) {
        CpuLCS_MinMax(const_cast<int *>(baseVals.data()), baseVals.size(),
                      const_cast<int *>(latestVals.data()), latestVals.size(),
                      verWeights.data(), verWeights.size(),
                      horWeights.data(), horWeights.size());
        return make_tuple(true, verWeights, horWeights);
    }

    // 计算分块信息
    // baseSliceSize: baseVals可以完整分成多少个step大小的块
    // baseRemainder: baseVals除以step后的余数（最后一块的大小）
    // latestSliceSize: latestVals可以完整分成多少个step大小的块
    // latestRemainder: latestVals除以step后的余数（最后一块的大小）
    int baseSliceSize = baseVals.size() / step;
    int baseRemainder = baseVals.size() % step;
    int latestSliceSize = latestVals.size() / step;
    int latestRemainder = latestVals.size() % step;

    int baseLTSize = baseSliceSize * step;
    int latestLTSize = latestSliceSize * step;

    // 处理左上角规整区域（使用HostLCS）
    // 注意：由于前面的条件判断，这里baseSliceSize和latestSliceSize都必然>0
    // 但为了代码健壮性，仍然保留这个检查
    if (baseSliceSize > 0 && latestSliceSize > 0) {
        vector<int> baseLTVals(baseLTSize);
        vector<int> verLTWeights(baseLTSize);

        vector<int> latestLTVals(latestLTSize);
        vector<int> horLTWeights(latestLTSize);

        copy(baseVals.begin(), baseVals.begin() + baseLTSize, baseLTVals.begin());
        copy(latestVals.begin(), latestVals.begin() + latestLTSize, latestLTVals.begin());

        // 调用HostLCS处理
        HostLCS_WaveFront(platformId, deviceId,
                          baseLTVals, latestLTVals,
                          verLTWeights, horLTWeights,
                          true, step, isDebug);

        // 将权重结果复制回原权重
        copy(verLTWeights.begin(), verLTWeights.end(), verWeights.begin());
        copy(horLTWeights.begin(), horLTWeights.end(), horWeights.begin());
    }

    // 处理右上角
    // 当latest有余数时处理（baseSliceSize必然>0，因为前面条件保证）
    if (latestRemainder > 0) {
        vector<int> baseRTVals(baseLTSize);
        vector<int> verRTWeights(baseLTSize);

        vector<int> latestRTVals(latestRemainder);
        vector<int> horRTWeights(latestRemainder);

        copy(baseVals.begin(), baseVals.begin() + baseRTVals.size(), baseRTVals.begin());
        copy(latestVals.begin() + latestLTSize, latestVals.end(), latestRTVals.begin());

        // 从已计算的权重中获取verWeights的初始值
        copy(verWeights.begin(), verWeights.begin() + verRTWeights.size(), verRTWeights.begin());
        copy(horWeights.begin() + latestLTSize, horWeights.begin() + latestLTSize + horRTWeights.size(),
             horRTWeights.begin());

        CpuLCS_MinMax(baseRTVals.data(), baseRTVals.size(),
                      latestRTVals.data(), latestRTVals.size(),
                      verRTWeights.data(), verRTWeights.size(),
                      horRTWeights.data(), horRTWeights.size());

        // 回填权重
        copy(verRTWeights.begin(), verRTWeights.end(), verWeights.begin());
        copy(horRTWeights.begin(), horRTWeights.end(), horWeights.begin() + latestLTSize);
    }

    // 处理左下角
    // 当base有余数时处理（latestSliceSize必然>0，因为前面条件保证）
    if (baseRemainder > 0) {
        vector<int> baseLBVals(baseRemainder);
        vector<int> verLBWeights(baseRemainder);

        vector<int> latestLBVals(latestLTSize);
        vector<int> horLBWeights(latestLTSize);

        copy(baseVals.begin() + baseLTSize, baseVals.end(), baseLBVals.begin());
        copy(latestVals.begin(), latestVals.begin() + latestLBVals.size(), latestLBVals.begin());

        copy(verWeights.begin() + baseLTSize, verWeights.end(), verLBWeights.begin());
        // 从已计算的权重中获取horWeights的初始值
        copy(horWeights.begin(), horWeights.begin() + horLBWeights.size(), horLBWeights.begin());

        CpuLCS_MinMax(baseLBVals.data(), baseLBVals.size(),
                      latestLBVals.data(), latestLBVals.size(),
                      verLBWeights.data(), verLBWeights.size(),
                      horLBWeights.data(), horLBWeights.size());

        // 更新权重
        copy(verLBWeights.begin(), verLBWeights.end(), verWeights.begin() + baseLTSize);
        copy(horLBWeights.begin(), horLBWeights.end(), horWeights.begin());
    }

    // 处理右下角
    // 当base和latest都有余数时处理
    if (baseRemainder > 0 && latestRemainder > 0) {
        vector<int> baseRBVals(baseRemainder);
        vector<int> verRBWeights(baseRemainder);

        vector<int> latestRBVals(latestRemainder);
        vector<int> horRBWeights(latestRemainder);

        copy(baseVals.begin() + baseLTSize, baseVals.end(), baseRBVals.begin());
        copy(latestVals.begin() + latestLTSize, latestVals.end(), latestRBVals.begin());

        // 从左下和右上的已计算的权重中获取初始值
        copy(verWeights.begin() + baseLTSize, verWeights.end(), verRBWeights.begin());
        copy(horWeights.begin() + latestLTSize, horWeights.end(), horRBWeights.begin());

        CpuLCS_MinMax(baseRBVals.data(), baseRBVals.size(),
                      latestRBVals.data(), latestRBVals.size(),
                      verRBWeights.data(), verRBWeights.size(),
                      horRBWeights.data(), horRBWeights.size());

        // 回填权重
        copy(verRBWeights.begin(), verRBWeights.end(), verWeights.begin() + baseLTSize);
        copy(horRBWeights.begin(), horRBWeights.end(), horWeights.begin() + latestLTSize);
    }

    // 返回最终的LCS权重
    return make_tuple(false, verWeights, horWeights);
}
