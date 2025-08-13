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


// 最终用户使用的版本
async function MegaLCSLen(baseVals, latestVals) {
    // 使用默认最佳值
    const step = 256;

    // 检查WebGPU支持
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported");
    }

    // 获取第一个GPU设备（这里简化处理，实际可能需要更复杂的设备选择逻辑）
    // 在JavaScript中我们无法直接获取platformId和deviceId，但可以检查adapter是否可用
    const hasGPU = true; // 假设系统有GPU

    const [processByCpu, verWeights, horWeights] = await MegaLCS_Fusion(
        baseVals, latestVals,
        step, false);

    return horWeights[horWeights.length - 1];
}

/*
为了方便测试，一般用户不要使用
Mega有2个意义：一个是【可以支持比对任意长度】，HostLCS做不到
二是长度可以到100万甚至更多，CpuLCS做不到
Fusion的意义就是融合Cpu/Host算法

算法大致步骤为：
- 首先看base和latest长度是否都小于step，如果是，那么直接调用CpuLCS即可，否则
- 把base展开到Y轴，latest展开到X轴，整个平面可以划分为4个部分
    - LeftTop是规整的，可以用HostLCS实现。
    - 运行完成后，再运行RightTop(CpuLCS)和LeftBottom(CpuLCS)
    - 最后再运行RightBottom(CpuLCS)
    - 他们之间的数据传递都是靠verWeights和horWeights完成
- 拼接权重，返回权重

 */
async function MegaLCS_Fusion(
    hasGPU,
    baseVals, latestVals,
    step,
    isDebug = false) {
    if (!(1 <= step && step <= 256)) {
        throw new Error("step is invalid.");
    }

    // 初始化权重数组
    const verWeights = new Array(baseVals.length).fill(0);
    const horWeights = new Array(latestVals.length).fill(0);

    // 首先检查是否可以直接使用CpuLCS
    // 如果任意一个序列长度小于等于step，直接使用CPU版本
    if (baseVals.length <= step ||
        latestVals.length <= step) {
        CpuLCS_MinMax(baseVals, latestVals,
            verWeights, horWeights);
        return [true, verWeights, horWeights];
    }

    // 如果没有找到GPU设备，则全部使用CPU处理
    if (!hasGPU) {
        CpuLCS_MinMax(baseVals, latestVals,
            verWeights, horWeights);
        return [true, verWeights, horWeights];
    }

    // 计算分块信息
    // baseSliceSize: baseVals可以完整分成多少个step大小的块
    // baseRemainder: baseVals除以step后的余数（最后一块的大小）
    // latestSliceSize: latestVals可以完整分成多少个step大小的块  
    // latestRemainder: latestVals除以step后的余数（最后一块的大小）
    const baseSliceSize = Math.floor(baseVals.length / step);
    const baseRemainder = baseVals.length % step;
    const latestSliceSize = Math.floor(latestVals.length / step);
    const latestRemainder = latestVals.length % step;

    const baseLTSize = baseSliceSize * step;
    const latestLTSize = latestSliceSize * step;

    // 处理左上角规整区域（使用HostLCS）
    // 注意：由于前面的条件判断，这里baseSliceSize和latestSliceSize都必然>0
    // 但为了代码健壮性，仍然保留这个检查
    if (baseSliceSize > 0 && latestSliceSize > 0) {
        const baseLTVals = baseVals.slice(0, baseLTSize);
        const verLTWeights = verWeights.slice(0, baseLTSize);

        const latestLTVals = latestVals.slice(0, latestLTSize);
        const horLTWeights = horWeights.slice(0, latestLTSize);

        // 调用HostLCS处理
        await HostLCS_WaveFront(
            baseLTVals, latestLTVals,
            verLTWeights, horLTWeights,
            true,
            step,
            isDebug);

        // 将权重结果复制回原权重
        for (let i = 0; i < baseLTSize; i++) {
            verWeights[i] = verLTWeights[i];
        }
        for (let i = 0; i < latestLTSize; i++) {
            horWeights[i] = horLTWeights[i];
        }
    }

    // 处理右上角
    // 当latest有余数时处理（baseSliceSize必然>0，因为前面条件保证）
    if (latestRemainder > 0) {
        const baseRTVals = baseVals.slice(0, baseLTSize);
        const verRTWeights = verWeights.slice(0, baseLTSize);

        const latestRTVals = latestVals.slice(latestLTSize, latestLTSize + latestRemainder);
        const horRTWeights = horWeights.slice(latestLTSize, latestLTSize + latestRemainder);

        CpuLCS_MinMax(baseRTVals, latestRTVals,
            verRTWeights, horRTWeights);

        // 回填权重
        for (let i = 0; i < baseLTSize; i++) {
            verWeights[i] = verRTWeights[i];
        }
        for (let i = 0; i < latestRemainder; i++) {
            horWeights[latestLTSize + i] = horRTWeights[i];
        }
    }

    // 处理左下角
    // 当base有余数时处理（latestSliceSize必然>0，因为前面条件保证）
    if (baseRemainder > 0) {
        const baseLBVals = baseVals.slice(baseLTSize, baseLTSize + baseRemainder);
        const verLBWeights = verWeights.slice(baseLTSize, baseLTSize + baseRemainder);

        const latestLBVals = latestVals.slice(0, latestLTSize);
        const horLBWeights = horWeights.slice(0, latestLTSize);

        CpuLCS_MinMax(baseLBVals, latestLBVals,
            verLBWeights, horLBWeights);

        // 更新权重
        for (let i = 0; i < baseRemainder; i++) {
            verWeights[baseLTSize + i] = verLBWeights[i];
        }
        for (let i = 0; i < latestLTSize; i++) {
            horWeights[i] = horLBWeights[i];
        }
    }

    // 处理右下角
    // 当base和latest都有余数时处理
    if (baseRemainder > 0 && latestRemainder > 0) {
        const baseRBVals = baseVals.slice(baseLTSize, baseLTSize + baseRemainder);
        const verRBWeights = verWeights.slice(baseLTSize, baseLTSize + baseRemainder);

        const latestRBVals = latestVals.slice(latestLTSize, latestLTSize + latestRemainder);
        const horRBWeights = horWeights.slice(latestLTSize, latestLTSize + latestRemainder);

        CpuLCS_MinMax(baseRBVals, latestRBVals,
            verRBWeights, horRBWeights);

        // 回填权重
        for (let i = 0; i < baseRemainder; i++) {
            verWeights[baseLTSize + i] = verRBWeights[i];
        }
        for (let i = 0; i < latestRemainder; i++) {
            horWeights[latestLTSize + i] = horRBWeights[i];
        }
    }

    // 返回最终的LCS权重
    return [false, verWeights, horWeights];
}
 
