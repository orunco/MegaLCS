/*
Copyright (C) 2025 Pete Zhang, rivxer@gmail.com, https://github.com/orunco

Licensed under the Apache License, Version 2.0 (the ""License"");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an ""AS IS"" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

using Silk.NET.OpenCL;

namespace MegaLCSLib.OpenCL;

public partial class Mega{
    // 最终用户使用的版本
    public static int MegaLCS(int[] baseVals, int[] latestVals){
        // 使用默认最佳值
        const int step = 256;

        // 获取第一个GPU设备，当然如果CPU够强，也可以
        var allDevices = GetAllDevices();
        var platformId = IntPtr.Zero;
        var deviceId = IntPtr.Zero;
        foreach (var device in allDevices){
            if (device.deviceType == DeviceType.Gpu){
                platformId = device.platformId;
                deviceId = device.deviceId;
                break;
            }
        }

        var (processByCpu, verWeights, horWeights) = MegaLCS_Fusion(
            platformId, deviceId,
            baseVals, latestVals,
            step, false);

        return horWeights[^1];
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
    public static (bool processByCpu,
        int[] verWeights,
        int[] horWeights)
        MegaLCS_Fusion(
            IntPtr platformId,
            IntPtr deviceId,
            int[] baseVals, int[] latestVals,
            int step,
            bool isDebug = false){
        if (!(1 <= step && step <= 256)){
            throw new Exception("step is invalid.");
        }

        // 初始化权重数组
        var verWeights = new int[baseVals.Length];
        var horWeights = new int[latestVals.Length];

        // 首先检查是否可以直接使用CpuLCS
        // 如果任意一个序列长度小于等于step，直接使用CPU版本
        if (baseVals.Length <= step ||
            latestVals.Length <= step){
            CpuLCS_MinMax(baseVals, latestVals,
                verWeights, horWeights);
            return (true, verWeights, horWeights);
        }

        // 如果没有找到GPU设备，则全部使用CPU处理
        if (platformId == IntPtr.Zero ||
            deviceId == IntPtr.Zero){
            CpuLCS_MinMax(baseVals, latestVals,
                verWeights, horWeights);
            return (true, verWeights, horWeights);
        }

        // 计算分块信息
        // baseSliceSize: baseVals可以完整分成多少个step大小的块
        // baseRemainder: baseVals除以step后的余数（最后一块的大小）
        // latestSliceSize: latestVals可以完整分成多少个step大小的块  
        // latestRemainder: latestVals除以step后的余数（最后一块的大小）
        var baseSliceSize = baseVals.Length / step;
        var baseRemainder = baseVals.Length % step;
        var latestSliceSize = latestVals.Length / step;
        var latestRemainder = latestVals.Length % step;

        var baseLTSize = baseSliceSize * step;
        var latestLTSize = latestSliceSize * step;

        // 处理左上角规整区域（使用HostLCS）
        // 注意：由于前面的条件判断，这里baseSliceSize和latestSliceSize都必然>0
        // 但为了代码健壮性，仍然保留这个检查
        if (baseSliceSize > 0 && latestSliceSize > 0){
            var baseLTVals = new int[baseLTSize];
            var verLTWeights = new int[baseLTSize];

            var latestLTVals = new int[latestLTSize];
            var horLTWeights = new int[latestLTSize];

            Array.Copy(baseVals, 0,
                baseLTVals, 0,
                baseLTSize);
            Array.Copy(latestVals, 0,
                latestLTVals, 0,
                latestLTSize);

            // 调用HostLCS处理
            HostLCS_WaveFront(
                platformId, deviceId,
                baseLTVals, latestLTVals,
                verLTWeights, horLTWeights,
                true,
                step,
                isDebug);

            // 将权重结果复制回原权重
            Array.Copy(verLTWeights, 0,
                verWeights, 0,
                baseLTSize);
            Array.Copy(horLTWeights, 0,
                horWeights, 0,
                latestLTSize);
        }

        // 处理右上角
        // 当latest有余数时处理（baseSliceSize必然>0，因为前面条件保证）
        if (latestRemainder > 0){
            var baseRTVals = new int[baseLTSize];
            var verRTWeights = new int[baseLTSize];

            var latestRTVals = new int[latestRemainder];
            var horRTWeights = new int[latestRemainder];

            Array.Copy(baseVals, 0,
                baseRTVals, 0,
                baseRTVals.Length);
            Array.Copy(latestVals, latestLTSize,
                latestRTVals, 0,
                latestRTVals.Length);

            // 从已计算的权重中获取verWeights的初始值
            Array.Copy(verWeights, 0,
                verRTWeights, 0,
                verRTWeights.Length);
            Array.Copy(horWeights, latestLTSize,
                horRTWeights, 0,
                horRTWeights.Length);

            CpuLCS_MinMax(baseRTVals, latestRTVals,
                verRTWeights, horRTWeights);

            // 回填权重
            Array.Copy(verRTWeights, 0,
                verWeights, 0,
                verRTWeights.Length);
            Array.Copy(horRTWeights, 0,
                horWeights, latestLTSize,
                horRTWeights.Length);
        }

        // 处理左下角
        // 当base有余数时处理（latestSliceSize必然>0，因为前面条件保证）
        if (baseRemainder > 0){
            var baseLBVals = new int[baseRemainder];
            var verLBWeights = new int[baseRemainder];

            var latestLBVals = new int[latestLTSize];
            var horLBWeights = new int[latestLTSize];

            Array.Copy(baseVals, baseLTSize,
                baseLBVals, 0,
                baseLBVals.Length);
            Array.Copy(latestVals, 0,
                latestLBVals, 0,
                latestLBVals.Length);

            Array.Copy(verWeights, baseLTSize,
                verLBWeights, 0,
                verLBWeights.Length);
            // 从已计算的权重中获取horWeights的初始值
            Array.Copy(horWeights, 0,
                horLBWeights, 0,
                horLBWeights.Length);

            CpuLCS_MinMax(baseLBVals, latestLBVals,
                verLBWeights, horLBWeights);

            // 更新权重
            Array.Copy(verLBWeights, 0,
                verWeights, baseLTSize,
                verLBWeights.Length);
            Array.Copy(horLBWeights, 0,
                horWeights, 0,
                horLBWeights.Length);
        }

        // 处理右下角
        // 当base和latest都有余数时处理
        if (baseRemainder > 0 && latestRemainder > 0){
            var baseRBVals = new int[baseRemainder];
            var verRBWeights = new int[baseRemainder];

            var latestRBVals = new int[latestRemainder];
            var horRBWeights = new int[latestRemainder];

            Array.Copy(baseVals, baseLTSize,
                baseRBVals, 0,
                baseRBVals.Length);
            Array.Copy(latestVals, latestLTSize, latestRBVals, 0,
                latestRBVals.Length);

            // 从左下和右上的已计算的权重中获取初始值
            Array.Copy(verWeights, baseLTSize,
                verRBWeights, 0,
                verRBWeights.Length);
            Array.Copy(horWeights, latestLTSize,
                horRBWeights, 0,
                horRBWeights.Length);

            CpuLCS_MinMax(baseRBVals, latestRBVals,
                verRBWeights, horRBWeights);

            // 回填权重
            Array.Copy(verRBWeights, 0,
                verWeights, baseLTSize,
                verRBWeights.Length);
            Array.Copy(horRBWeights, 0, horWeights, latestLTSize,
                horRBWeights.Length);
        }

        // 返回最终的LCS权重
        return (false, verWeights, horWeights);
    }
}