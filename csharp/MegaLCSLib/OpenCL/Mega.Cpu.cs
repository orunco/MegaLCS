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

using System.Runtime.CompilerServices;

namespace MegaLCSLib.OpenCL;

public partial class Mega{
    /*
    这个函数就是经典LCS的最优化、升级版本，更加具有竞争力，命名为CpuLCS，
    同时也是KernelLCS_Shared的原型
    因为内核很难调试，所以这个函数可以进行仿真验证
    bases可以看成是展开到Y轴; latest可以看成是展开到X轴;
    vers存储原DP的纵向权重，【是输入也是输出】
    hors存储原DP的横向权重，类似滚动数组；【是输入也是输出】
    这个函数的hors和vers是有基础权重的,不一定为0，且horWeights[0]可以和verWeights[0]不相等，因此DataIndependent表达了这个意义
    当hosrs和vers为0时退化到经典LCS
     */
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CpuLCS_MinMax(
        int[] baseVals, int[] latestVals,
        int[] verWeights, int[] horWeights){
        // 先做校验，这个是由理论分析后的结果，必须满足
        if (baseVals.Length == 0){
            throw new Exception("CpuLCS(): baseVals数组为空");
        }

        if (latestVals.Length == 0){
            throw new Exception("CpuLCS(): latestVals数组为空");
        }

        if (horWeights.Length == 0){
            throw new Exception("CpuLCS(): horWeights数组为空");
        }

        if (verWeights.Length == 0){
            throw new Exception("CpuLCS(): verWeights数组为空");
        }

        if (baseVals.Length != verWeights.Length){
            throw new Exception("CpuLCS(): baseVals数组长度与verWeights数组长度不匹配");
        }

        if (latestVals.Length != horWeights.Length){
            throw new Exception("CpuLCS(): latestVals数组长度与horWeights数组长度不匹配");
        }

        for (var b = 0; b < baseVals.Length; b++){
            for (var l = 0; l < latestVals.Length; l++){
                // 左值：当l>0时使用本行前一个值，l=0时使用纵向基础权重
                // 特殊点：左侧无元素，和基础权重vers[b]比较，而不是和0比较
                int leftWeight = verWeights[b];

                // 上值：当b>0时使用上一行同列值
                // 特殊点：当b=0，使用基础权重
                int topWeight = horWeights[l];

                // 计算对角值的三种边界情况【推导可知】
                int leftTopWeight = Math.Min(leftWeight, topWeight);

                if (baseVals[b] == latestVals[l]){
                    horWeights[l] = leftTopWeight + 1;
                }
                else{
                    // 不匹配时取max(左值, 上值)
                    horWeights[l] = Math.Max(leftWeight, topWeight);
                }

                // 每完成一次base/Y轴方向的处理后，当前hors权重的最后一个值需要存储到纵向权重
                verWeights[b] = horWeights[l];
            }
        }

        // Console.WriteLine("hors: " + string.Join(", ", horWeights));
        // Console.WriteLine("vers: " + string.Join(", ", verWeights));
    }

    /*
    这个函数同样也是经典LCS的优化版本，
    同时也是KernelLCS_Register的原型
    因为内核很难调试，所以这个函数可以进行仿真验证
    和上面函数的差别在于horWeights[0]可以和verWeights[0]必须相等
    这个约束条件太麻烦了

    注意：虽然有约束条件，但是也比一维滚动数组的内存占用要小一点点
     */
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CpuLCS_RollLeftTop(
        int[] baseVals, int[] latestVals,
        int[] verWeights, int[] horWeights){
        // 先做校验，这个是由理论分析后的结果，必须满足
        if (baseVals.Length == 0){
            throw new Exception("CpuLCS(): baseVals数组为空");
        }

        if (latestVals.Length == 0){
            throw new Exception("CpuLCS(): latestVals数组为空");
        }

        if (horWeights.Length == 0){
            throw new Exception("CpuLCS(): horWeights数组为空");
        }

        if (verWeights.Length == 0){
            throw new Exception("CpuLCS(): verWeights数组为空");
        }

        if (baseVals.Length != verWeights.Length){
            throw new Exception("CpuLCS(): baseVals数组长度与verWeights数组长度不匹配");
        }

        if (latestVals.Length != horWeights.Length){
            throw new Exception("CpuLCS(): latestVals数组长度与horWeights数组长度不匹配");
        }

        // 这个算法必须要求一致
        if (horWeights[0] != verWeights[0]){
            throw new Exception("CpuLCS(): horWeights[0]与verWeights[0]不相等");
        }

        var leftTop = 0; // 模拟DP左上角初始值
        var horLBackup = 0; // 备份当前值，供下一轮使用

        for (var b = 0; b < baseVals.Length; b++){
            // 每一行的初始权重不是0，而是hors里面的第0个元素开始的,和参数b无关
            horLBackup = horWeights[0];

            for (var l = 0; l < latestVals.Length; l++){
                leftTop = horLBackup;
                horLBackup = horWeights[l];

                // 高度优化：不相等的先命中
                if (baseVals[b] != latestVals[l]){
                    // 高度优化：不相等的先命中
                    horWeights[l] = l != 0
                        ? Math.Max(horWeights[l], horWeights[l - 1])
                        : Math.Max(horWeights[l], verWeights[b]);
                }
                else{
                    horWeights[l] = leftTop + 1;
                }
                // if (bases[b] == latests[l]){
                //     hors[l] = leftTop + 1;
                // }
                // else{
                //     if (l == 0){
                //         // 特殊点：左侧无元素，和基础权重vers[b]比较，而不是和0比较
                //         hors[l] = Math.Max(hors[l], vers[b]);
                //     }
                //     else{
                //         // 比较当前 hors[l] 和左侧元素 hors[l-1]
                //         hors[l] = Math.Max(hors[l], hors[l - 1]);
                //     }
                // }
            }

            // 每完成一次base/Y轴方向的处理后，当前hors权重的最后一个值需要存储到纵向权重
            verWeights[b] = horWeights[^1];
        }

        // Console.WriteLine("hors: " + string.Join(", ", horWeights));
        // Console.WriteLine("vers: " + string.Join(", ", verWeights));
    }

    // 小白入门经典版本，同时用于单元测试
    public static (int[] verWeights, int[] horWeights) CpuLCS_DPMatrix(
        int[] baseVals, int[] latestVals){
        if (baseVals.Length == 0 || latestVals.Length == 0)
            return (new int[0], new int[0]);

        var m = baseVals.Length;
        var n = latestVals.Length;
        var dp = new int[m + 1, n + 1];

        for (var i = 1; i <= m; i++){
            for (var j = 1; j <= n; j++){
                if (baseVals[i - 1] == latestVals[j - 1])
                    dp[i, j] = dp[i - 1, j - 1] + 1;
                else
                    dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
            }
        }

        // 提取最后一行的权重（除了守卫）
        var horWeights = new int[n];
        for (var j = 0; j < n; j++){
            horWeights[j] = dp[m, j + 1];
        }

        // 提取最后一列的权重（除了守卫）
        var verWeights = new int[m];
        for (var i = 0; i < m; i++){
            verWeights[i] = dp[i + 1, n];
        }

        return (verWeights, horWeights);
    }
}