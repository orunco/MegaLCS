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

/**
 * CpuLCS_MinMax - 经典LCS的优化版本，模拟KernelLCS_Shared
 * @param {number[]} baseVals - 基准序列（Y轴）
 * @param {number[]} latestVals - 最新序列（X轴）
 * @param {number[]} verWeights - 纵向权重数组（输入输出）
 * @param {number[]} horWeights - 横向权重数组（输入输出）
 */
function CpuLCS_MinMax(baseVals, latestVals, 
                       verWeights, horWeights) {
    if (baseVals.length === 0) throw new Error("CpuLCS(): baseVals数组为空");
    if (latestVals.length === 0) throw new Error("CpuLCS(): latestVals数组为空");
    if (horWeights.length === 0) throw new Error("CpuLCS(): horWeights数组为空");
    if (verWeights.length === 0) throw new Error("CpuLCS(): verWeights数组为空");
    if (baseVals.length !== verWeights.length) throw new Error("CpuLCS(): baseVals数组长度与verWeights数组长度不匹配");
    if (latestVals.length !== horWeights.length) throw new Error("CpuLCS(): latestVals数组长度与horWeights数组长度不匹配");

    for (let b = 0; b < baseVals.length; b++) {
        for (let l = 0; l < latestVals.length; l++) {
            // 左值：当l>0时使用本行前一个值，l=0时使用纵向基础权重
            const leftWeight = verWeights[b];

            // 上值：当b>0时使用上一行同列值，b=0时使用基础权重
            const topWeight = horWeights[l];

            // 计算对角值的三种边界情况【推导可知】
            const leftTopWeight = Math.min(leftWeight, topWeight);

            if (baseVals[b] === latestVals[l]) {
                horWeights[l] = leftTopWeight + 1;
            } else {
                // 不匹配时取max(左值, 上值)
                horWeights[l] = Math.max(leftWeight, topWeight);
            }

            // 每完成一次base/Y轴方向的处理后，当前hors权重的最后一个值需要存储到纵向权重
            verWeights[b] = horWeights[l];
        }
    }

    // console.log("hors: " + horWeights.join(", "));
    // console.log("vers: " + verWeights.join(", ")); 
}

/**
 * CpuLCS_RollLeftTop - 经典LCS的另一种优化版本，模拟KernelLCS_Register
 * @param {number[]} baseVals - 基准序列（Y轴）
 * @param {number[]} latestVals - 最新序列（X轴）
 * @param {number[]} verWeights - 纵向权重数组（输入输出）
 * @param {number[]} horWeights - 横向权重数组（输入输出）
 */
function CpuLCS_RollLeftTop(baseVals, latestVals,
                            verWeights, horWeights) {
    if (baseVals.length === 0) throw new Error("CpuLCS(): baseVals数组为空");
    if (latestVals.length === 0) throw new Error("CpuLCS(): latestVals数组为空");
    if (horWeights.length === 0) throw new Error("CpuLCS(): horWeights数组为空");
    if (verWeights.length === 0) throw new Error("CpuLCS(): verWeights数组为空");
    if (baseVals.length !== verWeights.length) throw new Error("CpuLCS(): baseVals数组长度与verWeights数组长度不匹配");
    if (latestVals.length !== horWeights.length) throw new Error("CpuLCS(): latestVals数组长度与horWeights数组长度不匹配");
    if (horWeights[0] !== verWeights[0]) throw new Error("CpuLCS(): horWeights[0]与verWeights[0]不相等");

    let leftTop = 0;
    let horLBackup = 0;

    for (let b = 0; b < baseVals.length; b++) {
        horLBackup = horWeights[0];

        for (let l = 0; l < latestVals.length; l++) {
            leftTop = horLBackup;
            horLBackup = horWeights[l];

            if (baseVals[b] !== latestVals[l]) {
                horWeights[l] = l !== 0
                    ? Math.max(horWeights[l], horWeights[l - 1])
                    : Math.max(horWeights[l], verWeights[b]);
            } else {
                horWeights[l] = leftTop + 1;
            }
        }

        // 每完成一次base/Y轴方向的处理后，当前hors权重的最后一个值需要存储到纵向权重
        verWeights[b] = horWeights[horWeights.length - 1];
    }

    // console.log("hors: " + horWeights.join(", "));
    // console.log("vers: " + verWeights.join(", ")); 
}

/**
 * CpuLCS_DPMatrix - 小白入门经典版本，用于单元测试
 * @param {number[]} baseVals - 基准序列（Y轴）
 * @param {number[]} latestVals - 最新序列（X轴）
 * @returns {{ verWeights: number[], horWeights: number[] }} 返回纵向和横向权重数组
 */
function CpuLCS_DPMatrix(baseVals, latestVals) {
    if (baseVals.length === 0 || latestVals.length === 0)
        return {verWeights: [], horWeights: []};

    const m = baseVals.length;
    const n = latestVals.length;
    const dp = Array.from({length: m + 1}, () => new Array(n + 1).fill(0));

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (baseVals[i - 1] === latestVals[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
    }

    // 提取最后一行的权重（除了守卫）
    const horWeights = new Array(n);
    for (let j = 0; j < n; j++) {
        horWeights[j] = dp[m][j + 1];
    }

    // 提取最后一列的权重（除了守卫）
    const verWeights = new Array(m);
    for (let i = 0; i < m; i++) {
        verWeights[i] = dp[i + 1][n];
    }

    return {verWeights, horWeights};
}
