## 上下文
首先请你深入阅读代码：
{|text:[](../../MegaLCSLib/OpenCL/Mega.Cpu.cs)|}
其中经典的LCS算法ClassicLCS，以及优化升级版CpuLCS_GoRightBottom，他实现了同时输入值和基础权重，以及最终输出修改后的权重。

其次我已经实现了GPU版本的HostLCS_WaveFront，可以实现百万数据的LCS，签名如下：

``` csharp
// 运行HostLCS, 输入的Array必须是STEP的倍数, HostLCS调用KernelLCS
public static unsafe void HostLCS_WaveFront(
        IntPtr platformId,
        IntPtr deviceId,
        int[] baseVals,
        int[] latestVals,
        int[] verWeights,
        int[] horWeights,
        bool isSharedVersion,
        int step,
        bool isDebug = false)
```
platformId和deviceId先不用管，是OpenCL环境，由外部代码传入。
isSharedVersion默认为true，isDebug = false
这个函数同样实现了LCS的功能：
baseVals和latestVals为输入值，verWeights和horWeights同样为输入和输出。
但是差异在于：baseVals和latestVals有一定的要求，如下：
`var _baseSliceSize = Valid(baseVals, isSharedVersion, step);
 var _latestSliceSize = Valid(latestVals, isSharedVersion, step);`

```csharp
public static int Valid(
        int[] originalValues,
        bool IsSharedVersion,
        int step){
        if (originalValues.Length == 0){
            throw new Exception("originalValues.Length is invalid.");
        }

        if (IsSharedVersion){
            // 实际测试256比较合适，再大测试用例错误
            if (!(1 <= step && step <= 256)){
                throw new Exception("step is invalid.");
            }
        }
        else{
            // 寄存器向量化最多是int16
            if (step != 2 &&
                step != 4 &&
                step != 8 &&
                step != 16){
                throw new Exception("step is invalid.");
            }
        }

        // 不允许step的值超过_originalArray的长度，没有意义
        if (originalValues.Length < step)
            throw new ArgumentException("N must be less than or equal to the length of the original array.");

        if (originalValues.Length % step != 0){
            throw new ArgumentException("originalValues.Length % step != 0");
        }

        return originalValues.Length / step;
    }
```
也就是有一定的约束。
STEP约束为2 4 8 16 ... 256

关于IntPtr platformId, IntPtr deviceId从哪里获取的问题，有现成的代码：
{|text:[](../../MegaLCSLib/OpenCL/Mega.Devices.cs)|}

## 指令
请你代码检视
{|text:[](../../MegaLCSLib/OpenCL/Mega.cs)|}

逻辑上是否有错误？

