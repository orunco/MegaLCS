## webgpu版本的LCS
{|text:[](./Mega.Cpu.js)|}
{|text:[](./WebGPUPrintf.js)|}
{|text:[](./Mega.Kernel.Shared.js)|}
{|text:[](./Mega.Host.js)|}
{|text:[](./Mega.Fusion.js)|}

## 指令
{|text:[](../../../csharp/MegaLCSLib/OpenCL/Mega.Fusion.cs)|}

bugfix：
CpuLCS_MinMax在js里面是无法直接改变ref参数的，所以返回值。你的Mega.Fusion.js移植版本有问题
比如这里
CpuLCS_MinMax(baseVals, latestVals,
verWeights, horWeights);
return [true, verWeights, horWeights];

返回的不是CpuLCS_MinMax处理后解构的，而是原数组，这是巨大bug
应该是
const 解构= CpuLCS_MinMax(baseVals, latestVals,
verWeights, horWeights);
return [true,解构];
 