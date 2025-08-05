using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using NUnit.Framework;

namespace MegaLCSTest.OpenCL;


/*
65536*65536,STEP=256规模:
Using device: AMD R5700G自带的也是[gfx90c]说明这就是集成显卡的极限，这应该就是集显的性能下限
https://github.com/ROCm/ROCR-Runtime/issues/180 linux下amd显卡的问题特别多
Using device: Thinkpad X13 4650U笔记本自带显卡[gfx90c]   Execution time:      227 ms
Using device: amd R5700U笔记本自带显卡[gfx902]半斤八两    Execution time:      470 ms
Using device: E590笔记本附带的显卡Radeon 500[gfx803]     Execution time:      466 ms AMD集成显卡半斤八两
Using device: E590笔记本附带的显卡Intel(R) UHD620        Execution time:    1 932 ms intel不太行
Using device: Intel(R) HD Graphics 4600(十年前显卡)     Execution time:    2 029 ms intel不太行
Using device: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz Execution time:    7 080 ms intel不太行
Using device: Apple [Intel(R) Core(TM) i5-4258U CPU @ 2.40GHz]: 出现执行错误，说明intel是有多差
Using device: Apple 2015笔记本[Iris]                   Execution time:    1 368 ms 早年的不太行

1048576*1048576,STEP=256规模，共享内存DualWaveFront版本
Using device: Tesla P40                                Execution time:    9 500 ms 其实也一般般
Using device: Thinkpad X13 4650U笔记本自带显卡[gfx90c]   Execution time:   71 383 ms
Using device: amd R5700U笔记本自带显卡[gfx902]           Execution time:   44 089 ms
Using device: E590笔记本附带的显卡Radeon 500[gfx803]     Execution time:   42 189 ms 半斤八两
Using device: E590笔记本附带的显卡Intel(R) UHD620        Execution time:  247 679 ms【intel显卡确实不太行】
Using device: Intel(R) HD Graphics 4600(十年前显卡)     Execution time:  451 231 ms
Using device: Intel(R) Core(TM) i5-4590 CPU @ 3.30GHz Execution time: 1771 419 ms(海量线程压垮了CPU)

 */


/*
20250206 实际测试
BenchmarkDotNet v0.13.12, Ubuntu 22.04.4 LTS (Jammy Jellyfish)
AMD Ryzen 7 5700G with Radeon Graphics, 1 CPU, 16 logical and 8 physical cores
.NET SDK 8.0.100
  [Host]     : .NET 8.0.0 (8.0.23.53103), X64 RyuJIT AVX2
  Job-DYBUIO : .NET 8.0.0 (8.0.23.53103), X64 RyuJIT AVX2
  Job-BZNPUY : .NET 8.0.0, X64 NativeAOT AVX2

IterationCount=3  LaunchCount=3  WarmupCount=3  

| Method                    | Runtime       | MAX   | STEP | Mean       | Error    | Allocated  |
|-------------------------- |-------------- |------ |----- |-----------:|---------:|-----------:|
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 1    | 8,529.2 ms | 13.04 ms | 8739.49 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 1    | 8,532.3 ms | 10.59 ms | 8739.41 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 2    | 2,580.2 ms |  4.26 ms | 4643.49 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 2    | 2,589.6 ms |  9.61 ms | 4643.41 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 4    |   986.6 ms |  2.22 ms | 2595.49 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 4    |   991.0 ms |  6.95 ms | 2595.41 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 8    |   511.5 ms |  2.81 ms | 1571.49 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 8    |   511.4 ms |  4.67 ms | 1571.41 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 16   |   302.1 ms |  2.66 ms | 1059.18 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 16   |   302.0 ms |  2.28 ms | 1059.11 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 32   |   199.1 ms |  1.45 ms |  803.06 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 32   |   196.0 ms |  2.15 ms |  802.99 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 64   |   162.1 ms |  2.14 ms |     675 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 64   |   162.3 ms |  1.65 ms |  674.93 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 128  |   144.3 ms |  2.22 ms |  611.05 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 128  |   144.6 ms |  1.19 ms |  610.98 KB | 
| perf_diff_main_all_same_0 | .NET 8.0      | 65536 | 256  |   139.1 ms |  2.37 ms |  579.05 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536 | 256  |   139.4 ms |  1.38 ms |  578.98 KB |

STEP 256确实是最优

IterationCount=3  LaunchCount=3  WarmupCount=3  

| Method                    | Runtime       | MAX     | STEP | Mean         | Allocated   |
|-------------------------- |-------------- |-------- |----- |-------------:|------------:|
| perf_diff_main_all_same_0 | .NET 8.0      | 65536   | 256  |     138.3 ms |   579.05 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 65536   | 256  |     137.8 ms |   578.98 KB |
| perf_diff_main_all_same_0 | .NET 8.0      | 1048576 | 256  |   9,532.7 ms |  8739.59 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 1048576 | 256  |   9,533.7 ms |  8739.51 KB |
| perf_diff_main_all_same_0 | .NET 8.0      | 2097152 | 256  |  36,642.7 ms | 17443.59 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 2097152 | 256  |  36,642.4 ms | 17443.51 KB |
| perf_diff_main_all_same_0 | NativeAOT 8.0 | 4194304 | 256  | 142,976.1 ms | 34851.51 KB |

真实的100万大概10秒，200万36秒（理论应该是40秒），400万142秒（理论应该是160秒）

 */
[SimpleJob(RuntimeMoniker.Net80, baseline: true,
    launchCount: 3, warmupCount: 3, iterationCount: 3)]
[SimpleJob(RuntimeMoniker.NativeAot80,
    launchCount: 3, warmupCount: 3, iterationCount: 3)]
[MemoryDiagnoser]
public class Perf_Opencl{
    int[] inputArray;
    int[] expectArray;

    [Params(65536,1048576,2097152,4194304)] public int MAX = 1024;
    [Params(256)] public int STEP = 1;
    
    [OneTimeSetUp]
    [GlobalSetup]
    public void Setup(){
        inputArray = new int[MAX];
        expectArray = new int[MAX];
        for (int i = 0; i < MAX; i++){
            expectArray[i] = i + 1;
        }
    }

    //相同是如此彻底
    //1 AB所有字符全部相同 所有数值都为0
    [Test]
    [Benchmark]
    public void perf_diff_main_all_same_0(){
        Test_RegularArray_Shared(inputArray, inputArray,
            inputArray, inputArray,
            expectArray, expectArray,
            STEP);
    }

    private static void Test_RegularArray_Shared(
        int[] baseVals,
        int[] latestVals,
        int[] verWeights,
        int[] horWeights,
        int[] versOutExpect,
        int[] horsOutExpect,
        int step){
        var allDevices = MegaLCSLib.OpenCL.MegaLCS.GetAllDevices();
        foreach (var device in allDevices){
            if (device.name != "Tesla P40"){
                return;
            }

            // Console.WriteLine($"Run on device : {device.name}");

            // 由于这个函数会直接修改权重数组，而测试是多设备反复的
            var versOut = (int[])verWeights.Clone();
            var horsOut = (int[])horWeights.Clone();

            MegaLCSLib.OpenCL.MegaLCS.KernelLCS(
                device.platformId,
                device.deviceId,
                baseVals,
                latestVals,
                versOut,
                horsOut,
                true,
                step,
                false
            );
        }
    }
}