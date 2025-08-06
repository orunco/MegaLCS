using NUnit.Framework;

namespace MegaLCSTest.OpenCL;

public class Mega_Kernel_Shared_Test{
    [Test]
    public void Test_classic(){
        // STEP = 1 原始经典值
        {
            int[] bases =['A', 'B', 'C', 'B', 'D', 'A', 'B'];
            int[] latests =['B', 'D', 'C', 'A', 'B', 'C'];
            int[] inputVerW = new int[bases.Length];
            int[] inputHorW = new int[latests.Length];
            int[] expectVers =[1, 2, 3, 3, 3, 3, 4];
            int[] expectHors =[1, 2, 2, 3, 4, 4];
            Test_KernelLCS_Shared(bases, latests,
                inputVerW, inputHorW,
                expectVers, expectHors, 1, true);
        }

        // STEP = 2
        {
            int[] bases =['A', 'B', 'C', 'B', 'D', 'A'];
            int[] latests =['B', 'D', 'C', 'A', 'B', 'C'];
            int[] inputVerW = new int[bases.Length];
            int[] inputHorW = new int[latests.Length];
            int[] expectVers =[1, 2, 3, 3, 3, 3];
            int[] expectHors =[1, 2, 2, 3, 3, 3];
            Test_KernelLCS_Shared(bases, latests,
                inputVerW, inputHorW,
                expectVers, expectHors, 1, true);
        }

        // STEP = 3
        {
            int[] bases =['A', 'B', 'C', 'B', 'D', 'A'];
            int[] latests =['B', 'D', 'C', 'A', 'B', 'C'];
            int[] inputVerW = new int[bases.Length];
            int[] inputHorW = new int[latests.Length];
            int[] expectVers =[1, 2, 3, 3, 3, 3];
            int[] expectHors =[1, 2, 2, 3, 3, 3];
            Test_KernelLCS_Shared(bases, latests,
                inputVerW, inputHorW,
                expectVers, expectHors, 1, true);
        }

        // STEP = 6
        {
            int[] bases =['A', 'B', 'C', 'B', 'D', 'A'];
            int[] latests =['B', 'D', 'C', 'A', 'B', 'C'];
            int[] inputVerW = new int[bases.Length];
            int[] inputHorW = new int[latests.Length];
            int[] expectVers =[1, 2, 3, 3, 3, 3];
            int[] expectHors =[1, 2, 2, 3, 3, 3];
            Test_KernelLCS_Shared(bases, latests,
                inputVerW, inputHorW,
                expectVers, expectHors, 1, true);
        }
    }

    [Test]
    public void Test_RegularArray1x1(){
        // 无基础权重的测试用例
        Test_KernelLCS_Shared([5], [5], [0], [0],
            [1], [1], 1);
        Test_KernelLCS_Shared([5], [6], [0], [0],
            [0], [0], 1);

        // 有基础权重的测试用例
        Test_KernelLCS_Shared([5], [5], [10], [10],
            [11], [11], 1);
        Test_KernelLCS_Shared([5], [6], [10], [10],
            [10], [10], 1);
    }

    [Test]
    public void Test_RegularArray2x2_step1(){
        // 无基础权重的测试用例
        Test_KernelLCS_Shared([5, 6], [5, 6], [0, 0], [0, 0],
            [1, 2], [1, 2], 1);
        Test_KernelLCS_Shared([5, 6], [7, 8], [0, 0], [0, 0],
            [0, 0], [0, 0], 1);

        // 有基础权重的测试用例
        Test_KernelLCS_Shared([5, 6], [5, 6], [10, 20], [30, 40],
            [40, 21], [20, 21], 1);
        Test_KernelLCS_Shared([5, 6], [7, 8], [10, 20], [30, 40],
            [40, 40], [30, 40], 1);
    }

    [Test]
    public void Test_Other(){
        // 这个测试用例的基础权重值虽然很奇葩，但是可以很清晰的解释核函数内部发生的流程，因为权重不符合实际，所以结果和正确的权重也有差异
        Test_KernelLCS_Shared([5, 6, 5], [5, 6]
            , [30, 40, 50], [10, 20]
            , [20, 21, 41], [41, 41], 1, true);
        
        Test_KernelLCS_Shared([5, 6], [5, 6]
            , [30, 40], [10, 20]
            , [20, 21], [40, 21], 1, true);
        // 下面这个测试是可以通过的，结果是正确的
        Test_KernelLCS_Shared([5, 6], [5, 6]
            , [11, 12], [10, 11]
            , [11, 12], [12, 12], 1, true);

        Test_KernelLCS_Shared([5, 6, 7, 8], [5, 6, 7, 8]
            , [11, 12, 13, 14], [10, 11, 12, 13]
            , [13, 13, 13, 14], [14, 14, 14, 14], 2, true);

        // 需要考虑虚拟机，所以只能是小的
        var arrayMax = 65536;
        var inputArray = new int[arrayMax];
        var expectArray = new int[arrayMax];
        for (int i = 0; i < arrayMax; i++){
            expectArray[i] = i + 1;
        }

        Test_KernelLCS_Shared(inputArray, inputArray,
            inputArray, inputArray,
            expectArray, expectArray,
            256, false);
    }

    private static void Test_KernelLCS_Shared(
        int[] baseVals,
        int[] latestVals,
        int[] verWeights,
        int[] horWeights,
        int[] versOutExpect,
        int[] horsOutExpect,
        int step,
        bool isDebug = false){
        var allDevices =
            MegaLCSLib.OpenCL.Mega.GetAllDevices();

        foreach (var device in allDevices){
            if (isDebug){
                Console.WriteLine($"Run on device : {device.name}");
            }

            var stopwatch = System.Diagnostics.Stopwatch.StartNew(); // 开始计时

            // 由于这个函数会直接修改权重数组，而测试是多设备反复的
            var versOut = (int[])verWeights.Clone();
            var horsOut = (int[])horWeights.Clone();

            MegaLCSLib.OpenCL.Mega.KernelLCS(
                device.platformId,
                device.deviceId,
                baseVals,
                latestVals,
                versOut,
                horsOut,
                true,
                step,
                isDebug
            );

            // Output the result buffer
            // for (int i = 0; i < ARRAY_SIZE; i++){
            //     Console.WriteLine(result[i]);
            // }
            stopwatch.Stop(); // 结束计时
            Console.WriteLine($"Execution time: {stopwatch.ElapsedMilliseconds} ms, Using device: [{device.name}] ");
            Console.WriteLine($"Last verWeight={versOut[^1]}, Last horWeight={horsOut[^1]}");

            // 直接验证正确性
            Assert.That(versOut, Is.EqualTo(versOutExpect));
            Assert.That(horsOut, Is.EqualTo(horsOutExpect));
        }
    }
}