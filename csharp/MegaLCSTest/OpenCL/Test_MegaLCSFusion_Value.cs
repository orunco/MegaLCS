using MegaLCSLib.OpenCL;
using NUnit.Framework;

namespace MegaLCSTest.OpenCL;

public class Test_MegaLCSFusion_Value{
    // STEP = 2, base = latest = 1 2 3 都相等
    [Test]
    public static void TestCase1_EqualArrays(){
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId,
            [1, 2, 3],
            [1, 2, 3],
            2);
        Assert.That(processByCpu, Is.False);
        Assert.That(verWeights[^1], Is.EqualTo(3));
        Assert.That(horWeights[^1], Is.EqualTo(3));
    }

    // STEP = 2, base = 1 2 3 latest = 5 6 7 都不相等
    [Test]
    public static void TestCase2_DifferentArrays(){
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId,
            [1, 2, 3],
            [5, 6, 7],
            2);
        Assert.That(processByCpu, Is.False);
        Assert.That(verWeights[^1], Is.EqualTo(0));
        Assert.That(horWeights[^1], Is.EqualTo(0));
    }

    // STEP = 2, 随机数组 + 验证
    [Test]
    public static void TestCase3_RandomWithValidation(){
        // 测试足够多的次数，数组足够长且随机，值随机
        for (int j = 0; j < 10; j++){
            var rand = new Random(j);
            const int MAX = 512;
            
            var baseLen = rand.Next(4, MAX);
            var baseVals = new int[baseLen];
            for (int i = 0; i < baseLen; i++){
                baseVals[i] = rand.Next(1, MAX);
            }

            var latestLen = rand.Next(4, MAX);
            var latestVals = new int[latestLen];
            for (int i = 0; i < latestLen; i++){
                latestVals[i] = rand.Next(1, MAX);
            }

            // 使用经典LCS验证
            var classicResult = Mega.ClassicLCS_DPMatrix(baseVals, latestVals);
            Console.WriteLine($"Classic: {baseLen} vs {latestLen} => LCS Result: {classicResult}");

            // 测试目标函数
            var (platformId, deviceId) = Mega.GetFirstGpuDevice();
            var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
                platformId, deviceId,
                baseVals, latestVals,
                2);
            Assert.That(processByCpu, Is.False);
            Assert.That(verWeights[^1], Is.EqualTo(classicResult));
            Assert.That(horWeights[^1], Is.EqualTo(classicResult));
        }
    }
}