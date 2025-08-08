using MegaLCSLib.OpenCL;
using NUnit.Framework;

namespace MegaLCSTest.OpenCL;

// ReSharper disable All
// ReSharper disable UselessBinaryOperation
// ReSharper disable TooWideLocalVariableScope
public class Test_CpuLCSMinMax{
    [Test]
    public void test_ex(){
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([], [], [], []); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([5], [5], [5], []); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([5], [5], [], [5]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([5], [], [5], [5]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([], [5], [5], [5]); });

        // ==
        // Assert.Throws<Exception>(() => { Mega.CpuLCS_RollLeftTop([5], [5], [5], [6]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([5, 6], [5], [5], [5, 6]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_MinMax([5], [5, 6], [5, 6], [5]); });
    }

    [Test]
    public void test_1x1(){
        // 无基础权重的测试用例 1
        {
            int[] verWeights =[0];
            int[] horWeights =[0];
            Mega.CpuLCS_MinMax([5], [6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 0 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 0 }));
        }

        {
            int[] verWeights =[0];
            int[] horWeights =[0];
            Mega.CpuLCS_MinMax([5], [5], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 1 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 1 }));
        }

        // 有基础权重的测试用例 1
        {
            int[] verWeights =[10];
            int[] horWeights =[10];
            Mega.CpuLCS_MinMax([5], [6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 10 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 10 }));
        }

        {
            int[] verWeights =[10];
            int[] horWeights =[10];
            Mega.CpuLCS_MinMax([5], [5], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 11 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 11 }));
        }
    }

    [Test]
    public void test_2x2(){
        // 无基础权重的测试用例 1
        {
            int[] verWeights =[0, 0];
            int[] horWeights =[0, 0];
            Mega.CpuLCS_MinMax([5, 6], [5, 6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 1, 2 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 1, 2 }));
        }

        {
            int[] verWeights =[0, 0];
            int[] horWeights =[0, 0];
            Mega.CpuLCS_MinMax([5, 6], [7, 8], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 0, 0 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 0, 0 }));
        }

        // 有基础权重的测试用例 1
        {
            int[] verWeights =[10, 11];
            int[] horWeights =[10, 12];
            Mega.CpuLCS_MinMax([5, 6], [5, 6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 11, 12 }));
        }

        {
            int[] verWeights =[10, 11];
            int[] horWeights =[10, 12];
            Mega.CpuLCS_MinMax([5, 6], [7, 8], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 11, 12 }));
        }
    }

    [Test]
    public void test_2x3(){
        // 有基础权重的测试用例 1
        {
            int[] verWeights =[10, 11, 12];
            int[] horWeights =[10, 11];
            Mega.CpuLCS_MinMax([5, 6, 7], [5, 6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 11, 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 12, 12 }));
        }

        {
            int[] verWeights =[10, 11];
            int[] horWeights =[10, 11, 12];
            Mega.CpuLCS_MinMax([5, 6], [5, 6, 7], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 11, 12, 12 }));
        }
    }

    [Test]
    public void test_all(){
        char[] chars ={ 'a', 'b', 'c' };
        var combinations = new List<int[]>();

        // 生成所有可能的2字符组合（笛卡尔积），转换为int数组
        for (int i = 0; i < chars.Length; i++){
            for (int j = 0; j < chars.Length; j++){
                combinations.Add(new int[]{ chars[i], chars[j] });
            }
        }

        // 生成81个测试用例
        for (int i = 0; i < combinations.Count; i++){
            for (int j = 0; j < combinations.Count; j++){
                var testCaseIndex = i * combinations.Count + j + 1;

                var baseVals = (int[])combinations[i].Clone();
                var latestVals = (int[])combinations[j].Clone();
                var verWeights = new int[baseVals.Length];
                var horWeights = new int[latestVals.Length];

                // 保存原始数组用于经典算法
                var baseValsOriginal = (int[])combinations[i].Clone();
                var latestValsOriginal = (int[])combinations[j].Clone();

                // 打印测试用例信息
                Console.WriteLine(
                    $"Test Case {testCaseIndex}: base=[{(char)baseVals[0]}, {(char)baseVals[1]}], latest=[{(char)latestVals[0]}, {(char)latestVals[1]}]");

                Mega.CpuLCS_MinMax(baseVals, latestVals, verWeights, horWeights);
                var classic = Mega.CpuLCS_DPMatrix(baseValsOriginal, latestValsOriginal);

                Assert.That(verWeights, Is.EqualTo(classic.verWeights));
                Assert.That(horWeights, Is.EqualTo(classic.horWeights));
            }
        }
    }
}