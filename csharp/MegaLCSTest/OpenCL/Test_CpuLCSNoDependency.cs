using MegaLCSLib.OpenCL;
using NUnit.Framework;

namespace MegaLCSTest.OpenCL;
// ReSharper disable All
// ReSharper disable UselessBinaryOperation
// ReSharper disable TooWideLocalVariableScope

public class Test_CpuLCSNoDependency{
    [Test]
    public void test_ex(){
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([], [], [], []); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([5], [5], [5], []); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([5], [5], [], [5]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([5], [], [5], [5]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([], [5], [5], [5]); });

        // ==
        // Assert.Throws<Exception>(() => { Mega.CpuLCS_RollLeftTop([5], [5], [5], [6]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([5, 6], [5], [5], [5, 6]); });
        Assert.Throws<Exception>(() => { Mega.CpuLCS_NoDependency([5], [5, 6], [5, 6], [5]); });
    }

    [Test]
    public void test_1x1(){
        // 无基础权重的测试用例 1
        {
            int[] verWeights =[0];
            int[] horWeights =[0];
            Mega.CpuLCS_NoDependency([5], [6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 0 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 0 }));
        }

        {
            int[] verWeights =[0];
            int[] horWeights =[0];
            Mega.CpuLCS_NoDependency([5], [5], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 1 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 1 }));
        }

        // 有基础权重的测试用例 1
        {
            int[] verWeights =[10];
            int[] horWeights =[10];
            Mega.CpuLCS_NoDependency([5], [6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 10 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 10 }));
        }

        {
            int[] verWeights =[10];
            int[] horWeights =[10];
            Mega.CpuLCS_NoDependency([5], [5], verWeights, horWeights);
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
            Mega.CpuLCS_NoDependency([5, 6], [5, 6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 1, 2 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 1, 2 }));
        }

        {
            int[] verWeights =[0, 0];
            int[] horWeights =[0, 0];
            Mega.CpuLCS_NoDependency([5, 6], [7, 8], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 0, 0 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 0, 0 }));
        }

        // 有基础权重的测试用例 1
        {
            int[] verWeights =[10, 11];
            int[] horWeights =[10, 12];
            Mega.CpuLCS_NoDependency([5, 6], [5, 6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 11, 12 }));
        }

        {
            int[] verWeights =[10, 11];
            int[] horWeights =[10, 12];
            Mega.CpuLCS_NoDependency([5, 6], [7, 8], verWeights, horWeights);
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
            Mega.CpuLCS_NoDependency([5, 6, 7], [5, 6], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 11, 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 12, 12 }));
        }

        {
            int[] verWeights =[10, 11];
            int[] horWeights =[10, 11, 12];
            Mega.CpuLCS_NoDependency([5, 6], [5, 6, 7], verWeights, horWeights);
            Assert.That(verWeights, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(horWeights, Is.EqualTo(new[]{ 11, 12, 12 }));
        }
    }
}