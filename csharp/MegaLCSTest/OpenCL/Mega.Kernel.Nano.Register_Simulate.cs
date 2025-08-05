using NUnit.Framework;

// ReSharper disable All

namespace MegaLCSTest.OpenCL;

// ReSharper disable UselessBinaryOperation
// ReSharper disable TooWideLocalVariableScope
// 内核函数调试非常困难，用这个代码先做一下仿真验证
public class Mega_Kernel_Nano_Register_Simulate{
    // bases可以看成是展开到Y轴; latest可以看成是展开到X轴;
    // hors存储原DP的横向权重，类似滚动数组；【是输入也是输出】
    // vers存储原DP的纵向权重，【是输入也是输出】
    // 和普通的LCS基础权重0略有差别，这个函数的hors和vers是有基础权重的,不一定为0
    private void NanoLCS_Register(
        int[] bases, int[] latests,
        int[] hors, int[] vers){
        // 先做校验，这个是由理论分析后的结果，必须满足
        if (bases.Length == 0 ||
            latests.Length == 0 ||
            hors.Length == 0 ||
            vers.Length == 0 ||
            bases.Length != vers.Length ||
            latests.Length != hors.Length ||
            hors[0] != vers[0]){
            throw new Exception();
        }

        var leftTop = 0; // 模拟DP左上角初始值
        var horLBackup = 0; // 备份当前值，供下一轮使用

        for (var b = 0; b < bases.Length; b++){
            // 每一行的初始权重不是0，而是hors里面的第0个元素开始的,和参数b无关
            horLBackup = hors[0];

            for (var l = 0; l < latests.Length; l++){
                leftTop = horLBackup;
                horLBackup = hors[l];

                // 高度优化：不相等的先命中
                if (bases[b] != latests[l]){
                    // 高度优化：不相等的先命中
                    hors[l] = l != 0
                        ? Math.Max(hors[l], hors[l - 1])
                        : Math.Max(hors[l], vers[b]);
                }
                else{
                    hors[l] = leftTop + 1;
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
            vers[b] = hors[hors.Length - 1];
        }

        Console.WriteLine("hors: " + string.Join(", ", hors));
        Console.WriteLine("vers: " + string.Join(", ", vers));
    }

    [Test]
    public void test_ex(){
        Assert.Throws<Exception>(() => { NanoLCS_Register([], [], [], []); });
        Assert.Throws<Exception>(() => { NanoLCS_Register([5], [5], [5], []); });
        Assert.Throws<Exception>(() => { NanoLCS_Register([5], [5], [], [5]); });
        Assert.Throws<Exception>(() => { NanoLCS_Register([5], [], [5], [5]); });
        Assert.Throws<Exception>(() => { NanoLCS_Register([], [5], [5], [5]); });

        // ==
        Assert.Throws<Exception>(() => { NanoLCS_Register([5], [5], [5], [6]); });
        Assert.Throws<Exception>(() => { NanoLCS_Register([5, 6], [5], [5], [5]); });
        Assert.Throws<Exception>(() => { NanoLCS_Register([5], [5, 6], [5], [5]); });
    }

    [Test]
    public void test_1x1(){
        // 无基础权重的测试用例 1
        {
            int[] hors =[0];
            int[] vers =[0];
            NanoLCS_Register([5], [6], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 0 }));
            Assert.That(vers, Is.EqualTo(new[]{ 0 }));
        }

        {
            int[] hors =[0];
            int[] vers =[0];
            NanoLCS_Register([5], [5], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 1 }));
            Assert.That(vers, Is.EqualTo(new[]{ 1 }));
        }

        // 有基础权重的测试用例 1
        {
            int[] hors =[10];
            int[] vers =[10];
            NanoLCS_Register([5], [6], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 10 }));
            Assert.That(vers, Is.EqualTo(new[]{ 10 }));
        }

        {
            int[] hors =[10];
            int[] vers =[10];
            NanoLCS_Register([5], [5], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 11 }));
            Assert.That(vers, Is.EqualTo(new[]{ 11 }));
        }
    }

    [Test]
    public void test_2x2(){
        // 无基础权重的测试用例 1
        {
            int[] hors =[0, 0];
            int[] vers =[0, 0];
            NanoLCS_Register([5, 6], [5, 6], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 1, 2 }));
            Assert.That(vers, Is.EqualTo(new[]{ 1, 2 }));
        }

        {
            int[] hors =[0, 0];
            int[] vers =[0, 0];
            NanoLCS_Register([5, 6], [7, 8], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 0, 0 }));
            Assert.That(vers, Is.EqualTo(new[]{ 0, 0 }));
        }

        // 有基础权重的测试用例 1
        {
            int[] hors =[10, 11];
            int[] vers =[10, 12];
            NanoLCS_Register([5, 6], [5, 6], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(vers, Is.EqualTo(new[]{ 11, 12 }));
        }

        {
            int[] hors =[10, 11];
            int[] vers =[10, 12];
            NanoLCS_Register([5, 6], [7, 8], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(vers, Is.EqualTo(new[]{ 11, 12 }));
        }
    }

    [Test]
    public void test_2x3(){
        // 有基础权重的测试用例 1
        {
            int[] hors =[10, 11];
            int[] vers =[10, 11, 12];
            NanoLCS_Register([5, 6, 7], [5, 6], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 12, 12 }));
            Assert.That(vers, Is.EqualTo(new[]{ 11, 12, 12 }));
        }

        {
            int[] hors =[10, 11, 12];
            int[] vers =[10, 11];
            NanoLCS_Register([5, 6], [5, 6, 7], hors, vers);
            Assert.That(hors, Is.EqualTo(new[]{ 11, 12, 12 }));
            Assert.That(vers, Is.EqualTo(new[]{ 12, 12 }));
        }
    }
}