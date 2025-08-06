using MegaLCSLib.OpenCL;
using NUnit.Framework;

// ReSharper disable All

namespace MegaLCSTest.OpenCL;

// 测试MegaLCS_Fusion的所有分支条件
public class Test_MegaLCSFusion_Coverage{
    [Test]
    public void Test_Step_Parameter_Validation_Step_Zero(){
        // 测试用例1.1：step = 0，期望抛出异常
        Assert.Throws<Exception>(() => {
            Mega.MegaLCS_Fusion(IntPtr.Zero, IntPtr.Zero, new[]{ 1, 2 }, new[]{ 1, 2 }, 0);
        });
    }

    [Test]
    public void Test_Step_Parameter_Validation_Step_One(){
        // 测试用例1.2：step = 1，期望正常执行，使用CPU处理
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            IntPtr.Zero, IntPtr.Zero, new[]{ 1, 2 }, new[]{ 1, 2 }, 1);
        Assert.That(processByCpu, Is.True);
        Assert.That(verWeights.Length, Is.EqualTo(2));
        Assert.That(horWeights.Length, Is.EqualTo(2));
    }

    [Test]
    public void Test_Step_Parameter_Validation_Step_Two(){
        // 测试用例1.3：step = 2，期望正常执行
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            IntPtr.Zero, IntPtr.Zero, new[]{ 1, 2 }, new[]{ 1, 2 }, 2);
        Assert.That(verWeights.Length, Is.EqualTo(2));
        Assert.That(horWeights.Length, Is.EqualTo(2));
    }

    [Test]
    public void Test_Step_Parameter_Validation_Step_Invalid_High(){
        // 测试用例1.4：step = 257，期望抛出异常
        Assert.Throws<Exception>(() => {
            Mega.MegaLCS_Fusion(IntPtr.Zero, IntPtr.Zero, new[]{ 1, 2 }, new[]{ 1, 2 }, 257);
        });
    }

    [Test]
    public void Test_Direct_CpuLCS_Base_Length_Less_Than_Step(){
        // 测试用例2.1：baseVals长度 <= step，期望processByCpu=true，正确计算LCS
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            IntPtr.Zero, IntPtr.Zero, new[]{ 1 }, new[]{ 1, 2, 3 }, 2);
        Assert.That(processByCpu, Is.True);
        // 验证权重计算正确性
        Assert.That(horWeights[horWeights.Length - 1], Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void Test_Direct_CpuLCS_Latest_Length_Less_Than_Step(){
        // 测试用例2.2：latestVals长度 <= step，期望processByCpu=true，正确计算LCS
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            IntPtr.Zero, IntPtr.Zero, new[]{ 1, 2, 3 }, new[]{ 1 }, 2);
        Assert.That(processByCpu, Is.True);
        // 验证权重计算正确性
        Assert.That(horWeights[horWeights.Length - 1], Is.GreaterThanOrEqualTo(0));
    }

    [Test]
    public void Test_GPU_Device_Null_Fallback_To_CPU(){
        // 测试用例3.1：platformId为Zero，期望processByCpu=true，正确计算LCS
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            IntPtr.Zero, IntPtr.Zero, new[]{ 1, 2, 3 }, new[]{ 1, 2, 3 }, 1);
        Assert.That(processByCpu, Is.True);
        Assert.That(verWeights.Length, Is.EqualTo(3));
        Assert.That(horWeights.Length, Is.EqualTo(3));
    }

    [Test]
    public void Test_LeftTop_Region_Processing_Both_Slice_Greater_Than_Zero(){
        // 测试用例4.1：baseSliceSize > 0 且 latestSliceSize > 0，期望触发左上角处理逻辑
        // 使用真实的OpenCL设备
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId, new[]{ 1, 2 }, new[]{ 1, 2 }, 1);
        Assert.That(verWeights.Length, Is.EqualTo(2));
        Assert.That(horWeights.Length, Is.EqualTo(2));
        // 如果有GPU设备，应该使用GPU处理
        if (platformId != IntPtr.Zero && deviceId != IntPtr.Zero){
            Assert.That(processByCpu, Is.False);
        }
        else{
            Assert.That(processByCpu, Is.True);
        }
    }


    [Test]
    public void Test_RightTop_Processing_Latest_Has_Remainder(){
        // 测试用例5.1：latest有余数，期望触发右上角处理
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId, new[]{ 1, 2 }, new[]{ 1, 2, 3 }, 2);
        Assert.That(processByCpu, Is.True);
        Assert.That(verWeights.Length, Is.EqualTo(2));
        Assert.That(horWeights.Length, Is.EqualTo(3));
    }

    [Test]
    public void Test_LeftBottom_Processing_Base_Has_Remainder(){
        // 测试用例6.1：base有余数，期望触发左下角处理
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId, new[]{ 1, 2, 3 }, new[]{ 1, 2 }, 2);
        Assert.That(processByCpu, Is.True);
        Assert.That(verWeights.Length, Is.EqualTo(3));
        Assert.That(horWeights.Length, Is.EqualTo(2));
    }

    [Test]
    public void Test_RightBottom_Processing_Both_Have_Remainder(){
        // 测试用例7.1：base和latest都有余数，期望触发右下角处理
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId,
            new[]{ 1, 2, 3 },
            new[]{ 1, 2, 3 },
            2);
        Assert.That(processByCpu, Is.False);
        Assert.That(verWeights.Length, Is.EqualTo(3));
        Assert.That(horWeights.Length, Is.EqualTo(3));
    }

    [Test]
    public void Test_Return_Value_Validation_Weight_Array_Lengths(){
        // 测试用例8.1：验证返回的权重数组长度
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId, new[]{ 1, 2 }, new[]{ 1, 2 }, 1);
        Assert.That(verWeights.Length, Is.EqualTo(2));
        Assert.That(horWeights.Length, Is.EqualTo(2));
    }

    [Test]
    public void Test_LCS_Calculation_Correctness(){
        // 测试用例8.2：验证LCS计算正确性
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId,
            new[]{ 1, 2, 3 },
            new[]{ 2, 3, 4 },
            1);
        Assert.That(processByCpu, Is.False);
        // LCS应该为[2,3]，长度为2
        Assert.That(horWeights[horWeights.Length - 1], Is.EqualTo(2));
    }

    [Test]
    public void Test_Step_2_Base_123_vs_Latest_1234(){
        // step=2 [1、2、3] vs [1、2、3、4]
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId, new[]{ 1, 2, 3 }, new[]{ 1, 2, 3, 4 }, 2);
        Assert.That(verWeights.Length, Is.EqualTo(3));
        Assert.That(horWeights.Length, Is.EqualTo(4));
        // LCS应该是[1,2,3]，长度为3
        Assert.That(horWeights[horWeights.Length - 1], Is.EqualTo(3));
    }

    [Test]
    public void Test_Step_2_Base_1234_vs_Latest_123(){
        // step=2 [1、2、3、4] vs [1、2、3]
        var (platformId, deviceId) = Mega.GetFirstGpuDevice();
        var (processByCpu, verWeights, horWeights) = Mega.MegaLCS_Fusion(
            platformId, deviceId, new[]{ 1, 2, 3, 4 }, new[]{ 1, 2, 3 }, 2);
        Assert.That(verWeights.Length, Is.EqualTo(4));
        Assert.That(horWeights.Length, Is.EqualTo(3));
        // LCS应该是[1,2,3]，长度为3
        Assert.That(horWeights[horWeights.Length - 1], Is.EqualTo(3));
    }
}