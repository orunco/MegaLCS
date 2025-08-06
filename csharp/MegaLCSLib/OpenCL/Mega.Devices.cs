using Silk.NET.OpenCL;

namespace MegaLCSLib.OpenCL;

public partial class Mega{
    
    // 遍历所有平台设备，返回必要的结果
    public static unsafe List<(
        IntPtr platformId,
        IntPtr deviceId,
        string name,
        DeviceType deviceType)> GetAllDevices(){
        var result = new List<(IntPtr platformId, IntPtr deviceId, string name, DeviceType type)>();

        // 查询设备的3个信息：
        // opencl版本
        //     每个thread的寄存器数量上限
        // 每个wrap的thread数量（调度单位）
        // 每个grid最多支持多少个block？
        // 可以自适应，不要超过上限
        // OpenCL C version 1.2 does not support the 'register' storage class specifier 

        // 获取平台数量
        uint platformCount = 0;
        cl.GetPlatformIDs(
            0,
            null,
            &platformCount);

        if (platformCount == 0){
            // Console.WriteLine("未找到任何OpenCL平台。");
            return result;
        }

        // 获取平台ID
        Span<nint> platformIds = new nint[(int)platformCount];
        cl.GetPlatformIDs(
            platformCount,
            platformIds,
            Span<uint>.Empty);

        // 遍历每个平台
        for (var p = 0; p < platformCount; p++){
            var platformId = platformIds[p];

            // 获取设备数量
            uint deviceCount = 0;
            cl.GetDeviceIDs(
                platformId,
                DeviceType.All,
                0,
                null,
                &deviceCount);

            if (deviceCount == 0){
                // Console.WriteLine($"平台 {p} 上没有找到任何设备。");
                continue;
            }

            // 获取设备ID
            Span<nint> deviceIds = new nint[(int)deviceCount];
            cl.GetDeviceIDs(
                platformId,
                DeviceType.All,
                deviceCount,
                deviceIds,
                Span<uint>.Empty);

            // Console.WriteLine($"平台 {p} 上有 {deviceCount} 个设备。");

            // 遍历每个设备
            for (var d = 0; d < deviceCount; d++){
                var deviceId = deviceIds[d];

                // 获取设备名称
                Span<byte> deviceName = new byte[1024];
                cl.GetDeviceInfo(
                    deviceId,
                    DeviceInfo.Name,
                    (nuint)deviceName.Length,
                    deviceName,
                    Span<UIntPtr>.Empty);

                // 找到第一个空字符的位置
                var length = 0;
                while (length < deviceName.Length && deviceName[length] != 0){
                    length++;
                }

                // 截取有效部分并转换为字符串
                var deviceNameString = System.Text.Encoding.UTF8.GetString(
                    deviceName.Slice(0, length));
                // Console.WriteLine($"  设备 {d}: {deviceNameString}");

                // 获取设备类型
                DeviceType deviceType = 0;
                cl.GetDeviceInfo(
                    deviceId,
                    DeviceInfo.Type,
                    (nuint)sizeof(DeviceType),
                    &deviceType,
                    Span<UIntPtr>.Empty);

                result.Add((platformId, deviceId, deviceNameString, (DeviceType)deviceType));
            }
        }

        return result;
    }
    
    public static (IntPtr platformId, IntPtr deviceId) GetFirstGpuDevice(){
        // 获取第一个GPU设备，当然如果CPU够强，也可以
        var allDevices = Mega.GetAllDevices();
        var platformId = IntPtr.Zero;
        var deviceId = IntPtr.Zero;
        foreach (var device in allDevices){
            if (device.deviceType == DeviceType.Gpu){
                platformId = device.platformId;
                deviceId = device.deviceId;
                break;
            }
        }

        return (platformId, deviceId);
    }
}