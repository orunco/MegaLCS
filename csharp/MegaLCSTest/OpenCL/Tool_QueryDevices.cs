namespace MegaLCSTest.OpenCL;

using Silk.NET.OpenCL;
using System;

class Tool_QueryDevices{
    
    public static unsafe void QueryAll(){
        // 初始化OpenCL
        var cl = CL.GetApi();

        // 获取平台数量
        uint platformCount = 0;
        cl.GetPlatformIDs(0, null, &platformCount);

        if (platformCount == 0){
            Console.WriteLine("未找到任何OpenCL平台。");
            return;
        }

        // 获取平台ID
        Span<nint> platformIds = new nint[(int)platformCount];
        cl.GetPlatformIDs(platformCount, platformIds, Span<uint>.Empty);

        // 遍历每个平台
        for (var i = 0; i < platformCount; i++){
            var platformId = platformIds[i];

            // 获取设备数量
            uint deviceCount = 0;
            cl.GetDeviceIDs(platformId, DeviceType.All, 0, null, &deviceCount);

            if (deviceCount == 0){
                Console.WriteLine($"平台 {i} 上没有找到任何设备。");
                continue;
            }

            // 获取设备ID
            Span<nint> deviceIds = new nint[(int)deviceCount];
            cl.GetDeviceIDs(platformId, DeviceType.All, deviceCount, deviceIds, Span<uint>.Empty);

            Console.WriteLine($"平台 {i} 上有 {deviceCount} 个设备。");

            // 遍历每个设备
            for (var j = 0; j < deviceCount; j++){
                var deviceId = deviceIds[j];

                // 获取设备名称
                Span<byte> deviceName = new byte[1024];
                cl.GetDeviceInfo(deviceId, DeviceInfo.Name, (nuint)deviceName.Length, deviceName, Span<UIntPtr>.Empty);

                // 找到第一个空字符的位置
                int length = 0;
                while (length < deviceName.Length && deviceName[length] != 0){
                    length++;
                }

                // 截取有效部分并转换为字符串
                string deviceNameString = System.Text.Encoding.UTF8.GetString(deviceName.Slice(0, length));
                Console.WriteLine($"  设备 {j}: {deviceNameString}");
            }
        }
    }
}