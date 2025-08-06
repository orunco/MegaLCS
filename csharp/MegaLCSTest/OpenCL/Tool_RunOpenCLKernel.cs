// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// https://github.com/dotnet/Silk.NET/blob/main/examples/CSharp/OpenCL%20Demos/HelloWorld/Program.cs

using Silk.NET.OpenCL;

namespace MegaLCSTest.OpenCL;

class Tool_RunOpenCLKernel{
    private const int N = 32;
    private const int Max = 1048576 / N;
    private const int ITERATIONS = 3;

    private const string KernelSource =
        """
        __kernel void hello_kernel(__global const float *a,__global const float *b,__global float *result)
        {
        int gid = get_global_id(0);
        result[gid] = a[gid] + b[gid];
        }
        """;

    const int ARRAY_SIZE = 1000;

    public static unsafe void run_kernel(){
        Silk.NET.OpenCL.CL cl;
        try{
            cl = CL.GetApi();
        }
        catch (Exception e){
            /*
            比如出现FileNotFoundException，是因为opencl.dll没有找到，原因是系统没有安装opencl的驱动，
            出现在qemu的windows虚拟机里面。
            qemu的windows虚拟机，即使显卡选择了virtio，也无法安装opencl。
            第三方倒是有一个：Oclgrind-21.10-Windows，下载，install后通过GPU Caps Viewer应用
            可以查询到支持opencl了，然后必须把27kb的opencl.dll复制到这里根目录下面，就不会出现文件找不到了
            接着debug代码，发现内核报错：内存访问异常；即使内核不访问内存，那么应用就hang了
            说明第三方的opencl模拟器无法有效支持。
            算了，直接用物理机吧，测试用例到时候处理一下，规避虚拟机的情况
            https://forums.opensuse.org/t/opencl-at-qemu-kvm-windows-guest/170485
             */
            Console.WriteLine(e);
            throw;
        }


        // 获取平台数量
        uint platformCount = 0;
        cl.GetPlatformIDs(
            0,
            null,
            &platformCount);

        if (platformCount == 0){
            Console.WriteLine("未找到任何OpenCL平台。");
            return;
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
                Console.WriteLine($"平台 {p} 上没有找到任何设备。");
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

            Console.WriteLine($"平台 {p} 上有 {deviceCount} 个设备。");

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
                Console.WriteLine($"  设备 {d}: {deviceNameString}");


                nint context = 0;
                nint commandQueue = 0;
                nint program = 0;
                nint kernel = 0;
                nint device = 0;

                var memObjects = new nint[3];

                // 创建上下文
                var contextProperties = new nint[]{
                    (nint)ContextProperties.Platform,
                    (nint)platformId,
                    0
                };
                fixed (nint* pProps = contextProperties){
                    context = cl.CreateContext(
                        pProps,
                        1,
                        &deviceId,
                        null,
                        null,
                        null);
                }

                if (context == IntPtr.Zero){
                    Console.WriteLine("Failed to create OpenCL context for device.");
                    continue;
                }

                // Create an OpenCL context on first available platform
                // 不再单独只选择一个opencl设备，而是遍历所有的设备
                // context = CreateContext(cl);
                // if (context == IntPtr.Zero){
                //     Console.WriteLine("Failed to create OpenCL context.");
                //     return;
                // }

                // Create a command-queue on the first device available
                // on the created context
                commandQueue = CreateCommandQueue(
                    cl,
                    context,
                    ref device);
                if (commandQueue == IntPtr.Zero){
                    Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                    return;
                }

                // Create OpenCL program from HelloWorld.cl kernel source
                program = CreateProgram(
                    cl,
                    context,
                    device,
                    "HelloWorld.cl");
                if (program == IntPtr.Zero){
                    Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                    return;
                }

                // Create OpenCL kernel
                kernel = cl.CreateKernel(
                    program,
                    "hello_kernel",
                    null);
                if (kernel == IntPtr.Zero){
                    Console.WriteLine("Failed to create kernel");
                    Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                    return;
                }

                // Create memory objects that will be used as arguments to
                // kernel.  First create host memory arrays that will be
                // used to store the arguments to the kernel
                var result = new float[ARRAY_SIZE];
                var a = new float[ARRAY_SIZE];
                var b = new float[ARRAY_SIZE];
                for (var i = 0; i < ARRAY_SIZE; i++){
                    a[i] = i;
                    b[i] = (i * 2);
                }

                if (!CreateMemObjects(cl, context, memObjects, a, b)){
                    Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                    return;
                }

                // Set the kernel arguments (result, a, b)
                var ret = cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint),
                    memObjects[0]);
                ret |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint),
                    memObjects[1]);
                ret |= cl.SetKernelArg(kernel, 2, (nuint)sizeof(nint),
                    memObjects[2]);

                if (ret != (int)ErrorCodes.Success){
                    Console.WriteLine("Error setting kernel arguments.");
                    Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                    return;
                }


                // Queue the kernel up for execution across the array
                var stopwatch = System.Diagnostics.Stopwatch.StartNew(); // 开始计时


                for (var k = 1; k <= Max; k++){
                    var globalWorkSize = new nuint[]{ (nuint)k * N };
                    var localWorkSize = new nuint[]{ N };

                    ret = cl.EnqueueNdrangeKernel(
                        commandQueue,
                        kernel,
                        1,
                        (nuint*)null,
                        globalWorkSize,
                        localWorkSize,
                        0,
                        (nint*)null,
                        (nint*)null);
                    if (ret != (int)ErrorCodes.Success){
                        Console.WriteLine("Error queuing kernel for execution.");
                        Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                        return;
                    }

                    ret = cl.Finish(commandQueue);
                    if (ret != (int)ErrorCodes.Success){
                        Console.WriteLine("Error queuing kernel for execution Finish.");
                        Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                        return;
                    }
                }

                fixed (void* pValue = result){
                    // Read the output buffer back to the Host
                    ret = cl.EnqueueReadBuffer(
                        commandQueue,
                        memObjects[2],
                        true,
                        0,
                        ARRAY_SIZE * sizeof(float),
                        pValue,
                        0,
                        null,
                        null);
                    if (ret != (int)ErrorCodes.Success){
                        Console.WriteLine("Error reading result buffer.");
                        Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                        return;
                    }
                }

                // Output the result buffer
                // for (int i = 0; i < ARRAY_SIZE; i++){
                //     Console.WriteLine(result[i]);
                // }
                stopwatch.Stop(); // 结束计时
                Console.WriteLine($"Execution time: {stopwatch.ElapsedMilliseconds} ms");

                Console.WriteLine($"Last value = {result[ARRAY_SIZE - 1]}");
                Console.WriteLine("Executed program succesfully.");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
            }
        }
    }

    /// <summary>
    /// Create memory objects used as the arguments to the kernel
    /// The kernel takes three arguments: result (output), a (input),
    /// and b (input)
    /// </summary>
    /// <param name="context"></param>
    /// <param name="memObjects"></param>
    /// <param name="cl"></param>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    static unsafe bool CreateMemObjects(
        CL cl,
        nint context,
        nint[] memObjects,
        float[] a,
        float[] b){
        fixed (void* pa = a){
            memObjects[0] = cl.CreateBuffer(
                context,
                MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                sizeof(float) * ARRAY_SIZE,
                pa,
                null);
        }

        fixed (void* pb = b){
            memObjects[1] = cl.CreateBuffer(
                context,
                MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                sizeof(float) * ARRAY_SIZE,
                pb,
                null);
        }

        memObjects[2] = cl.CreateBuffer(
            context,
            MemFlags.ReadWrite,
            sizeof(float) * ARRAY_SIZE,
            null,
            null);

        if (memObjects[0] == IntPtr.Zero ||
            memObjects[1] == IntPtr.Zero ||
            memObjects[2] == IntPtr.Zero){
            Console.WriteLine("Error creating memory objects.");
            return false;
        }

        return true;
    }

    /// <summary>
    /// Create an OpenCL program from the kernel source file
    /// </summary>
    /// <param name="cl"></param>
    /// <param name="context"></param>
    /// <param name="device"></param>
    /// <param name="fileName"></param>
    /// <returns></returns>
    static unsafe nint CreateProgram(CL cl, nint context, nint device, string fileName){
        // if (!File.Exists(fileName))
        // {
        //     Console.WriteLine($"File does not exist: {fileName}");
        //     return IntPtr.Zero;
        // }
        // using StreamReader sr = new StreamReader(fileName);
        // string clStr = sr.ReadToEnd();


        var program = cl.CreateProgramWithSource(
            context,
            1,
            new string[]{ KernelSource },
            null,
            null);
        if (program == IntPtr.Zero){
            Console.WriteLine("Failed to create CL program from source.");
            return IntPtr.Zero;
        }

        var ret = cl.BuildProgram(
            program,
            0,
            null,
            (byte*)null,
            null,
            null);

        if (ret != (int)ErrorCodes.Success){
            _ = cl.GetProgramBuildInfo(
                program,
                device,
                ProgramBuildInfo.BuildLog,
                0,
                null,
                out nuint buildLogSize);
            byte[] log = new byte[buildLogSize / (nuint)sizeof(byte)];
            fixed (void* pValue = log){
                cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
            }

            string? build_log = System.Text.Encoding.UTF8.GetString(log);

            //Console.WriteLine("Error in kernel: ");
            Console.WriteLine("=============== OpenCL Program Build Info ================");
            Console.WriteLine(build_log);
            Console.WriteLine("==========================================================");

            cl.ReleaseProgram(program);
            return IntPtr.Zero;
        }

        return program;
    }

    /// <summary>
    /// Cleanup any created OpenCL resources
    /// </summary>
    /// <param name="cl"></param>
    /// <param name="context"></param>
    /// <param name="commandQueue"></param>
    /// <param name="program"></param>
    /// <param name="kernel"></param>
    /// <param name="memObjects"></param>
    static void Cleanup(
        CL cl,
        nint context,
        nint commandQueue,
        nint program,
        nint kernel,
        nint[] memObjects){
        foreach (var t in memObjects){
            if (t != 0)
                cl.ReleaseMemObject(t);
        }

        if (commandQueue != 0)
            cl.ReleaseCommandQueue(commandQueue);

        if (kernel != 0)
            cl.ReleaseKernel(kernel);

        if (program != 0)
            cl.ReleaseProgram(program);

        if (context != 0)
            cl.ReleaseContext(context);
    }

    /// <summary>
    /// Create a command queue on the first device available on the
    /// context
    /// </summary>
    /// <param name="cL"></param>
    /// <param name="context"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    static unsafe nint CreateCommandQueue(CL cL, nint context, ref nint device){
        var ret = cL.GetContextInfo(
            context,
            ContextInfo.Devices,
            0,
            null,
            out nuint deviceBufferSize);
        if (ret != (int)ErrorCodes.Success){
            Console.WriteLine("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)");
            return IntPtr.Zero;
        }

        if (deviceBufferSize <= 0){
            Console.WriteLine("No devices available.");
            return IntPtr.Zero;
        }

        nint[] devices = new nint[deviceBufferSize / (nuint)sizeof(nuint)];
        fixed (void* pValue = devices){
            ret = cL.GetContextInfo(
                context,
                ContextInfo.Devices,
                deviceBufferSize,
                pValue,
                null);
        }

        if (ret != (int)ErrorCodes.Success){
            devices = null;
            Console.WriteLine("Failed to get device IDs");
            return IntPtr.Zero;
        }


        // Get the device name
        const int maxNameLength = 1024;
        byte* deviceName = stackalloc byte[maxNameLength];
        ret = cL.GetDeviceInfo(
            devices[0],
            DeviceInfo.Name,
            maxNameLength,
            deviceName,
            out _);
        if (ret != (int)ErrorCodes.Success){
            Console.WriteLine("Failed to get device name.");
            return IntPtr.Zero;
        }

        var name = System.Text.Encoding.UTF8.GetString(
            deviceName,
            maxNameLength).TrimEnd('\0');
        Console.WriteLine($"Using device: {name}");


        // In this example, we just choose the first available device.  In a
        // real program, you would likely use all available devices or choose
        // the highest performance device based on OpenCL device queries
        var commandQueue = cL.CreateCommandQueue(
            context,
            devices[0],
            CommandQueueProperties.None,
            null);
        if (commandQueue == IntPtr.Zero){
            Console.WriteLine("Failed to create commandQueue for device 0");
            return IntPtr.Zero;
        }

        device = devices[0];
        return commandQueue;
    }

    /// <summary>
    /// Create an OpenCL context on the first available platform using
    /// either a GPU or CPU depending on what is available.
    /// </summary>
    /// <param name="cL"></param>
    /// <returns></returns>
    static unsafe nint CreateContext(CL cL){
        var ret = cL.GetPlatformIDs(
            1,
            out var firstPlatformId,
            out var numPlatforms);
        if (ret != (int)ErrorCodes.Success || numPlatforms <= 0){
            Console.WriteLine("Failed to find any OpenCL platforms.");
            return IntPtr.Zero;
        }

        // Next, create an OpenCL context on the platform.  Attempt to
        // create a GPU-based context, and if that fails, try to create
        // a CPU-based context.
        var contextProperties = new nint[]{
            (nint)ContextProperties.Platform,
            firstPlatformId,
            0
        };

        fixed (nint* p = contextProperties){
            var context = cL.CreateContextFromType(
                p,
                DeviceType.Gpu,
                null,
                null,
                out ret);
            if (ret == (int)ErrorCodes.Success) return context;
            Console.WriteLine("Could not create GPU context, trying CPU...");

            context = cL.CreateContextFromType(
                p,
                DeviceType.Cpu,
                null,
                null,
                out ret);

            if (ret == (int)ErrorCodes.Success) return context;
            Console.WriteLine("Failed to create an OpenCL GPU or CPU context.");
            return IntPtr.Zero;
        }
    }
}