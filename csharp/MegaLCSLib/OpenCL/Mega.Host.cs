/*
Copyright (C) 2025 Pete Zhang, rivxer@gmail.com, https://github.com/orunco

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// https://github.com/dotnet/Silk.NET/blob/main/examples/CSharp/OpenCL%20Demos/HelloWorld/Program.cs

using Silk.NET.OpenCL;

namespace MegaLCSLib.OpenCL;

public partial class Mega{
    // 使用 Lazy<T> 进行延迟初始化
    private static readonly Lazy<CL> lazyCL = new Lazy<CL>(() => {
        try{
            return CL.GetApi();
        }
        catch (Exception e){
            Console.WriteLine(e);
            throw;
        }
    });

    private static CL cl => lazyCL.Value;

    // 运行HostLCS, 输入的Array必须是STEP的倍数, HostLCS调用KernelLCS
    public static unsafe void HostLCS_WaveFront(
        IntPtr platformId,
        IntPtr deviceId,
        int[] baseVals,
        int[] latestVals,
        int[] verWeights,
        int[] horWeights,
        bool isSharedVersion,
        int step,
        bool isDebug = false){
        var _baseSliceSize = Valid(baseVals, isSharedVersion, step);
        var _latestSliceSize = Valid(latestVals, isSharedVersion, step);

        // 防止参数配置错误，导致大量的task，假设为1048576长度，按照1024切割，对角线为1024，也就是最大1024个task
        // if (_baseChunkCount + _latestChunkCount > 2048){
        //     throw new Exception("参数配置错误，导致task数量过大，请检查代码");
        // }

        nint context = 0;
        nint commandQueue = 0;
        nint program = 0;
        nint kernel = 0;
        nint device = 0;

        var deviceMemObjects = new nint[4];

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
            return;
        }

        // Create a command-queue on the first device available
        // on the created context
        commandQueue = CreateCommandQueue(
            cl,
            context,
            ref device);
        if (commandQueue == IntPtr.Zero){
            Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        // Create OpenCL program from HelloWorld.cl kernel source
        program = CreateProgram(
            cl,
            context,
            device,
            isSharedVersion,
            step,
            isDebug);
        if (program == IntPtr.Zero){
            Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        // Create OpenCL kernel
        kernel = cl.CreateKernel(
            program,
            "KernelLCS_MinMax",
            null);
        if (kernel == IntPtr.Zero){
            Console.WriteLine("Failed to create kernel");
            Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        // Create memory objects that will be used as arguments to
        // kernel.  First create host memory arrays that will be
        // used to store the arguments to the kernel

        if (!CreateMemObjects(
                cl,
                context,
                deviceMemObjects,
                commandQueue,
                baseVals,
                latestVals,
                verWeights,
                horWeights
            )){
            Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        // Set the kernel arguments (gBases,gLatests,gVerWeights,gHorWeights)
        var ret = cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint),
            deviceMemObjects[0]);
        ret |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint),
            deviceMemObjects[1]);
        ret |= cl.SetKernelArg(kernel, 2, (nuint)sizeof(nint),
            deviceMemObjects[2]);
        ret |= cl.SetKernelArg(kernel, 3, (nuint)sizeof(nint),
            deviceMemObjects[3]);

        // 有多少slice是确定的
        var baseSliceSize = _baseSliceSize;
        var latestSliceSize = _latestSliceSize;
        ret |= cl.SetKernelArg(kernel, 4, (nuint)sizeof(int), &baseSliceSize);
        ret |= cl.SetKernelArg(kernel, 5, (nuint)sizeof(int), &latestSliceSize);

        if (ret != (int)ErrorCodes.Success){
            Console.WriteLine("Error setting kernel arguments.");
            Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
            return;
        }

        // Queue the kernel up for execution across the array

        var totalWave = _baseSliceSize + _latestSliceSize - 1;
        // wavefront算法类似波，沿着对角带的方向前进，这里为什么命名为Band? 如果STEP>=2,则一次W覆盖了宽度为2
        // 的条带，只有核函数内部才是对角线
        for (var outerWaveFrontBand = 0;
             outerWaveFrontBand < totalWave;
             outerWaveFrontBand++){
            // if (100 * waveFrontID / totalWF % 10 == 0){
            //     Console.WriteLine($"{waveFrontID}/{totalWF}");
            // }

            // 首先：共享内存版本STEP个thread每Block，block内元素处理和线程一一对应
            var threadPerBlock = step;
            var localWorkSize_ThreadPerBlock = new nuint[]{ (nuint)threadPerBlock };

            // latest是X轴/水平方向，sliceID最小值: 逆方向，随着wavefront的逐渐减少，有可能小于0; 且同一波前处理的切片满足 baseSliceID + latestSliceID = waveFrontID
            var latestSliceIDMin = Math.Max(0, outerWaveFrontBand - (_baseSliceSize - 1));

            // latest方向的sliceID的最大值：随着wavefrontID逐渐增加，有可能超过LATEST_SLICE_SIZE，所以取小值
            var latestSliceIDMax = Math.Min(outerWaveFrontBand, _latestSliceSize - 1);

            // 其次：对于当前的wavefront，总共有多少个block? 也就是对角线的小方块数量
            // 这个算法是推导出来的，不需要用if翻越中线的方法，非常巧妙
            var totalBlockInWaveFront = Math.Max(0, latestSliceIDMax - latestSliceIDMin + 1);

            // globalWorkSize 决定了内核函数会被执行多少次。每个工作项会独立执行内核函数，并且可以通过内置函数（如 get_global_id）获取自己在全局执行空间中的唯一标识符，从而访问不同的数据。
            // 【必须整除】
            var totalThread = totalBlockInWaveFront * threadPerBlock;
            var globalWorkSize_AllThreadInOneGrid = new nuint[]{ (nuint)(totalThread) };

            if (isDebug){
                Console.WriteLine(
                    $"\n【Start new kernel】\nouterW={outerWaveFrontBand} blocks={totalBlockInWaveFront}■              totalThread={totalThread} latestSliceID={latestSliceIDMin}->{latestSliceIDMax} step={step} (in host)");
            }


            // 每一次参数是有差异的 
            ret = cl.SetKernelArg(kernel, 6, (nuint)sizeof(int), &outerWaveFrontBand);
            ret |= cl.SetKernelArg(kernel, 7, (nuint)sizeof(int), &totalThread);

            if (ret != (int)ErrorCodes.Success){
                Console.WriteLine("Error setting kernel arguments.");
                Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }

            // 如果是主显卡，sleep 10ms每次 降低CPU? 否则CPU挂死的状态
            ret = cl.EnqueueNdrangeKernel(
                commandQueue,
                kernel,
                1,
                (nuint*)null,
                globalWorkSize_AllThreadInOneGrid,
                localWorkSize_ThreadPerBlock,
                0,
                (nint*)null,
                (nint*)null);
            if (ret != (int)ErrorCodes.Success){
                Console.WriteLine("Error queuing kernel for execution.");
                Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }

            ret = cl.Finish(commandQueue);
            if (ret != (int)ErrorCodes.Success){
                Console.WriteLine("Error queuing kernel for execution Finish.");
                Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }

            if (isDebug){
                var newVerWeights = new int[baseVals.Length];
                fixed (void* pVerWeights = newVerWeights){
                    // Read the output buffer back to the Host
                    ret = cl.EnqueueReadBuffer(
                        commandQueue,
                        deviceMemObjects[2],
                        true,
                        0,
                        (uint)baseVals.Length * sizeof(int),
                        pVerWeights,
                        0,
                        null,
                        null);
                    if (ret != (int)ErrorCodes.Success){
                        Console.WriteLine("Error reading result buffer.");
                        Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                        return;
                    }
                }

                var newHorWeights = new int[latestVals.Length];
                fixed (void* pHorWeights = newHorWeights){
                    // Read the output buffer back to the Host
                    ret = cl.EnqueueReadBuffer(
                        commandQueue,
                        deviceMemObjects[3],
                        true,
                        0,
                        (uint)latestVals.Length * sizeof(int),
                        pHorWeights,
                        0,
                        null,
                        null);
                    if (ret != (int)ErrorCodes.Success){
                        Console.WriteLine("Error reading result buffer.");
                        Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                        return;
                    }
                }

                Console.WriteLine($"vers={string.Join(",", newVerWeights)}");
                Console.WriteLine($"hors={string.Join(",", newHorWeights)}");
            } // end of if (isDebug)
        } // end of for 

        fixed (void* pVerWeights = verWeights){
            // Read the output buffer back to the Host
            ret = cl.EnqueueReadBuffer(
                commandQueue,
                deviceMemObjects[2],
                true,
                0,
                (uint)baseVals.Length * sizeof(int),
                pVerWeights,
                0,
                null,
                null);
            if (ret != (int)ErrorCodes.Success){
                Console.WriteLine("Error reading result buffer.");
                Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }
        }

        fixed (void* pHorWeights = horWeights){
            // Read the output buffer back to the Host
            ret = cl.EnqueueReadBuffer(
                commandQueue,
                deviceMemObjects[3],
                true,
                0,
                (uint)latestVals.Length * sizeof(int),
                pHorWeights,
                0,
                null,
                null);
            if (ret != (int)ErrorCodes.Success){
                Console.WriteLine("Error reading result buffer.");
                Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
                return;
            }
        }

        Cleanup(cl, context, commandQueue, program, kernel, deviceMemObjects);
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
        nint commandQueue,
        int[] bases,
        int[] latests,
        int[] verWeights,
        int[] horWeights){
        var INT_BASE_AXIS_BYTES = (uint)bases.Length * sizeof(int);
        var INT_LATEST_AXIS_BYTES = (uint)latests.Length * sizeof(int);

        fixed (void* pBases = bases){
            memObjects[0] = cl.CreateBuffer(
                context,
                MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                INT_BASE_AXIS_BYTES,
                pBases,
                null);
        }

        fixed (void* pLatests = latests){
            memObjects[1] = cl.CreateBuffer(
                context,
                MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                INT_LATEST_AXIS_BYTES,
                pLatests,
                null);
        }

        memObjects[2] = cl.CreateBuffer(
            context,
            MemFlags.ReadWrite,
            INT_BASE_AXIS_BYTES,
            null,
            null);

        memObjects[3] = cl.CreateBuffer(
            context,
            MemFlags.ReadWrite,
            INT_LATEST_AXIS_BYTES,
            null,
            null);

        // 读写对象必须主动写代码，将数据从主机端传输到设备端内存里面
        fixed (int* pVerWeights = verWeights){
            cl.EnqueueWriteBuffer(
                commandQueue,
                memObjects[2],
                true,
                0,
                INT_BASE_AXIS_BYTES,
                pVerWeights,
                0,
                Span<IntPtr>.Empty,
                Span<IntPtr>.Empty);
        }

        fixed (int* pHorWeights = horWeights){
            cl.EnqueueWriteBuffer(
                commandQueue,
                memObjects[3],
                true,
                0,
                INT_LATEST_AXIS_BYTES,
                pHorWeights,
                0,
                Span<IntPtr>.Empty,
                Span<IntPtr>.Empty);
        }

        if (memObjects[0] == IntPtr.Zero ||
            memObjects[1] == IntPtr.Zero ||
            memObjects[2] == IntPtr.Zero ||
            memObjects[3] == IntPtr.Zero){
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
    static unsafe nint CreateProgram(
        CL cl,
        nint context,
        nint device,
        bool IsSharedVersion,
        int _step,
        bool isDebug){
        var code = IsSharedVersion
            ? Mega.KernelLCS_Shared
            : Mega.KernelLCS_Register;

        var program = cl.CreateProgramWithSource(
            context,
            1,
            new string[]{
                code.Replace(
                    "__STEP__",
                    _step.ToString())
            },
            null,
            null);
        if (program == IntPtr.Zero){
            Console.WriteLine("Failed to create CL program from source.");
            return IntPtr.Zero;
        }

        // 编译选项，设置优化级别为 3
        // string compileOptions = "-cl-opt-level 3";

        // 定义编译开关
        var compileOptions = isDebug ? "-DDEBUG" : null;
        var ret = cl.BuildProgram(
            program,
            0,
            null,
            compileOptions,
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
        // Console.WriteLine($"Using device: {name}");


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
    // static unsafe nint CreateContext(CL cL){
    //     var ret = cL.GetPlatformIDs(
    //         1,
    //         out var firstPlatformId,
    //         out var numPlatforms);
    //     if (ret != (int)ErrorCodes.Success || numPlatforms <= 0){
    //         Console.WriteLine("Failed to find any OpenCL platforms.");
    //         return IntPtr.Zero;
    //     }
    //
    //     // Next, create an OpenCL context on the platform.  Attempt to
    //     // create a GPU-based context, and if that fails, try to create
    //     // a CPU-based context.
    //     var contextProperties = new nint[]{
    //         (nint)ContextProperties.Platform,
    //         firstPlatformId,
    //         0
    //     };
    //
    //     fixed (nint* p = contextProperties){
    //         var context = cL.CreateContextFromType(
    //             p,
    //             DeviceType.Gpu,
    //             null,
    //             null,
    //             out ret);
    //         if (ret == (int)ErrorCodes.Success) return context;
    //         Console.WriteLine("Could not create GPU context, trying CPU...");
    //
    //         context = cL.CreateContextFromType(
    //             p,
    //             DeviceType.Cpu,
    //             null,
    //             null,
    //             out ret);
    //
    //         if (ret == (int)ErrorCodes.Success) return context;
    //         Console.WriteLine("Failed to create an OpenCL GPU or CPU context.");
    //         return IntPtr.Zero;
    //     }
    // }
    public static int Valid(
        int[] originalValues,
        bool IsSharedVersion,
        int step){
        if (originalValues.Length == 0){
            throw new Exception("originalValues.Length is invalid.");
        }

        if (IsSharedVersion){
            // 实际测试256比较合适，再大测试用例错误
            if (!(1 <= step && step <= 256)){
                throw new Exception("step is invalid.");
            }
        }
        else{
            // 寄存器向量化最多是int16
            if (step != 2 &&
                step != 4 &&
                step != 8 &&
                step != 16){
                throw new Exception("step is invalid.");
            }
        }

        // 不允许step的值超过_originalArray的长度，没有意义
        if (originalValues.Length < step)
            throw new ArgumentException("N must be less than or equal to the length of the original array.");

        if (originalValues.Length % step != 0){
            throw new ArgumentException("originalValues.Length % step != 0");
        }

        return originalValues.Length / step;
    }
}