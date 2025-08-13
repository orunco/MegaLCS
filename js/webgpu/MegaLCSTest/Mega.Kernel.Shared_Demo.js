async function KernelLCS_MinMax_ForTest(
    baseVals,
    latestVals,
    verWeights,
    horWeights) {

    // 参数检查
    if (!Array.isArray(baseVals) ||
        !Array.isArray(latestVals) ||
        !Array.isArray(verWeights) ||
        !Array.isArray(horWeights)) {
        throw new Error("All parameters must be arrays.");
    }

    const baseLen = baseVals.length;
    const latestLen = latestVals.length;

    if (verWeights.length !== baseLen ||
        horWeights.length !== latestLen) {
        throw new Error("Weight arrays must match base and latest lengths.");
    }

    // 打印输入参数（前10个元素）
    console.log("Input:");
    console.log("  baseVals:", baseVals.slice(0, 10),
        baseLen > 10 ? `...(${baseLen} total)` : '');
    console.log("  latestVals:", latestVals.slice(0, 10),
        latestLen > 10 ? `...(${latestLen} total)` : '');
    console.log("  verWeights (before):", verWeights.slice(0, 10),
        baseLen > 10 ? `...(${baseLen} total)` : '');
    console.log("  horWeights (before):", horWeights.slice(0, 10),
        latestLen > 10 ? `...(${latestLen} total)` : '');

    // 使用最小的STEP进行测试
    const STEP = 1; // 每一个工作组内开启多少线程
    const workgroupCount = Math.ceil(Math.max(baseLen, latestLen) / STEP);
    const THREAD_LOG_SIZE = 100 * 1024;
    console.log(`STEP ${STEP}, Workgroup count: ${workgroupCount}, THREAD_LOG_SIZE ${THREAD_LOG_SIZE}`);

    // 初始化 WebGPU
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found");
    }

    const device = await adapter.requestDevice();

    // 生成WGSL代码
    const KernelLCS_MinMax_WGSL = generateWGSLKernel(
        {
            step: STEP,
            threadLogSize: THREAD_LOG_SIZE,
            debug: true
        });

    // 创建管线
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: KernelLCS_MinMax_WGSL,
            }),
            entryPoint: 'KernelLCS_MinMax'
        },
    });

    // 缓冲区大小
    const bufferSizeBase = baseLen * 4;
    const bufferSizeLatest = latestLen * 4;
    // 计算日志缓冲区大小：
    // 总大小 = 工作组数量 * 2 * __THREAD_LOG_SIZE__ * 4（每个u32占4字节）
    const bufferSizeLog = workgroupCount * 2 * THREAD_LOG_SIZE * 4;
    // 参数缓冲区大小：4个i32，每个4字节
    const bufferSizeParams = 4 * 4;

    // 创建 GPU 缓冲区
    const gBases = device.createBuffer({
        label: 'gBases',
        size: bufferSizeBase,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const gLatests = device.createBuffer({
        label: 'gLatests',
        size: bufferSizeLatest,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const gVerWeights = device.createBuffer({
        label: 'gVerWeights',
        size: bufferSizeBase,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const gHorWeights = device.createBuffer({
        label: 'gHorWeights',
        size: bufferSizeLatest,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // 初始化日志缓冲区
    const logBuffer = device.createBuffer({
        size: bufferSizeLog,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // 创建参数缓冲区（uniform buffer）
    const paramsBuffer = device.createBuffer({
        size: bufferSizeParams,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 创建绑定组
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: gBases}},
            {binding: 1, resource: {buffer: gLatests}},
            {binding: 2, resource: {buffer: gVerWeights}},
            {binding: 3, resource: {buffer: gHorWeights}},
            {binding: 4, resource: {buffer: paramsBuffer}}, // uniform buffer
            {binding: 5, resource: {buffer: logBuffer}},    // log buffer
        ],
    });

    // 写入输入数据
    device.queue.writeBuffer(gBases, 0, new Int32Array(baseVals));
    device.queue.writeBuffer(gLatests, 0, new Int32Array(latestVals));
    device.queue.writeBuffer(gVerWeights, 0, new Int32Array(verWeights));
    device.queue.writeBuffer(gHorWeights, 0, new Int32Array(horWeights));

    // 写入参数数据到uniform buffer
    const paramsData = new Int32Array([
        baseLen,                    // baseSliceSize
        latestLen,                  // latestSliceSize
        0,    // outerW
        STEP * workgroupCount       // totalThread
    ]);
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    // 创建命令编码器
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount); // 启动足够的工作组

    pass.end();

    // 提交命令
    device.queue.submit([encoder.finish()]);

    // 等待 GPU 完成执行
    await device.queue.onSubmittedWorkDone();

    // 异步读取结果
    const verReadBuffer = device.createBuffer({
        size: bufferSizeBase,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const horReadBuffer = device.createBuffer({
        size: bufferSizeLatest,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // 创建日志读取缓冲区
    const logReadBuffer = device.createBuffer({
        size: bufferSizeLog,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder2 = device.createCommandEncoder();
    encoder2.copyBufferToBuffer(gVerWeights, 0, verReadBuffer, 0, bufferSizeBase);
    encoder2.copyBufferToBuffer(gHorWeights, 0, horReadBuffer, 0, bufferSizeLatest);
    // 拷贝日志缓冲区内容到可读取缓冲区
    encoder2.copyBufferToBuffer(logBuffer, 0, logReadBuffer, 0, bufferSizeLog);
    device.queue.submit([encoder2.finish()]);

    // 映射并读取数据
    await verReadBuffer.mapAsync(GPUMapMode.READ);
    const verArrayBuffer = verReadBuffer.getMappedRange();
    const verResult = new Int32Array(verArrayBuffer.slice(0));
    verReadBuffer.unmap();

    await horReadBuffer.mapAsync(GPUMapMode.READ);
    const horArrayBuffer = horReadBuffer.getMappedRange();
    const horResult = new Int32Array(horArrayBuffer.slice(0));
    horReadBuffer.unmap();

    // 映射并读取日志数据
    await logReadBuffer.mapAsync(GPUMapMode.READ);
    const logArrayBuffer = logReadBuffer.getMappedRange();
    const logData = new Uint32Array(logArrayBuffer.slice(0));
    logReadBuffer.unmap();

    // 写回结果
    for (let i = 0; i < baseLen; i++) {
        verWeights[i] = verResult[i];
    }

    for (let i = 0; i < latestLen; i++) {
        horWeights[i] = horResult[i];
    }

    // 解析日志数据（支持线程隔离）
    function decodeLog(bufferData, workgroupCount, THREAD_LOG_SIZE) {
        let result = "";

        for (let wgIdx = 0; wgIdx < workgroupCount; wgIdx++) {
            for (let threadId = 0; threadId < 2; threadId++) {
                const baseOffset = wgIdx * 2 * THREAD_LOG_SIZE + threadId * THREAD_LOG_SIZE;
                let threadLog = "";

                for (let i = 0; i < THREAD_LOG_SIZE; i++) {
                    const charCode = bufferData[baseOffset + i];
                    if (charCode === 0) break; // 遇到结束符
                    threadLog += String.fromCharCode(charCode);
                }

                if (threadLog.length > 0) {
                    result += `workgroup_idx ${wgIdx} thread_idx ${threadId}: \n${threadLog}\n`;
                }
            }
        }

        return result;
    }

    console.log("GPU Log:", decodeLog(logData, workgroupCount, THREAD_LOG_SIZE));

    // 打印输出结果（前10个元素）
    console.log("Output:");
    console.log("  verWeights (after):", verWeights.slice(0, 10), baseLen > 10 ? `...(${baseLen} total)` : '');
    console.log("  horWeights (after):", horWeights.slice(0, 10), latestLen > 10 ? `...(${latestLen} total)` : '');

    // 清理资源
    gBases.destroy();
    gLatests.destroy();
    gVerWeights.destroy();
    gHorWeights.destroy();
    verReadBuffer.destroy();
    horReadBuffer.destroy();
    logReadBuffer.destroy();
    paramsBuffer.destroy();
}
