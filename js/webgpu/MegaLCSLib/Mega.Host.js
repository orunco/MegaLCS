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

// HostLCS_WaveFront 运行 HostLCS, 输入的Array必须是STEP的倍数, HostLCS调用KernelLCS
async function HostLCS_WaveFront(
    baseVals,
    latestVals,
    verWeights,
    horWeights,
    isSharedVersion,
    step,
    isDebug = false) {

    const _baseSliceSize = Valid(baseVals, isSharedVersion, step);
    const _latestSliceSize = Valid(latestVals, isSharedVersion, step);

    if (!navigator.gpu) {
        throw new Error("WebGPU not supported");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found");
    }

    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl");
    if (gl) {
        const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
        if (debugInfo) {
            const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            console.log("GPU Vendor:", vendor);
            console.log("GPU Renderer:", renderer);
        }
    }
    
    const device = await adapter.requestDevice();

    // Create kernel code
    const KernelLCS_MinMax_WGSL = generateWGSLKernel({
        step: step,
        threadLogSize: 1024 * 100,
        debug: isDebug
    });

    // Create bind group layout entries dynamically
    const bindGroupLayoutEntries = [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ];

    // Only add log buffer binding if debug is enabled
    if (isDebug) {
        bindGroupLayoutEntries.push({ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    }

    // Create bind group layout and pipeline layout
    const bindGroupLayout = device.createBindGroupLayout({
        entries: bindGroupLayoutEntries
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    // Create pipeline with explicit layout
    const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: device.createShaderModule({ code: KernelLCS_MinMax_WGSL }),
            entryPoint: 'KernelLCS_MinMax'
        },
    });

    const baseLen = baseVals.length;
    const latestLen = latestVals.length;

    const bufferSizeBase = baseLen * 4;
    const bufferSizeLatest = latestLen * 4;
    const workgroupCount = Math.ceil(Math.max(_baseSliceSize, _latestSliceSize));
    const THREAD_LOG_SIZE = 1024 * 100;
    const bufferSizeLog = workgroupCount * 2 * THREAD_LOG_SIZE * 4;
    const bufferSizeParams = 4 * 4; // baseSliceSize, latestSliceSize, outerW, totalThread

    const gBases = device.createBuffer({
        label: 'gBases',
        size: bufferSizeBase,
        usage: GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });

    const gLatests = device.createBuffer({
        label: 'gLatests',
        size: bufferSizeLatest,
        usage: GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });

    const gVerWeights = device.createBuffer({
        label: 'gVerWeights',
        size: bufferSizeBase,
        usage: GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });

    const gHorWeights = device.createBuffer({
        label: 'gHorWeights',
        size: bufferSizeLatest,
        usage: GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
    });

    // Create log buffer only if debug is enabled
    let logBuffer = null;
    if (isDebug) {
        logBuffer = device.createBuffer({
            label: 'logBuffer',
            size: bufferSizeLog,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
    }

    const paramsBuffer = device.createBuffer({
        label: 'paramsBuffer',
        size: bufferSizeParams,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create bind group entries dynamically
    const bindGroupEntries = [
        { binding: 0, resource: { buffer: gBases } },
        { binding: 1, resource: { buffer: gLatests } },
        { binding: 2, resource: { buffer: gVerWeights } },
        { binding: 3, resource: { buffer: gHorWeights } },
        { binding: 4, resource: { buffer: paramsBuffer } }
    ];

    // Only add log buffer entry if debug is enabled
    if (isDebug && logBuffer) {
        bindGroupEntries.push({ binding: 5, resource: { buffer: logBuffer } });
    }

    const bindGroup = device.createBindGroup({
        label: 'bindGroupX',
        layout: bindGroupLayout,
        entries: bindGroupEntries,
    });

    device.queue.writeBuffer(gBases, 0, new Int32Array(baseVals));
    device.queue.writeBuffer(gLatests, 0, new Int32Array(latestVals));
    device.queue.writeBuffer(gVerWeights, 0, new Int32Array(verWeights));
    device.queue.writeBuffer(gHorWeights, 0, new Int32Array(horWeights));

    const totalWave = _baseSliceSize + _latestSliceSize - 1;

    // wavefront算法类似波，沿着对角带的方向前进，这里为什么命名为Band? 如果STEP>=2,则一次W覆盖了宽度为2
    // 的条带，只有核函数内部才是对角线
    for (let outerWaveFrontBand = 0;
         outerWaveFrontBand < totalWave;
         outerWaveFrontBand++) {
        // latest是X轴/水平方向，sliceID最小值: 逆方向，随着wavefront的逐渐减少，有可能小于0; 且同一波前处理的切片满足 baseSliceID + latestSliceID = waveFrontID
        const latestSliceIDMin = Math.max(0, outerWaveFrontBand - (_baseSliceSize - 1));
        // latest方向的sliceID的最大值：随着wavefrontID逐渐增加，有可能超过LATEST_SLICE_SIZE，所以取小值
        const latestSliceIDMax = Math.min(outerWaveFrontBand, _latestSliceSize - 1);
        // 其次：对于当前的wavefront，总共有多少个block? 也就是对角线的小方块数量
        // 这个算法是推导出来的，不需要用if翻越中线的方法，非常巧妙
        const totalBlockInWaveFront = Math.max(0, latestSliceIDMax - latestSliceIDMin + 1);
        // globalWorkSize 决定了内核函数会被执行多少次。每个工作项会独立执行内核函数，并且可以通过内置函数（如 get_global_id）获取自己在全局执行空间中的唯一标识符，从而访问不同的数据。
        // 【必须整除】
        const totalThread = totalBlockInWaveFront * step;

        if (isDebug) {
            console.log(`\n【Start new kernel】\nouterW=${outerWaveFrontBand} blocks=${totalBlockInWaveFront}■              totalThread=${totalThread} latestSliceID=${latestSliceIDMin}->${latestSliceIDMax} step=${step} (in host)`);
        }

        const paramsData = new Int32Array([
            _baseSliceSize,
            _latestSliceSize,
            outerWaveFrontBand,
            totalThread
        ]);
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();

        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(totalBlockInWaveFront, 1, 1); // 启动足够的工作组

        pass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        if (isDebug) {
            const verReadBuffer = device.createBuffer({
                size: bufferSizeBase,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            const horReadBuffer = device.createBuffer({
                size: bufferSizeLatest,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            const encoder2 = device.createCommandEncoder();
            encoder2.copyBufferToBuffer(gVerWeights, 0, verReadBuffer, 0, bufferSizeBase);
            encoder2.copyBufferToBuffer(gHorWeights, 0, horReadBuffer, 0, bufferSizeLatest);
            device.queue.submit([encoder2.finish()]);

            await verReadBuffer.mapAsync(GPUMapMode.READ);
            const verArrayBuffer = verReadBuffer.getMappedRange();
            const newVerWeights = new Int32Array(verArrayBuffer.slice(0));
            verReadBuffer.unmap();

            await horReadBuffer.mapAsync(GPUMapMode.READ);
            const horArrayBuffer = horReadBuffer.getMappedRange();
            const newHorWeights = new Int32Array(horArrayBuffer.slice(0));
            horReadBuffer.unmap();

            console.log(`vers=${newVerWeights.join(",")}`);
            console.log(`hors=${newHorWeights.join(",")}`);

            verReadBuffer.destroy();
            horReadBuffer.destroy();
        }
    }

    const verReadBuffer = device.createBuffer({
        size: bufferSizeBase,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const horReadBuffer = device.createBuffer({
        size: bufferSizeLatest,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    let logReadBuffer = null;
    if (isDebug && logBuffer) {
        logReadBuffer = device.createBuffer({
            size: bufferSizeLog,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
    }

    const encoderFinal = device.createCommandEncoder();
    encoderFinal.copyBufferToBuffer(gVerWeights, 0, verReadBuffer, 0, bufferSizeBase);
    encoderFinal.copyBufferToBuffer(gHorWeights, 0, horReadBuffer, 0, bufferSizeLatest);

    if (isDebug && logBuffer && logReadBuffer) {
        encoderFinal.copyBufferToBuffer(logBuffer, 0, logReadBuffer, 0, bufferSizeLog);
    }

    device.queue.submit([encoderFinal.finish()]);

    await verReadBuffer.mapAsync(GPUMapMode.READ);
    const verArrayBuffer = verReadBuffer.getMappedRange();
    const verResult = new Int32Array(verArrayBuffer.slice(0));
    verReadBuffer.unmap();

    await horReadBuffer.mapAsync(GPUMapMode.READ);
    const horArrayBuffer = horReadBuffer.getMappedRange();
    const horResult = new Int32Array(horArrayBuffer.slice(0));
    horReadBuffer.unmap();

    for (let i = 0; i < baseLen; i++) {
        verWeights[i] = verResult[i];
    }

    for (let i = 0; i < latestLen; i++) {
        horWeights[i] = horResult[i];
    }

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

    if (isDebug && logReadBuffer) {
        await logReadBuffer.mapAsync(GPUMapMode.READ);
        const logArrayBuffer = logReadBuffer.getMappedRange();
        const logData = new Uint32Array(logArrayBuffer.slice(0));
        logReadBuffer.unmap();

        console.log("GPU Log:", decodeLog(logData, workgroupCount, THREAD_LOG_SIZE));
    }

    // Cleanup
    gBases.destroy();
    gLatests.destroy();
    gVerWeights.destroy();
    gHorWeights.destroy();
    verReadBuffer.destroy();
    horReadBuffer.destroy();

    if (logBuffer) logBuffer.destroy();
    if (logReadBuffer) logReadBuffer.destroy();

    paramsBuffer.destroy();
}

/**
 * 验证输入数组是否符合要求
 * @param {number[]} originalValues
 * @param {boolean} IsSharedVersion
 * @param {number} step
 * @returns {number}
 */
function Valid(originalValues, IsSharedVersion, step) {
    if (originalValues.length === 0) {
        throw new Error("originalValues.length is invalid.");
    }

    if (IsSharedVersion) {
        if (!(1 <= step && step <= 256)) {
            throw new Error("step is invalid.");
        }
    } else {
        if (![2, 4, 8, 16].includes(step)) {
            throw new Error("step is invalid.");
        }
    }

    if (originalValues.length < step)
        throw new Error("N must be less than or equal to the length of the original array.");

    if (originalValues.length % step !== 0) {
        throw new Error("originalValues.length % step != 0");
    }

    return originalValues.length / step;
}
