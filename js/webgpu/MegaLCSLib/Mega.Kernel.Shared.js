// __STEP__ MUST = [1->256]
const KernelLCS_MinMax_WGSL_Template = `
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

#ifdef DEBUG
//---------缓冲区printf默认代码---------------------------------------
/*
- 本质：GPU 无法直接调用 console.log，必须通过缓冲区传递数据到 CPU
- 原理：
    在着色器中把数据写入特定缓冲区
    执行完成后，CPU 从缓冲区读取数据
    在JavaScript 中解析并打印
*/

// 存储结构，每一个线程的日志使用同一个buffer的不同位置，这个是全局唯一的
// 包含了多个工作组
struct LogBuffer {
    data: array<u32>,
}

// 获取当前线程的完整日志偏移
fn get_thread_log_offset(workgroup_id: u32, thread_id: u32) -> u32 {
    // 每个工作组有2个线程可以打印日志
    // 总偏移 = 工作组ID * 2 * 日志大小 + 线程ID * 日志大小
    return workgroup_id * (2 * __THREAD_LOG_SIZE__) + thread_id * __THREAD_LOG_SIZE__;
}

// 每一个线程的游标，最多支持前2个线程打印，太多根本看不清楚执行路径
// 注意：这里是每一个工作组下面的前2个线程可以打印日志，内核函数可能会有非常多的工作组
// log_cursors 是工作组内的共享变量，每个工作组有自己的一份
// 也就是虽然看起来代码只定义了2个日志流，其实是定义了2*工作组数多个日志流
var<workgroup> log_cursors: array<u32, 2>;


fn c(c: u32, workgroup_idx: u32, thread_idx: u32) {
 
    if (thread_idx >= 2) {
        return; // 不记录超出范围的线程，防止越界
    }

    let base_offset = get_thread_log_offset(workgroup_idx, thread_idx);
    let cursor_pos = log_cursors[thread_idx];

    if cursor_pos < __THREAD_LOG_SIZE__ - 1 {
        gLog.data[base_offset + cursor_pos] = c;
        log_cursors[thread_idx] = cursor_pos + 1;
    } else if cursor_pos == __THREAD_LOG_SIZE__ - 1 {
        gLog.data[base_offset + cursor_pos] = 126; // 126表达'~' 表示日志尾部溢出
        log_cursors[thread_idx] = cursor_pos + 1;
    }
    // 超出部分直接忽略
}

fn i(num: i32, workgroup_idx: u32,  thread_idx: u32) {
    var n = num;
    if (n == 0) {
        c(48, workgroup_idx, thread_idx); // '0'
        return;
    }
    if (n < 0) {
        c(45, workgroup_idx, thread_idx); // '-'
        n = -n;
    }
    var temp: array<u32, 10>;
    var idx: u32 = 0;
    while (n > 0) {
        temp[idx] = 48 + u32(n % 10);
        n = n / 10;
        idx = idx + 1;
    }
    while (idx > 0) {
        idx = idx - 1;
        c(temp[idx], workgroup_idx, thread_idx);
    }
}

//---------缓冲区printf默认代码---------------------------------------
#endif


struct Params {
    baseSliceSize: i32,
    latestSliceSize: i32,
    outerW: i32,
    totalThread: i32,
};

//与CUDA不同，在CUDA中共享内存可以在函数内部声明。
//在WGSL中，工作组变量必须在模块作用域声明，但在执行期间仍局限于每个工作组
// 共享内存
var<workgroup> bases: array<i32, __STEP__>;
var<workgroup> latests: array<i32, __STEP__>;
var<workgroup> vers: array<i32, __STEP__>;
var<workgroup> hors: array<i32, __STEP__>;
    
// __kernel void KernelLCS_MinMax(
    @group(0) @binding(0) var<storage, read> gBases: array<i32>;
    @group(0) @binding(1) var<storage, read> gLatests: array<i32>;
    @group(0) @binding(2) var<storage, read_write> gVerWeights: array<i32>;
    @group(0) @binding(3) var<storage, read_write> gHorWeights: array<i32>;
    @group(0) @binding(4) var<uniform> gParams: Params;
    
#ifdef DEBUG    
    @group(0) @binding(5) var<storage, read_write> gLog: LogBuffer;
#endif

    
#ifdef DEBUG
    const step: i32 = __STEP__;
#endif

@compute @workgroup_size(__STEP__)
fn KernelLCS_MinMax(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // 实现方法和CUDA类似
    let threadGIdx = i32(global_id.x);
    let blockIdx = i32(workgroup_id.x);
    let threadIdx = i32(local_id.x);

    let baseSliceSize = gParams.baseSliceSize;
    let latestSliceSize = gParams.latestSliceSize;
    let outerW = gParams.outerW;
    let totalThread = gParams.totalThread;
    
#ifdef DEBUG    
//---------缓冲区printf默认代码---------------------------------------

    // 这里可以非常肯定，必须要workgroup_id和工作组内的threadIdx配合，否则无法唯一定位日志
    let w = u32(blockIdx); // 简化宏定义代码展开 
    let t = u32(threadIdx); // 简化宏定义代码展开
    if (threadIdx < 2) { // 确保每个工作组内的前2个log初始化位置
        log_cursors[threadIdx] = 0;//确保在 c() 被调用前执行
    }
    workgroupBarrier();
    
//---------缓冲区printf默认代码---------------------------------------
#endif


#ifdef DEBUG
    printf("outerW=%d block =%d> thread=g%d,%d| totalThread=%d (in device)\n",
            outerW,blockIdx,
            threadGIdx,threadIdx,totalThread);
#endif

    // workgroupBarrier要求工作组（workgroup）内的所有线程必须全部执行到这一行，才能继续往下运行。
    // 如果某些线程提前退出（比如原来if (threadGIdx >= totalThread) { return; }），就会导致死锁或未定义行为
    // 不要用 return 提前退出，而是让所有线程都执行完整代码，但通过条件判断跳过无效线程的计算
    let activeThread = threadGIdx < totalThread;
    // 丢弃不在范围内的线程，做边界保护
    // if (threadGIdx >= totalThread) {
    //    return;
    // }


    // 计算当前处理的latestSlice范围: latest轴相当于X轴/水平轴
    let latestSliceIDMin = max(0, outerW - (baseSliceSize - 1));
    // 计算当前处理的latestSliceID,相当于slice(cpu)/group(opencl)/block(cuda)级别的offset，每个block处理一个slice
    let latestSliceID = latestSliceIDMin + blockIdx;
    let baseSliceID = outerW - latestSliceID;

    // 计算当前全局内存偏移量
    let baseGlobalOffset = baseSliceID * __STEP__;
    let latestGlobalOffset = latestSliceID * __STEP__;
    let baseValGlobalOffset = baseGlobalOffset + threadIdx;
    let latestValGlobalOffset = latestGlobalOffset + threadIdx;
    

#ifdef DEBUG
    printf("baseSliceID=%d latestSliceID=%d baseGlobalOffset=%d  latestGlobalOffset=%d\n",
       baseSliceID, latestSliceID, baseGlobalOffset, latestGlobalOffset);
       
    printf("outerW=%d block =%d> thread=g%d,%d| baseSliceID=%d latestSliceIDMin=%d latestSliceID=%d\n",
            outerW, blockIdx
            ,threadGIdx, threadIdx 
            ,baseSliceID, latestSliceIDMin,latestSliceID);

    printf("                   thread=g%d,%d| INIT    gBases={%d %d ..} gLatests={%d %d ..} gVers={%d %d ..} gHors={%d %d ..} \n",
            threadGIdx, threadIdx
            , gBases[0], gBases[1]
            , gLatests[0], gLatests[1]
            , gVerWeights[0], gVerWeights[1]
            , gHorWeights[0], gHorWeights[1]);
#endif

    // 设备端全局内部搬迁到设备端全局内存，线程和元素正好一一对应
    // 假设STEP=2，则需要搬迁2次，分别是0 1【外层会启动好2个线程】
    if (threadIdx < __STEP__) {
        bases[threadIdx] = gBases[baseValGlobalOffset];
        vers[threadIdx] = gVerWeights[baseValGlobalOffset];

        latests[threadIdx] = gLatests[latestValGlobalOffset];
        hors[threadIdx] = gHorWeights[latestValGlobalOffset];

#ifdef DEBUG
        printf("                   thread=g%d,%d| LOAD    baseValGlobalOffset=g%d latestValGlobalOffset=g%d\n",
            threadGIdx, threadIdx
            , baseValGlobalOffset
            , latestValGlobalOffset);
#endif
    } // end of load

    // 等待所有线程完成数据加载
    workgroupBarrier();


#ifdef DEBUG
    // 打印一次内部的所有值，调试用，最多打印前2个值，下面的+0 +1是为了代替threadIdx
    if( step == 1 ){
        printf("                   thread=g%d,%d| LOAD OK bases={%d} latests={%d} hors={%d} vers={%d}\n",
            threadGIdx, threadIdx
            , bases[0], latests[0]                       
            , vers[0], hors[0]);    
    } 

#endif


    /*
    下面这段代码是平行世界，需要带入每一个线程的视角去考虑，而且线程的展开类似Hi算法，是一个迅速展开的过程
    绝不是一个平凡的过程
    先是outerW的展开，然后是innerWaveFrontLine的线程展开，画图吧，否则难以理解
    也许是因为共享内存被加载到了寄存器，又因为全部并行化了，所以性能提升很快
    */
    for (var innerWaveFrontLine: i32 = 0; 
             innerWaveFrontLine < 2 * __STEP__ - 1; 
             innerWaveFrontLine++) {
        // 每个wavefront对应一条反斜对角线,每个线程根据 wavefrontID 计算自己的坐标 (l,b)         
        let l = threadIdx;        // X轴坐标（latest索引）
        let b = innerWaveFrontLine - l;       // Y轴坐标（base索引）

        // 线程激活逻辑: 仅允许对角线上的有效坐标参与计算，通过下列条件自动过滤无效坐标，无需硬编码。
        if (b >= 0 && b < __STEP__ && l >= 0 && l < __STEP__) {
            // 左值：当l>0时使用本行前一个值，l=0时使用纵向基础权重
            // 特殊点：左侧无元素，和基础权重vers[b]比较，而不是和0比较
            let leftWeight = vers[b];
            // 上值：当b>0时使用上一行同列值
            // 特殊点：当b=0，使用基础权重
            let topWeight = hors[l];

            // 计算对角值的三种边界情况【推导可知】
            let leftTopWeight = min(leftWeight, topWeight);

            // 每个hors/vers只会被一个线程更新，不会出现竞争条件
            if (bases[b] == latests[l]) {
                hors[l] = leftTopWeight + 1;
            } else {
                // 不匹配时取max(左值, 上值)
                hors[l] = max(leftWeight, topWeight);
            }

            // hors代表了横向方向的逐个更新，而vers代表了纵向方向的逐个更新
            // 从整体宏观角度看，当前算法的核函数的vers和hors是对称的
            // 这个和传统的单线程版本只需要更新尾部vers不太一样，是平行时空
            vers[b] = hors[l];

#ifdef DEBUG
            printf("                   thread=g%d,%d| ACTIVE  b=%d l=%d left=%d top=%d leftTop=%d => vers[%d]=%d hors[%d]=%d\n",
            threadGIdx, threadIdx
            ,b ,l 
            ,leftWeight, topWeight, leftTopWeight
            ,b,vers[b]
            ,l,hors[l]);
#endif
        } // 确保在指定的线程内处理

#ifdef DEBUG
        printf("                   thread=g%d,%d| BARRIER\n",threadGIdx, threadIdx);
#endif
        // 等待当前wavefront的所有线程完成计算
        workgroupBarrier();
    } // end for innerWaveFrontLine
    
    // 等待所有线程完成计算
    workgroupBarrier();

#ifdef DEBUG
    // 打印一次内部的所有值，调试用，最多打印前2个值
    if( step == 1 ){
        printf("                   thread=g%d,%d| CALC OK bases={%d} latests={%d} vers={%d} hors={%d} \n",
            threadGIdx, threadIdx
            , bases[0]
            , latests[0]  
            , vers[0]                     
            , hors[0]);    
    }

#endif

    // 设备端共享内存搬迁到设备端全局内存，线程和元素正好一一对应
    if (threadIdx < __STEP__) {
        gVerWeights[baseValGlobalOffset] = vers[threadIdx];
        gHorWeights[latestValGlobalOffset] = hors[threadIdx];
        
#ifdef DEBUG
        printf("                   thread=g%d,%d| WRIT    baseValGlobalOffset=g%d latestValGlobalOffset=g%d \n",
            threadGIdx, threadIdx
            , baseValGlobalOffset
            , latestValGlobalOffset);
#endif
    }

#ifdef DEBUG
    // 打印一次内部的所有值，调试用，最多打印前2个值，下面的+0 +1是为了代替threadIdx
    if( step == 1 ){        
        printf("                   thread=g%d,%d| WRIT OK vers={%d} gvers={%d->g%d} hors={%d} ghors={%d->g%d}\n",
            threadGIdx, threadIdx   
            , vers[0]
            , gVerWeights[baseGlobalOffset+0], baseGlobalOffset+0       
            , hors[0]
            , gHorWeights[latestGlobalOffset+0], latestGlobalOffset+0);    
    } 

#endif
}
`;

// Function to generate the final WGSL code
function generateWGSLKernel(options = {}) {
    const {step, threadLogSize, debug} = options;

    let wgsl = KernelLCS_MinMax_WGSL_Template
        .replace(/__STEP__/g, step.toString())
        .replace(/__THREAD_LOG_SIZE__/g, threadLogSize.toString());

    if (debug) {
        wgsl = wgsl
            .replace(/#ifdef DEBUG/g, '')
            .replace(/#endif/g, '');

        // 展开调试
        wgsl = expandPrintfMacros(wgsl);

    } else {
        // Remove all debug-related code
        wgsl = wgsl
            .replace(/#ifdef DEBUG[\s\S]*?#endif/g, '')
            .replace(/printf/g, '// printf removed');
    }

    if (debug) {
        console.log(wgsl);
    }

    return wgsl;
}
