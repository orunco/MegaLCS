namespace MegaLCSLib.OpenCL;

public partial class Mega{

    // __STEP__ MUST = [1->256]
    private const string KernelLCS_Shared = @"
/*
Copyright (C) 2025 Pete Zhang, rivxer@gmail.com, https://github.com/orunco

Licensed under the Apache License, Version 2.0 (the ""License"");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an ""AS IS"" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

__kernel void KernelLCS_NoDependency(
    __global int *gBases,
    __global int *gLatests,
    __global int *gVerWeights,
    __global int *gHorWeights,
    const int baseSliceSize,
    const int latestSliceSize,
    const int outerW,
    const int totalThread) {

#ifdef DEBUG
    const int step = __STEP__;
#endif

    // 实现方法和CUDA类似
    const int threadGIdx = get_global_id(0);
    const int blockIdx = get_group_id(0);
    const int threadIdx = get_local_id(0);

#ifdef DEBUG
    printf(""outerW=%d block =%d> thread=g%d,%d| totalThread=%d (in device)\n"",
            outerW,blockIdx,
            threadGIdx,threadIdx,totalThread);
#endif

    // 丢弃不在范围内的线程，做边界保护
    if (threadGIdx >= totalThread) {
        return;
    }

    // 共享内存
    __local int bases[__STEP__];
    __local int latests[__STEP__];
    __local int vers[__STEP__];
    __local int hors[__STEP__];
    
    // 计算当前处理的latestSlice范围: latest轴相当于X轴/水平轴
    const int latestSliceIDMin = max(0, outerW - (baseSliceSize - 1));
    // 计算当前处理的latestSliceID,相当于slice(cpu)/group(opencl)/block(cuda)级别的offset，每个block处理一个slice
    const int latestSliceID = latestSliceIDMin + blockIdx;
    const int baseSliceID = outerW - latestSliceID;

    // 计算当前全局内存偏移量
    const int baseGlobalOffset = baseSliceID * __STEP__;
    const int latestGlobalOffset = latestSliceID * __STEP__;
    
    const int baseValGlobalOffset = baseGlobalOffset + threadIdx;
    const int latestValGlobalOffset = latestGlobalOffset + threadIdx;
    

#ifdef DEBUG
    printf(""outerW=%d block =%d> thread=g%d,%d| baseSliceID=%d latestSliceIDMin=%d latestSliceID=%d\n"", 
            outerW, blockIdx
            ,threadGIdx, threadIdx 
            ,baseSliceID, latestSliceIDMin,latestSliceID);

    printf(""                   thread=g%d,%d| INIT    gBases={%d %d ..} gLatests={%d %d ..} gVers={%d %d ..} gHors={%d %d ..} \n"",
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
        printf(""                   thread=g%d,%d| LOAD    baseValGlobalOffset=g%d latestValGlobalOffset=g%d\n"",
            threadGIdx, threadIdx
            , baseValGlobalOffset
            , latestValGlobalOffset);
#endif
    } // end of load

    // 等待所有线程完成数据加载
    barrier(CLK_LOCAL_MEM_FENCE);


#ifdef DEBUG
    // 打印一次内部的所有值，调试用，最多打印前2个值，下面的+0 +1是为了代替threadIdx
    if( step == 1 ){
        printf(""                   thread=g%d,%d| LOAD OK bases={%d} latests={%d} hors={%d} vers={%d}\n"",
            threadGIdx, threadIdx
            , bases[0], latests[0]                       
            , vers[0], hors[0]);    
    }else if( step == 2 ){
        printf(""                   thread=g%d,%d| LOAD OK bases={%d %d} latests={%d %d} vers={%d %d} hors={%d %d}\n"",
            threadGIdx, threadIdx
            , bases[0], bases[1]  
            , latests[0], latests[1]
            , vers[0], vers[1]          
            , hors[0], hors[1]);      
    }else{
        printf(""                   thread=g%d,%d| LOAD OK bases={..} latests={..} vers={..} hors={..} \n"",
            threadGIdx, threadIdx);      
    }

#endif


    /*
    下面这段代码是平行世界，需要带入每一个线程的视角去考虑，而且线程的展开类似Hi算法，是一个迅速展开的过程
    绝不是一个平凡的过程
    先是outerW的展开，然后是innerWaveFrontLine的线程展开，画图吧，否则难以理解
    也许是因为共享内存被加载到了寄存器，又因为全部并行化了，所以性能提升很快
    */
    for (int innerWaveFrontLine = 0; 
             innerWaveFrontLine < 2 * __STEP__ - 1; 
             innerWaveFrontLine++) {
        // 每个wavefront对应一条反斜对角线,每个线程根据 wavefrontID 计算自己的坐标 (l,b)         
        int l = threadIdx;        // X轴坐标（latest索引）
        int b = innerWaveFrontLine - l;       // Y轴坐标（base索引）

        // 线程激活逻辑: 仅允许对角线上的有效坐标参与计算，通过下列条件自动过滤无效坐标，无需硬编码。
        if (b >= 0 && b < __STEP__ && l >= 0 && l < __STEP__) {
            // 左值：当l>0时使用本行前一个值，l=0时使用纵向基础权重
            // 特殊点：左侧无元素，和基础权重vers[b]比较，而不是和0比较
            int leftWeight = vers[b];
            // 上值：当b>0时使用上一行同列值
            // 特殊点：当b=0，使用基础权重
            int topWeight = hors[l];

            // 计算对角值的三种边界情况【推导可知】
            int leftTopWeight = min(leftWeight, topWeight);

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
            printf(""                   thread=g%d,%d| ACTIVE  b=%d l=%d left=%d top=%d leftTop=%d => vers[%d]=%d hors[%d]=%d\n"",
            threadGIdx, threadIdx
            ,b ,l 
            ,leftWeight, topWeight, leftTopWeight
            ,b,vers[b]
            ,l,hors[l]);
#endif
        } // 确保在指定的线程内处理

#ifdef DEBUG
        printf(""                   thread=g%d,%d| BARRIER\n"",threadGIdx, threadIdx);
#endif
        // 等待当前wavefront的所有线程完成计算
        barrier(CLK_LOCAL_MEM_FENCE);
    } // end for innerWaveFrontLine

    // 等待所有线程完成计算
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef DEBUG
    // 打印一次内部的所有值，调试用，最多打印前2个值
    if( step == 1 ){
        printf(""                   thread=g%d,%d| CALC OK bases={%d} latests={%d} vers={%d} hors={%d} \n"",
            threadGIdx, threadIdx
            , bases[0]
            , latests[0]  
            , vers[0]                     
            , hors[0]);    
    }else if( step == 2 ){
        printf(""                   thread=g%d,%d| CALC OK bases={%d %d} latests={%d %d} vers={%d %d} hors={%d %d}\n"",
            threadGIdx, threadIdx
            , bases[0], bases[1] 
            , latests[0], latests[1]
            , vers[0], vers[1]       
            , hors[0], hors[1]);      
    }else{
        printf(""                   thread=g%d,%d| CALC OK bases={..} latests={..} vers={..} hors={..}\n"",
            threadGIdx, threadIdx);      
    }

#endif

    // 设备端共享内存搬迁到设备端全局内存，线程和元素正好一一对应
    if (threadIdx < __STEP__) {
        gVerWeights[baseValGlobalOffset] = vers[threadIdx];
        gHorWeights[latestValGlobalOffset] = hors[threadIdx];
        
#ifdef DEBUG
        printf(""                   thread=g%d,%d| WRIT    baseValGlobalOffset=g%d latestValGlobalOffset=g%d \n"",
            threadGIdx, threadIdx
            , baseValGlobalOffset
            , latestValGlobalOffset);
#endif
    }

#ifdef DEBUG
    // 打印一次内部的所有值，调试用，最多打印前2个值，下面的+0 +1是为了代替threadIdx
    if( step == 1 ){        
        printf(""                   thread=g%d,%d| WRIT OK vers={%d} gvers={%d->g%d} hors={%d} ghors={%d->g%d}\n"",
            threadGIdx, threadIdx   
            , vers[0]
            , gVerWeights[baseGlobalOffset+0], baseGlobalOffset+0       
            , hors[0]
            , gHorWeights[latestGlobalOffset+0], latestGlobalOffset+0);    
    }else if( step == 2 ){
        printf(""                   thread=g%d,%d| WRIT OK vers={%d %d} gvers={%d->g%d %d->g%d} hors={%d %d} ghors={%d->g%d %d->g%d}\n"",
            threadGIdx, threadIdx          
            , hors[0], hors[1]
            , vers[0], vers[1] 
            , gVerWeights[baseGlobalOffset+0], baseGlobalOffset+0
            , gVerWeights[baseGlobalOffset+1], baseGlobalOffset+1
            , gHorWeights[latestGlobalOffset+0], latestGlobalOffset+0
            , gHorWeights[latestGlobalOffset+1], latestGlobalOffset+1);        
    }else{
        printf(""                   thread=g%d,%d| WRIT OK vers={..} gvers={..} hors={..} ghors={..} \n"",
            threadGIdx, threadIdx);      
    }

#endif
}";
}