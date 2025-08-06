namespace MegaLCSLib.OpenCL;

// 实际测试，这个核函数性能不太行
public partial class Mega{

    // __STEP__ MUST = 2,4,8,16
    private const string NanoLCS_GotoRightBottom_Kernel_Register = @"
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

__kernel void NanoLCS_GotoRightBottom_Kernel(
    __global int *gBases,
    __global int *gLatests,
    __global int *gVerWeights,
    __global int *gHorWeights,
    const int baseSliceSize,
    const int latestSliceSize,
    const int outerWFID,
    const int totalThread) {

    // 实现方法和CUDA类似
    const int threadGlobalID = get_global_id(0);
    const int blockIdx = get_group_id(0);
    const int threadIdx = get_local_id(0);
    
#ifdef DEBUG
    printf(""threadGlobalID=%d threadIdx=%d | blockIdx=%d | totalThread=%d\n"", 
            threadGlobalID,threadIdx,blockIdx,totalThread);
#endif

    // 丢弃不在范围内的线程，做边界保护
    if (threadGlobalID >= totalThread) {
        return;
    }

    if (threadIdx == 0) {

        int__STEP__ bases;
        int__STEP__ latests;
        int__STEP__ hors;
        int__STEP__ vers;

        // 计算当前处理的latestSlice范围: latest轴相当于X轴/水平轴
        const int latestSliceIDMin = max(0, outerWFID - (baseSliceSize - 1));
        // 计算当前处理的latestSliceID,相当于slice(cpu)/group(opencl)/block(cuda)级别的offset 每个block处理一个slice
        const int latestSliceID = latestSliceIDMin + blockIdx;
        const int baseSliceID = outerWFID - latestSliceID;

#ifdef DEBUG
        printf(""threadGlobalID=%d threadIdx=%d | outerWFID=%d latestSliceIDMin=%d blockIdx=%d latestSliceID=%d baseSliceID=%d\n"", 
                threadGlobalID, threadIdx, 
                outerWFID,latestSliceIDMin,blockIdx,latestSliceID,baseSliceID);

        printf(""threadGlobalID=%d threadIdx=%d | Before load gBases=[%d %d ..] gLatests=[%d %d ..] gHors=[%d %d ..] gVers=[%d %d ..] ...\n"",
                threadGlobalID, threadIdx
                , gBases[0], gBases[1]
                , gLatests[0], gLatests[1]
                , gHorWeights[0], gHorWeights[1]
                , gVerWeights[0], gVerWeights[1]);
#endif

        // 设备端全局内存搬迁到设备端共享内存，一次性加载__STEP__个元素
        // vload加载第一个参数是按照sliceID作为基本单位的
        bases = vload__STEP__(baseSliceID, gBases);
        latests = vload__STEP__(latestSliceID, gLatests);
        
        vers = vload__STEP__(baseSliceID, gVerWeights);        
        hors = vload__STEP__(latestSliceID, gHorWeights);
        
#ifdef DEBUG
        // 每当开始一行的计算，打印一次内部的所有值，调试用，最多打印前2个值
        printf(""threadGlobalID=%d threadIdx=%d | Before calc  bases=[%d %d ..]  latests=[%d %d ..]  hors=[%d %d ..]  vers=[%d %d ..] ...\n"",
                threadGlobalID, threadIdx
                , bases[0], bases[1]
                , latests[0], latests[1]
                , hors[0], hors[1]
                , vers[0], vers[1]);
#endif

        /*
        开始计算。实测Tesla p40 1048576*1048576在STEP=16,加载和保存10秒，计算为20秒
        大致说明开启核函数和数据加载保存并不是主要矛盾，而且下面的计算应该已经是最高效的
        不搞向量化计算版本，太复杂了，意义不大
        反而当STEP=8时，开启了大量的核函数，导致总时间上升到60秒

        共享内存版本，实测Tesla p40 1048576*1048576在STEP=32,加载和保存3秒，计算为35秒
        可以理论分析，假设寄存器版本计算循环中纯粹计算10秒，读取寄存器10秒。相同的逻辑，则共享内存读取需要25秒。
        即使共享内存版本的纯粹计算做得很复杂搞miniWavefront，也不可能压低到0秒，因此总体不可能低于25秒。
        */
        
        int leftTop = 0; // 模拟DP左上角初始值
        int horLBackup = 0; // 备份当前值，供下一轮使用

        #pragma unroll
        for (int b = 0; b < __STEP__; b++){
            // 每一行的初始权重不是0，而是hors里面的第0个元素开始的,和参数b无关
            horLBackup = hors[0];

            #pragma unroll
            for (int l = 0; l < __STEP__; l++){
                leftTop = horLBackup;
                horLBackup = hors[l];

                // 高度优化：不相等的先命中
                if (bases[b] != latests[l]){
                    // 高度优化：不相等的先命中
                    // 实测：即使优化写法尽量不更新hors[l]性能也不会有显著提升反而代码很难看
                    // 实测：下面这种写法已经是最优解
                    // 实测：因为已经都是寄存器了，所以current_left写法意义不大，反而性能不稳定
                    // 实测：max改成?:也差不多，反而代码变复杂了
                    hors[l] = l != 0
                        ? max(hors[l], hors[l - 1])
                        : max(hors[l], vers[b]);       // 左侧没有hors，因此和vers[b]比较
                }
                else{
                    hors[l] = leftTop + 1;
                }
            } // end for l

            // 每完成一次base/Y轴方向的处理后，当前hors权重的最后一个值需要存储到纵向权重
            vers[b] = hors[__STEP__ - 1];

#ifdef DEBUG
            // 每完成一行计算，打印日志，进行调试
            printf(""threadGlobalID=%d threadIdx=%d | After  calc b=%d                                hors=[%d %d ..]  vers=[%d %d ..]...\n"",
                        threadGlobalID, threadIdx, b
                        , hors[0], hors[1]
                        , vers[0], vers[1]);
#endif
            
        } // end for b

        // 设备端共享内存搬迁到设备端全局内存，一次性写入__STEP__个元素
        vstore__STEP__(hors, latestSliceID, gHorWeights);
        vstore__STEP__(vers, baseSliceID, gVerWeights);

#ifdef DEBUG
        printf(""threadGlobalID=%d threadIdx=%d | After store                                   gHors=[%d %d ..] gVers=[%d %d ..] ...\n"",
                threadGlobalID, threadIdx
                , gHorWeights[0], gHorWeights[1]
                , gVerWeights[0], gVerWeights[1]);
#endif

    } // if (threadIdx == 0)

}";
}