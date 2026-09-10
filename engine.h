/* ============================================================================
 * engine.h - C 引擎跨模块接口与"单次请求上下文"(EngineJob)
 *
 * 模块划分:
 *   server.c   HTTP 服务入口, 收到 POST /api/run 后调用 runEngineRequest();
 *   jsonx.c    JSON 解析/写出(jsonx.h), 与仿真逻辑无关;
 *   cfg.c      请求配置解码 + 跨页一致性校验 + 单位换算(cycles 转换);
 *   work.c     建模: 激励(模型生成 / 算子编排) -> 逻辑内核序列;
 *   sim.c      Cycle 级近似离散仿真引擎;
 *   run.c      顶层编排: 解码 -> 建模 -> 多算法仿真 -> 结果 JSON。
 *
 * 数据流(一次请求):
 *   runEngineRequest()
 *     -> decodeJobConfig()        把请求 JSON 解码为 EngineJob
 *     -> buildWorkloadKernels()   生成 WorkKernel 内核序列
 *     -> simulateAlgorithm()×N    逐调度算法跑 Cycle 级仿真
 *     -> simulateDetailLanes()    输出泳道/气泡等可视化详情
 * ==========================================================================*/
#ifndef ENGINE_H
#define ENGINE_H

#include "simcore.h"
#include "jsonx.h"
#include "logx.h"

/* ============================================================================
 * **【GEMM_1】本头文件在贯穿用例中的角色(接口面, 无实现)
 * 对 GEMM_1(单算子/单卡/单算法 "dfs"), 一次 runEngineRequest 的关键对象:
 *   job.hardware: 1卡×8核/1GHz/320T Cube/UB256KB/HBM1.2TB/s(见 cfg.c 解码);
 *   job.stimulus.mode=STIMULUS_OPERATORS(ops 编排), repeat=1;
 *   job.algorithms[0]: id="dfs", 全默认参(group0/dbuf2/maxChunk320/ubFrac0.66);
 *   job.kernels: 恰好 1 个(op0-GEMM, KERNEL_CUBE), kernelCount=1,
 *     cubeFlops≈1.3743895e11, loadBytes=67,108,864, storeBytes=33,554,432;
 *   job.lowerBoundCycles=Tlb≈429,497(work.c 建模时按"每卡并行上限"算好);
 *   job.detailView: algo=0/npu=0/core0=0/core1=7/bins=80(泳道行数=4+8×3=28)。
 * 下面的 AlgorithmResult 是 simulateAlgorithm() 对每个算法写回的聚合指标,
 * run.c 再逐字段搬到 scoreboard JSON —— 每个字段的 GEMM_1 实测值已列于字段表。
 * ==========================================================================*/

/* ============================================================================
 * 单次 /api/run 请求的完整上下文(贯穿 解码->建模->仿真->出JSON)
 * ==========================================================================*/
typedef struct {
    HardwareConfig   hardware;            /* 硬件配置(解码 + 校验后)          */
    ModelConfig      model;               /* 模型配置                          */
    StimulusConfig   stimulus;            /* 激励来源与参数                    */
    AlgorithmConfig  algorithms[MAX_ALGORITHMS]; /* 待对比的调度算法列表      */
    int              algorithmCount;      /* 实际参与对比的算法个数            */
    DetailViewConfig detailView;          /* 泳道详情视图参数                  */
    char             errorMessage[512];   /* 阻断性错误(失败时回给前端)        */
    char             warningMessage[512]; /* 非阻断告警(如每卡流式数据超 HBM)  */
    WorkKernel      *kernels;             /* 建模生成的内核数组(work.c 分配)   */
    int              kernelCount;         /* 内核个数                          */
    int              pipelineActive;      /* 是否启用 PP 流水并行(pp>1)        */
    int              microBatchWaves;     /* 每阶段波形数(≈microBatch)         */
    double           lowerBoundCycles;    /* 理论下界(cycles, 效率分母)        */
    double           ffnWidth;            /* 自定义模型的 FFN 宽度(配置键 ff)  */
} EngineJob;

/* ============================================================================
 * **【GEMM_1】本用例的瓶颈判定结论(BOUND_COMPUTE)
 * sim.c simulateAlgorithm 末尾比较三个归一化份额:
 *   cubeFrac=cubeBusy/分母≈3,435,970/3,455,304≈0.994(Cube 几乎满);
 *   memFrac=(mte2+mte3)/(2×T)≈104,366/863,826≈0.121;
 *   controlFrac=aicpu/(T)≈10,240/431,913≈0.024; stallFrac≈10,820/3,455,304≈0.003。
 * 三个判定分支: controlFrac>0.5 且大于其余 → BOUND_OVERHEAD;
 * stallFrac>0.5 且大于其余 → BOUND_OVERHEAD; 否则 cubeFrac≥memFrac → COMPUTE,
 * 否则 MEMORY。GEMM_1: 开销/气泡份额都≪0.5, 且 0.994≥0.121 ⇒ Compute-bound
 * (0.5 阈值的含义: 只有"等待类开销"占掉一半以上时间槽才算 Overhead-bound)。
 * ==========================================================================*/

/* ============================================================================
 * 受限类型判定(BottleneckKind): 说明系统当前被哪类资源卡住
 * ==========================================================================*/
typedef enum {
    BOUND_COMPUTE  = 0,   /* 计算受限(Compute-bound)      */
    BOUND_MEMORY   = 1,   /* 访存受限(Memory-bound)       */
    BOUND_OVERHEAD = 2,   /* 控制/开销受限(Overhead-bound) */
    BOUND_BALANCED = 3    /* 均衡(Balanced, 兜底)          */
} BottleneckKind;

/* ============================================================================
 * **【GEMM_1】AlgorithmResult 字段 ↔ 仿真指标 ↔ scoreboard 键 对照(全为 GEMM_1 实测值)
 *   · algoKey/algoName: 算法标识/名("dfs"/"默认参数调度"), scoreboard id/name。
 *   · totalCycles: 端到端总周期; ops 模式=全局最后微任务结束 cycle=431,913
 *     (Tlb=429,497 之上 +0.56%: 逐任务 40/15/40/40 cyc 时延与排队起停),
 *     scoreboard totalCycles/totalUs(÷freq×1000)。
 *   · cubeBusyCycles/vectorBusyCycles: 全卡 Cube/Vector 忙时合计
 *     (durationCycles 累加)=3,435,970 / 0; 利用率分子; scoreboard cubeBusy/vecBusy。
 *   · mte2BusyCycles/mte3BusyCycles: 载入/写出引擎忙时≈66,163/38,202;
 *     合计 dmaBusy≈104,366(scoreboard 键 dmaBusy 直接=mte2Busy+mte3Busy)。
 *   · aicpuBusyCycles=10,240(40cyc×256任务) → scoreboard ctrlBusy;
 *     commBusyCycles=0(单卡无 HCCS) → commBusy。
 *   · stallCycles: 计算任务在单元队列上的等待(依赖晚于队尾)=10,820。
 *   · loadDmaBytes/storeDmaBytes: MTE2/MTE3 累计字节=67,108,864/33,554,432;
 *     totalCommBytes=0; dmaBytes=load+store=100,663,296(掩盖率分母);
 *     maskedBytes≈100,270,000(被计算覆盖的字节)→ scoreboard ldBytes/stBytes。
 *   · sramPeakKB: 片上驻留峰值≈640KB(=(chunkIn×dbuf2+chunkOut)/1024)。
 *   · loadBalanceDeviation: 负载均衡偏差; 单卡单阶段=0(scoreboard lbDev)。
 *   · cubeUtilizationPct/vectorUtilizationPct/mte2UtilizationPct/mte3UtilizationPct/
 *     mteUtilizationPct: 99.44/0/15.32/8.84/12.08(口径见 sim.c)→ scoreboard
 *     utilCube/utilVec/utilMte2/utilMte3/utilMte。
 *   · bubblePercent=0.559(100-计算占用率) → bubblePct。
 *   · memoryMaskRate=0.9961(≈99.61%, 掩盖率) → maskRate。
 *   · lowerBoundCycles: 理论下界=429,497(效率=431,913 的分母)。
 *   · compositeScore=99.575 =100×(0.35·eff+0.2·(1-bub)+0.3·mask+0.15·(1-lb))
 *     → scoreboard score。
 *   · bottleneck=BOUND_COMPUTE(cubeFrac≈0.994>memFrac≈0.121 且无开销/气泡超标)
 *     → scoreboard bound 文本"Compute-bound 计算受限"。
 *   · npuDutyMax/npuDutyMin/npuDutyAvg/serialEquivalentCycles/pipelineFillCycles:
 *     ops 模式无 PP,
 *     duty 恒 1.0、后两者恒 0(GEMM_1 无波形概念)。
 * ==========================================================================*/

/* ============================================================================
 * 单条调度算法的仿真结果聚合(AlgorithmResult)
 * ==========================================================================*/
typedef struct {
    char     algoKey[24];             /* 算法标识(拷贝自 AlgorithmConfig)   */
    char     algoName[80];            /* 算法显示名                         */
    double   totalCycles;             /* 端到端总 Cycles(PP 波形合并后)     */
    double   cubeBusyCycles;          /* Cube 计算单元累计忙时             */
    double   vectorBusyCycles;        /* Vector 计算单元累计忙时           */
    double   mte2BusyCycles;          /* MTE2 载入单元累计忙时             */
    double   mte3BusyCycles;          /* MTE3 写出单元累计忙时             */
    double   aicpuBusyCycles;         /* AI CPU 下发单元累计忙时           */
    double   commBusyCycles;          /* HCCS 通信单元累计忙时             */
    double   stallCycles;             /* 计算单元等待产生的空闲(核·cycle)  */
    double   loadDmaBytes;            /* MTE2 载入累计字节                  */
    double   storeDmaBytes;           /* MTE3 写出累计字节                  */
    double   totalCommBytes;          /* 跨卡通信累计字节                   */
    double   dmaBytes;                /* DMA 搬运总字节(掩盖率分母)         */
    double   maskedBytes;             /* 被计算覆盖的 DMA 字节数            */
    double   sramPeakKB;              /* SRAM 峰值占用 KB(近似)             */
    double   loadBalanceDeviation;    /* 阶段间负载均衡偏差(stddev/mean)    */
    double   cubeUtilizationPct;      /* Cube 利用率 %                      */
    double   vectorUtilizationPct;    /* Vector 利用率 %                    */
    double   mteUtilizationPct;       /* MTE 综合占用(见 sim.c 口径)        */
    double   mte2UtilizationPct;      /* MTE2 利用率 %                      */
    double   mte3UtilizationPct;      /* MTE3 利用率 %                      */
    double   bubblePercent;           /* 气泡占比 %(100-计算占用率)         */
    double   memoryMaskRate;          /* 访存掩盖率 0..1                    */
    double   lowerBoundCycles;        /* 理论下界(对照用)                   */
    double   compositeScore;          /* 综合优度评分 0..100                */
    BottleneckKind bottleneck;        /* 受限类型判定                       */
    double   npuDutyMax;              /* 各卡负载占比最大值                 */
    double   npuDutyMin;              /* 各卡负载占比最小值                 */
    double   npuDutyAvg;              /* 各卡负载占比平均值                 */
    double   serialEquivalentCycles;  /* 串行等效总时间(PP 对照)            */
    double   pipelineFillCycles;      /* PP 流水填充(起停)开销              */
} AlgorithmResult;

/* ============================================================================
 * 顶层入口(run.c 实现)
 * 处理一次完整的 /api/run 请求: 返回 0 且 out 为结果 JSON(ok:1);
 * 失败返回非 0 且 out 为错误 JSON(ok:0, err)。
 * ==========================================================================*/
int runEngineRequest(JsonValue *requestJson, StringBuffer *out);

/* ============================================================================
 * cfg.c: 请求配置解码 / 单位换算
 * ==========================================================================*/
int decodeJobConfig(EngineJob *job, JsonValue *requestJson);  /* 0=成功; 失败置 job->errorMessage */

/* 把“字节数/带宽(字节每秒)/主频”换算为 cycles: T = bytes/(bwPerSec/freq) */
double cyclesForByteTransfer(double bytes, double bandwidthBytesPerSec, double clockGhz);
/* 微秒 -> cycles: us * freq * 1000 */
double cyclesForMicroseconds(double microseconds, double clockGhz);
/* FLOPs 在指定核组上的计算 cycles */
double cyclesForFlops(double flops, double peakTflops, int coresPerNpu, double clockGhz, int groupCores);
/* 卡间拓扑跳数: 支持 ring/linear/mesh/torus/full */
int computeTopologyHops(int npuCount, const char *topology, int fromNpu, int toNpu);

/* ============================================================================
 * work.c: 建模
 * ==========================================================================*/
int  buildWorkloadKernels(EngineJob *job, JsonValue *requestJson); /* 生成 job->kernels; 0=成功 */
void freeJobResources(EngineJob *job);                             /* 释放建模阶段分配的内存   */

/* ============================================================================
 * sim.c: Cycle 级仿真
 * ==========================================================================*/
void simulateAlgorithm(EngineJob *job, const AlgorithmConfig *algorithm, AlgorithmResult *result);
/* 输出指定算法/算卡/核范围的泳道+气泡+采样段+DAG 详情 JSON 片段 */
void simulateDetailLanes(EngineJob *job, const AlgorithmConfig *algorithm, int npuIndex,
                         int timeBinCount, int firstCore, int lastCore,
                         int algorithmIndex, StringBuffer *out);

/* ============================================================================
 * 内核级调试打印开关(定义于 sim.c; 由 COSIM_DBG=1 或 --run 时开启)
 * ==========================================================================*/
extern int g_cosim_dbg;

#endif /* ENGINE_H */
