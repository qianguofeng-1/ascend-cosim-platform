/* ============================================================================
 * sim.c - Cycle 级近似离散仿真引擎
 *
 * 缩写约定(本文件注释通用):
 *   LD = MTE2 载入(Load, 把数据从 HBM/L2 搬到片上 UB);
 *   ST = MTE3 写出(Store, 把结果从片上写回 HBM);
 *   M1 = MTE1 上载(把刚载入的数据从 L1 搬到 L0, 喂给 Cube)。
 * 仿真模型概述:
 *   - 硬件单元: 每算卡有 AI CPU 下发 / MTE2(载入) / MTE3(写出) / HCCS(通信)
 *     四个卡级共享单元 + 每核三个独占单元(MTE1 / Cube / Vector);
 *   - 微任务链: 每个 chunk 经历  AI CPU 下发 -> MTE2 载入(LD)
 *     -> [MTE1 上载(L1->L0)] -> Cube/Vector 计算 -> MTE3 写出(ST);
 *   - 双/多级缓冲: 块(chunk)的载入要等"序号比它早 bufferDepth 的那块"计算完成
 *     (缓冲复用), 形成 搬运-计算 重叠;
 *   - PP 流水: 每卡独立波形链(waves≈microBatch), 端到端
 *     T = (waves-1)*max(D_stage) + ΣD_stage;
 *   - ops 模式: 多卡联合仿真, 跨卡依赖自动插入 HCCS 通信微任务。
 *
 * 【过程逻辑 · 每个角色在干什么(读泳道图前先看这里)】
 *   · AI CPU(ROLE_AI_CPU_ISSUE): 每个 chunk 算前的"指令下发/事件点火"任务,
 *     固定 40 cycles, 通过依赖链与内核级就绪时刻承担"AICPU 事件协调";
 *     共享卡级单队列。下发总时长=40×任务数, 相对整段时间通常很小,
 *     若泳道该行长期满, 才说明 Overhead-bound。
 *   · MTE2(ROLE_MTE2_LOAD): 把"本块输入字节(A/B 数据)"从 HBM 载入片上(UB),
 *     时延 = 搬运折算 cycles(bytes, hbmBandwidthTBs×1e12, 主频) + mte2LatencyCycles;
 *     卡级共享带宽 => 同卡所有载入在一队列上排队。
 *   · MTE1(ROLE_MTE1_MOVE): 每核独占, 把刚载入数据 L1->L0 喂给 Cube, 仅
 *     KERNEL_CUBE/KERNEL_MIXED 的计算前置出现。
 *   · Cube/Vector(ROLE_CUBE_COMPUTE / ROLE_VECTOR_COMPUTE): 真正的计算。
 *     纯 GEMM/Conv(内核 kind=KERNEL_CUBE)只会产生 Cube 任务、不会有 Vector
 *     任务, 所以泳道 Vector 行全空属正常; 只有元素级/混合内核才有 Vector。
 *   · MTE3(ROLE_MTE3_STORE): 把"本块输出字节(C 结果)"写回 HBM, 卡级共享。
 *   · HCCS(ROLE_HCCS_COMM): 仅跨卡依赖时产生(前驱在别的卡), 单卡执行恒为空。
 *   · 时间推进 = 单队列离散事件: start=max(依赖就绪时刻, 所属单元队尾),
 *     依赖边=数据/缓冲就绪事件, 队尾=带宽/资源排队 -> 等价实现 AICPU 事件协调。
 *   · 泳道图 = 上述各单元在时间轴上的忙/闲: 行=单元, 每格=忙碌占比‰;
 *     判定口诀: Cube 行满=Compute-bound; MTE2/3 在 Cube 空档仍忙=搬运没藏住;
 *     AI CPU 行满=下发开销大; 多核行深浅不一=负载不均(详见 simulateDetailLanes)。
 *
 * 【过程逻辑 · 切块与三个旋钮】
 *   切块公式与 ubKb/ubFrac/maxChunk/tileSplit 的完整关系见 simcore.h 的
 *   AlgorithmConfig 上方说明; 本文件 scheduleOneKernel() 是公式的实现点,
 *   块大小是"结果而非输入"(本仿真不建模 M×N×K 几何, 故无显式 tile 尺寸参数)。
 *
 * 【过程逻辑 · 理论 vs 仿真(roofline 口径)】
 *   理论下界 Tlb=FLOPs×主频/(卡峰值) 与 搬运带宽时间=字节/带宽 两本账取大者;
 *   仿真在此基础上额外显式计费: 逐任务时延(lat×任务数)、CPU 下发、缓冲/排队等待,
 *   因此 仿真总周期 = 理论下界 + 流水起停 + 下发/排队损耗(差值即"理论不含量")。
 * ==========================================================================*/
#include "engine.h"

/* ============================================================================
 * **【GEMM_1】sim.c 视角总览(cycle 级调度与掩盖)
 * 对 GEMM_1(1 个 KERNEL_CUBE 内核、0 号卡 8 核、dfs 算法), 仿真实际发生:
 *   1) simulationInit: 建单元表 —— 每卡 28 单元(4 卡级 + 8核×3), 单卡共 28;
 *   2) simulateAlgorithm → runOperatorGraph: 1 内核无依赖 ⇒ 就绪循环只转 1 轮,
 *      直接 scheduleOneKernel(0, floor=0);
 *   3) scheduleOneKernel: P=32 块 × 8 核 = 256 份, 每份 5 个微任务
 *      (AI CPU→MTE2 LD→MTE1→Cube→MTE3 ST) ⇒ 共 1,280 个微任务
 *      (=segsTotal 实测值), chunk-major 交织下发; 双缓冲(dbuf=2)令
 *      第 c 块的载入等第 c-2 块的计算完, 载入被计算掩盖;
 *   4) 端到端 T=431,913(=core7 最后一块 MTE3 写回结束), Cube 满载 99.44%,
 *      MTE2/MTE3 合计 dmaBusy≈104,366, 掩盖率 99.61% —— 典型 Compute-bound;
 *   5) simulateDetailLanes 输出 28 行泳道(4 卡级+8×3)与 80 分箱(bins)。
 * 后续每个函数均给出 GEMM_1 的变量级批注。
 * ==========================================================================*/

/* ============================================================================
 * 微任务(MicroTask): 仿真的最小调度单位
 * ==========================================================================*/
typedef struct {
    MicroTaskRole role;        /* 硬件角色(决定占用哪个单元)          */
    int  npuIndex;             /* 所在算卡                            */
    int  coreIndex;            /* 所在核(-1=卡级共享单元)             */
    int  kernelIndex;          /* 归属内核下标                        */
    int  chunkIndex;           /* 该核内 chunk 序号(-1=通信类任务)    */
    double durationCycles;     /* 执行耗时(cycles)                    */
    double byteCount;          /* DMA/通信字节(计算类任务为 0)        */
    int  depTaskIndexA;        /* 依赖微任务 A(-1=无)                 */
    int  depTaskIndexB;        /* 依赖微任务 B(-1=无)                 */
    double depReadyCycle;      /* 内核级依赖就绪时刻(cycle)下限       */
    long long startCycle;      /* 实际开始 cycle                      */
    long long endCycle;        /* 实际结束 cycle                      */
    int  unitIndex;            /* 占用的硬件单元下标                  */
} MicroTask;

/* 取依赖就绪时刻: max(floorCycle, 依赖A结束, 依赖B结束) */
static long long latestDependencyEnd(const MicroTask *tasks, int depIndexA,
                                     int depIndexB, double floorCycle) {
    long long ready = (long long)floorCycle;
    if (depIndexA >= 0 && tasks[depIndexA].endCycle > ready)
        ready = tasks[depIndexA].endCycle;
    if (depIndexB >= 0 && tasks[depIndexB].endCycle > ready)
        ready = tasks[depIndexB].endCycle;
    return ready;
}

/* ============================================================================
 * 仿真运行上下文(SimulationContext)
 * ==========================================================================*/
typedef struct {
    EngineJob  *job;            /* 请求上下文                       */
    const AlgorithmConfig *algorithm;  /* 本次使用的调度算法       */

    int unitsPerNpu;            /* 每卡硬件单元数(=4+核数*3)       */
    int totalUnits;             /* 全部单元数                       */
    int coresPerNpu;            /* 每卡核数                         */
    int npuCount;               /* 算卡数                           */

    long long *unitTailCycle;   /* 每个单元队列末尾的结束 cycle     */
    MicroTask *tasks;           /* 微任务数组(动态扩容)             */
    int  taskCount;             /* 已创建微任务数                   */
    int  taskCapacity;          /* 微任务数组容量                   */

    /* 硬件参数缓存(避免层层解引用) */
    double coreClockGhz;
    double hbmBandwidthTBs;
    double l0BandwidthTBsPerCore;
    double hccsBandwidthGBs;
    double cubePeakTflops;
    double vectorPeakTflops;

    /* 资源忙时累计(用于利用率统计) */
    double cubeBusyCycles;
    double vectorBusyCycles;
    double mte2BusyCycles;
    double mte3BusyCycles;
    double aicpuBusyCycles;
    double commBusyCycles;
    double stallCycles;         /* 计算单元因等待产生的空闲(核·cycle) */
    double mte2LoadBytes;       /* MTE2 载入累计字节                 */
    double mte3StoreBytes;      /* MTE3 写出累计字节                 */
    double commBytes;           /* HCCS 通信累计字节                 */

    double *coreBusyCycles;     /* 每核忙碌累计: [npu*cores+core]    */
    int    *kernelLastTaskIndex;/* 每个内核最后一个微任务的下标      */
    double  sramPeakKB;         /* SRAM 峰值占用(近似)               */
} SimulationContext;

/* 内核级调试打印开关(server.c / run.c 通过 extern 引用) */
int g_cosim_dbg = 0;

/* ============================================================================
 * 硬件单元编号: 卡级单元(角色<ROLE_MTE1_MOVE)独占每卡 4 个编号;
 * 核级单元(MTE1/Cube/Vector)按 核号*3+角色偏移 排布。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】computeUnitIndex —— 角色→单元编号映射(每卡 28 个单元)
 * 公式分两族(perCardBase = npuIndex × unitsPerNpu, 单卡时 =0):
 *   卡级共享单元(role 0..3, coreIndex 无意义):
 *     perCardBase + role —— AI CPU=0、MTE2=1、MTE3=2、HCCS=3;
 *   核级独占单元(role 4..6 = ROLE_MTE1_MOVE..):
 *     perCardBase + ROLE_MTE1_MOVE + coreIndex×3 + (role-ROLE_MTE1_MOVE)
 *     —— Core c: MTE1=4+3c、Cube=5+3c、Vector=6+3c。
 * GEMM_1(1 卡): core7 Cube → 4+7×3+1=26, core7 Vector → 27 —— 单元 0..27
 * 恰好与"每卡单元数 28"一一对应; 每个单元有独立队尾 unitTailCycle,
 * 卡级单元被 8 个核共享(排队), 核级单元各核私有(并行)。
 * 编号是否重复决定资源是否争用: 两个任务算到同一 unitIndex ⇒ 后者必须等
 * 前者的 unitTailCycle, 这就是"带宽/引擎/核 排队"的离散事件实现。
 * ==========================================================================*/
static int computeUnitIndex(SimulationContext *ctx, int npuIndex,
                            MicroTaskRole role, int coreIndex) {
    int perCardBase = npuIndex * ctx->unitsPerNpu;
    if (role < ROLE_MTE1_MOVE)
        return perCardBase + (int)role;
    return perCardBase + ROLE_MTE1_MOVE + coreIndex * 3 + (int)(role - ROLE_MTE1_MOVE);
}

/* ---- 各类硬件动作的耗时换算(cycles) ----
 * **【GEMM_1】五个耗时助手 —— 口径全部是"带宽账 + 固定 lat"或"纯算力账",
 * 入参都是"每核每块"的量(已除 groupCores×chunkCount), GEMM_1 数值如下:
 *   mte2LoadCycleCost(262,144B) = 262144/(1.2e12/1e9) + 40
 *                              = 218.45+40 ≈ 258.45 cyc —— MTE2 每块载入;
 *   mte3StoreCycleCost(131,072B) = 131072/1200 + 40 ≈ 109.23+40 ≈ 149.23 cyc
 *                              —— MTE3 每块写回;
 *   mte1MoveCycleCost(262,144B)  = 262144/(2.0e12/1e9) + 15 = 131.07+15
 *                              ≈ 146.07 cyc —— MTE1 每核每块 L1→L0;
 *   cubeComputeCycleCost(536,870,912) ≈ 13,421.77 cyc —— Cube 每核每块;
 *   vectorComputeCycleCost(0) = 0 —— GEMM_1 vectorFlops=0, 从不被调用
 *   (只有 KERNEL_VECTOR/MIXED 才走 Vector 链)。
 * 带宽分子带宽单位: hbmBandwidthTBs×1e12 = B/s(TB→B), l0BandwidthTBsPerCore
 * ×1e12 = 每核 B/s; 固定 lat 取 hardware->mte1/2/3LatencyCycles = 15/40/40。 */
static double mte2LoadCycleCost(SimulationContext *ctx, double bytes) {
    return cyclesForByteTransfer(bytes, ctx->hbmBandwidthTBs * 1e12, ctx->coreClockGhz)
           + ctx->job->hardware.mte2LatencyCycles;
}
static double mte3StoreCycleCost(SimulationContext *ctx, double bytes) {
    return cyclesForByteTransfer(bytes, ctx->hbmBandwidthTBs * 1e12, ctx->coreClockGhz)
           + ctx->job->hardware.mte3LatencyCycles;
}
static double mte1MoveCycleCost(SimulationContext *ctx, double bytes) {
    return cyclesForByteTransfer(bytes, ctx->l0BandwidthTBsPerCore * 1e12, ctx->coreClockGhz)
           + ctx->job->hardware.mte1LatencyCycles;
}
/* **【GEMM_1】cube/vector 助手里的 groupCores 恒传 1: 入参 chunkCubeFlops/
 * chunkVectorFlops 在 scheduleOneKernel 里已按 groupCores×chunkCount 均分,
 * 即已是"每核每块"量, 此处不再分摊 ⇒ 有效算力即单核峰值
 * cubePeakTflops×1e12/coresPerNpu = 320e12/8 = 4e13 FLOP/s(1GHz 下 4e4 FLOP/cyc)。
 * 时间账 = chunkFlops ÷ 4e4(FLOP/cyc) = 536,870,912/40,000 ≈ 13,421.77 cyc。 */
static double cubeComputeCycleCost(SimulationContext *ctx, double flops) {
    return cyclesForFlops(flops, ctx->cubePeakTflops, ctx->coresPerNpu,
                          ctx->coreClockGhz, 1);
}
static double vectorComputeCycleCost(SimulationContext *ctx, double flops) {
    return cyclesForFlops(flops, ctx->vectorPeakTflops, ctx->coresPerNpu,
                          ctx->coreClockGhz, 1);
}

/* ============================================================================
 * 创建并调度一个微任务
 * 规则: start = max(依赖就绪时刻, 单元队尾); 同时累计资源忙时/字节/气泡。
 * 返回新任务的数组下标。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】createMicroTask —— 单任务调度语义(离散事件核心)与累加变量
 * 依赖就绪: dependencyReady = max(dependencyFloorCycle, depA.end, depB.end)
 *   —— 内核级就绪下限(dependencyFloorCycle, 跨内核/前驱依赖) 与 两条微任务
 *   依赖边(depTaskIndexA/B: 数据/缓冲就绪事件)取最晚。GEMM_1 例: Cube(c0) 的
 *   依赖A=MTE1(c0)end=446 ⇒ dependencyReady=446;
 *   双缓冲 LD(c) 依赖B=chunkComputeHistory[c-2](Cube 结束) ⇒ c≥2 时受其钳制。
 * 单元队尾: unitQueueTail = unitTailCycle[unitIndex] —— 同单元前一个任务的
 *   结束时刻(资源/带宽排队)。start = max(dependencyReady, unitQueueTail),
 *   end = start + ceil(durationCycles)(end 取整, 忙时账另用未取整值)。
 * 累加变量含义(写进 SimulationContext, 供利用率/掩盖率/瓶颈使用):
 *   · cubeBusyCycles/vectorBusyCycles += durationCycles(未取整的纯忙时):
 *     GEMM_1 Cube 总忙时=256×13,421.77≈3,435,973(utilCube=99.44% 的分子);
 *   · coreBusyCycles[npu*cores+core] += (end-start)(取整跨度): 每核忙时,
 *     泳道/负载统计用; GEMM_1 每核 32×13,422;
 *   · stallCycles += (start - unitQueueTail) 当 start>tail: 计算任务
 *     因依赖未就绪产生的"队外等待"(GEMM_1 合计 10,820, 主要是每轮首块等 MTE1
 *     与双缓冲就绪时刻比队尾晚的部分);
 *   · mte2BusyCycles/mte2LoadBytes: MTE2 忙时与载入字节累加
 *     (合计≈66,163 cyc / 67,108,864B);
 *   · mte3BusyCycles/mte3StoreBytes: 合计≈38,202 cyc / 33,554,432B
 *     (66,163+38,202=104,365≈dmaBusy 实测 104,366);
 *   · aicpuBusyCycles: 40×256=10,240(ctrlBusy);
 *   · commBusyCycles/commBytes: GEMM_1 无跨卡 ⇒ 恒 0;
 *   · unitTailCycle[unit]=end: 把本任务写回队尾供下个任务排队。
 * 任务总量: 每 chunk/核 5 个(CPU,LD,MTE1,Cube,ST) × 32 × 8 = 1,280。
 * ==========================================================================*/
static int createMicroTask(SimulationContext *ctx, MicroTaskRole role,
                           int npuIndex, int coreIndex,
                           int kernelIndex, int chunkIndex,
                           double durationCycles, double byteCount,
                           int depIndexA, int depIndexB, double dependencyFloorCycle) {
    MicroTask *task;
    int unitIndex;
    long long dependencyReady, unitQueueTail, startCycle, endCycle;

    /* 扩容 */
    if (ctx->taskCount == ctx->taskCapacity) {
        int newCapacity = ctx->taskCapacity ? ctx->taskCapacity * 2 : (1 << 16);
        ctx->tasks = (MicroTask *)realloc(ctx->tasks,
                                          (size_t)newCapacity * sizeof(MicroTask));
        ctx->taskCapacity = newCapacity;
    }

    unitIndex = computeUnitIndex(ctx, npuIndex, role, coreIndex);
    dependencyReady = latestDependencyEnd(ctx->tasks, depIndexA, depIndexB,
                                          dependencyFloorCycle);
    unitQueueTail = ctx->unitTailCycle[unitIndex];
    startCycle = dependencyReady > unitQueueTail ? dependencyReady : unitQueueTail;
    endCycle = startCycle + (long long)ceil(durationCycles);

    task = &ctx->tasks[ctx->taskCount];
    task->role = role;
    task->npuIndex = npuIndex;
    task->coreIndex = coreIndex;
    task->kernelIndex = kernelIndex;
    task->chunkIndex = chunkIndex;
    task->durationCycles = durationCycles;
    task->byteCount = byteCount;
    task->depTaskIndexA = depIndexA;
    task->depTaskIndexB = depIndexB;
    task->depReadyCycle = dependencyFloorCycle;
    task->startCycle = startCycle;
    task->endCycle = endCycle;
    task->unitIndex = unitIndex;

    /* 累计资源占用 */
    if (role == ROLE_CUBE_COMPUTE || role == ROLE_VECTOR_COMPUTE) {
        if (role == ROLE_CUBE_COMPUTE)
            ctx->cubeBusyCycles += durationCycles;
        else
            ctx->vectorBusyCycles += durationCycles;
        ctx->coreBusyCycles[npuIndex * ctx->coresPerNpu + coreIndex] +=
            (double)(endCycle - startCycle);
        if (startCycle > unitQueueTail)
            ctx->stallCycles += (double)(startCycle - unitQueueTail);  /* 等待气泡 */
    } else if (role == ROLE_MTE2_LOAD) {
        ctx->mte2BusyCycles += durationCycles;
        ctx->mte2LoadBytes += byteCount;
    } else if (role == ROLE_MTE3_STORE) {
        ctx->mte3BusyCycles += durationCycles;
        ctx->mte3StoreBytes += byteCount;
    } else if (role == ROLE_AI_CPU_ISSUE) {
        ctx->aicpuBusyCycles += durationCycles;
    } else if (role == ROLE_HCCS_COMM) {
        ctx->commBusyCycles += durationCycles;
        ctx->commBytes += byteCount;
    }

    ctx->unitTailCycle[unitIndex] = endCycle;
    return ctx->taskCount++;
}

/* ============================================================================
 * 调度单个内核: 按缓冲深度切成 K-chunk, chunk-major 交织地逐核下发
 * 微任务链(AI CPU 下发 -> MTE2 载入(LD) -> [MTE1 上载(M1) -> Cube/Vector 计算]
 * -> MTE3 写出(ST)), 以形成搬运/计算重叠。
 *
 * 【切块公式(参数语义见 simcore.h AlgorithmConfig 上方说明; 本函数是公式实现点)】
 *   groupCores      = coresPerKernel>0 ? coresPerKernel : 每卡核数(全部核)
 *   ubCapacityBytes = ubSizeKb×1024×ubCapacityFraction   ← 每块"在飞数据"预算(软上限)
 *   每核最大驻留字节 = max(loadBytes/groupCores, storeBytes/groupCores)
 *                     (代码: perCoreInput/perCoreOutput, 取大者存 chunkBytesLimit)
 *   chunkCount      = min(ceil(每核最大驻留字节/ubCapacityBytes),
 *                         maxChunksPerCore, maxTileSplitsPerKernel), 且 >=1
 *                     ← 代码变量 chunkCount 即"最终切块数"
 *   关系: ubCapacityBytes 限"每块能多大"(容量主导: 足够大则理想切块数小),
 *   maxChunk/tileSplit 限"最多切几块"(数量钳制: 理想切块数>maxChunk 时每块将被
 *   强制放大到超过 ubCapacityBytes, 属近似超容, 照跑)。
 *   每核每块: chunkInputBytes  = loadBytes/(groupCores×chunkCount),
 *            chunkOutputBytes = storeBytes/(groupCores×chunkCount),
 *            chunkCubeFlops   = cubeFlops/(groupCores×chunkCount),
 *            chunkVectorFlops = vectorFlops/(groupCores×chunkCount)
 *            块大小是"结果", 不是配置输入。
 *
 * 【为什么没有"显式 tile 大小"设置】
 *   WorkKernel 只携带 总FLOPs/总字节, 不保存 M×N×K 形状与切分方向, 无法表达
 *   "每块 M_k×N_k×K_k"几何; 真机 tile 尺寸亦由 L0/L1/UB 容量 Tiling 推导,
 *   因此本仿真只开放 ubFrac/maxChunk/tileSplit 等"约束", 由容量自动反推块大小。
 *
 * 【微任务链与依赖(单块过程)】
 *   AI CPU 下发(40cyc, 核内 prevCpuTask 链保证顺序)
 *     -> MTE2 载入(LD): 依赖 本块 AI CPU 下发 + "序号比本块早 bufferDepth 的块
 *        计算完成"(缓冲槽释放, 多级缓冲)
 *     -> MTE1 上载(M1): L1->L0 喂数 (仅 CUBE/MIXED 内核)
 *     -> Cube/Vector: 计算(每块一整段, 含该块全部 K 方向累加)
 *     -> MTE3 写出(ST): 若每核每块输出字节 chunkOutputBytes>0.5 才写回
 *   下发顺序 chunk-major(外层块、内层核)是为了避免后核任务整体排在共享
 *   MTE2/CPU 队尾之后造成核间串行。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】scheduleOneKernel 全量走查(把上面的通用公式落到本用例)
 * 输入: kernel=op0-GEMM(KERNEL_CUBE), kernelStartFloorCycle=0(单内核无前驱)。
 * 局部量逐个说明(数值为 GEMM_1 实测推导):
 *   groupCores: coresPerKernel=0 ⇒ =hardware->coresPerNpu=8(随后两组钳制
 *     仅作防御: >8→8、<1→1, 不触发);
 *   ubCapacityBytes = ubSizeKb×1024×ubFrac = 256×1024×0.66 ≈ 173,015B
 *     —— 单块"在飞数据"软预算(KB→B 的 ×1024 在这里发生);
 *   perCoreInput  = loadBytes/groupCores   = 67,108,864/8 = 8,388,608B;
 *   perCoreOutput = storeBytes/groupCores  = 33,554,432/8 = 4,194,304B;
 *   chunkBytesLimit = max(perCoreInput, perCoreOutput) = 8,388,608B —— 每核
 *     最大驻留按输入算(输入>输出, 是"载入要占的 UB"主导);
 *   chunkCount: ①ceil(8,388,608/173,015)=ceil(48.48)=49(容量想切 49 块)
 *    ②>maxChunksPerCore(320)? 否 ⇒ 仍 49
 *    ③>maxTileSplitsPerKernel(tileSplit=32)? 是 ⇒ 49→32(P=32, 第三道钳制命中);
 *   chunkInputBytes  = loadBytes/(groupCores×chunkCount) = 67,108,864/256
 *                    = 262,144B(256KB; 49→32 强切粗 ⇒ 超 ubCapacityBytes 约
 *                    51%, 属"tileSplit 造成的近似超容", 仿真照跑);
 *   chunkOutputBytes = storeBytes/256 = 131,072B(128KB);
 *   chunkCubeFlops   = cubeFlops/256  = 1.3743895e11/256 = 536,870,912;
 *   chunkVectorFlops = 0/256 = 0(kind=CUBE, Vector 永不用);
 *   SRAM 峰值 = (chunkInputBytes×bufferDepth(2)+chunkOutputBytes)/1024
 *             = (524,288+131,072)/1024 = 640KB(实测 sramPeakKB=640);
 *   trackComputeHistory=1(32≤4096 且 8×32=256≤4,000,000) ⇒ 申请
 *     chunkComputeHistory[8×32]=256 槽记录"每核每块 Cube 任务下标"供双缓冲;
 *   prevCpuTask[8] 初值全 -1(每核上一条 CPU 下发任务, 串起核内 CPU 链);
 * 下发总量: 32×8×(CPU+LD+MTE1+Cube+ST=5)=1,280 微任务(=segsTotal)。
 * ==========================================================================*/
static void scheduleOneKernel(SimulationContext *ctx, int kernelIndex,
                              double kernelStartFloorCycle) {
    EngineJob *job = ctx->job;
    WorkKernel *kernel = &job->kernels[kernelIndex];
    const HardwareConfig *hardware = &job->hardware;
    const AlgorithmConfig *algorithm = ctx->algorithm;
    int groupCores = algorithm->coresPerKernel > 0
                         ? algorithm->coresPerKernel : hardware->coresPerNpu;
    double ubCapacityBytes, perCoreInput, perCoreOutput, chunkBytesLimit;
    long chunkCount;
    double chunkInputBytes, chunkOutputBytes, chunkCubeFlops, chunkVectorFlops;
    int coreIndex, chunkIndex;
    int trackComputeHistory;
    int *chunkComputeHistory = NULL;   /* 每核每 chunk 的计算任务下标(双缓冲用) */
    int *prevCpuTask = NULL;           /* 每核上一个 CPU 下发任务下标 */
    int kernelLastTask = -1;
    long long lastEndCycle = -1;

    if (groupCores > hardware->coresPerNpu)
        groupCores = hardware->coresPerNpu;
    if (groupCores < 1)
        groupCores = 1;

    /* ---- 切块: 每核承载的输入/输出字节 超过 UB 可占容量就多切几块 ----
     * **【GEMM_1】逐行核对(局部量含义见函数头):
     * ubCapacityBytes=173,015B(每块软上限, 只作"想切几块"的除法分母);
     * perCoreInput/Output=8,388,608/4,194,304B(每核要驻留的输入/输出);
     * chunkBytesLimit=取大者=输入 8,388,608B(输入主导 UB 预算);
     * chunkCount 三级: ceil(48.48)=49 → 不超 maxChunk(320) → 被 tileSplit(32)
     * 砍到 32 —— 前 1 级是"容量想切几块", 后 2 级是"最多允许切几块"两道钳。 */
    ubCapacityBytes = hardware->ubSizeKb * 1024.0 * algorithm->ubCapacityFraction;
    perCoreInput = kernel->loadBytes / groupCores;
    perCoreOutput = kernel->storeBytes / groupCores;
    chunkBytesLimit = perCoreInput > perCoreOutput ? perCoreInput : perCoreOutput;
    chunkCount = (long)ceil(chunkBytesLimit / ubCapacityBytes);
    if (chunkCount < 1)
        chunkCount = 1;
    if (chunkCount > algorithm->maxChunksPerCore)
        chunkCount = algorithm->maxChunksPerCore;
    if (job->model.maxTileSplitsPerKernel >= 1 &&
        chunkCount > job->model.maxTileSplitsPerKernel)
        chunkCount = job->model.maxTileSplitsPerKernel;   /* 每算子 Tile 切分上限 */

    chunkInputBytes = kernel->loadBytes / ((double)groupCores * chunkCount);
    chunkOutputBytes = kernel->storeBytes / ((double)groupCores * chunkCount);
    chunkCubeFlops = kernel->cubeFlops / ((double)groupCores * chunkCount);
    chunkVectorFlops = kernel->vectorFlops / ((double)groupCores * chunkCount);
    /* **【GEMM_1】上式即"每核每块"的均分结果(分母=8×32=256):
     * chunkInputBytes=67,108,864/256=262,144B; chunkOutputBytes=33,554,432/256
     * =131,072B; chunkCubeFlops=1.3743895e11/256=536,870,912;
     * chunkVectorFlops=0。它们喂给五个耗时助手(sim.c 上节)算单块时间。 */

    /* SRAM 峰值近似: 每核在飞缓冲 ≈ 输入×缓冲深度 + 输出 */
    /* **【GEMM_1】peakKb=(chunkInputBytes×bufferDepth+chunkOutputBytes)/1024
     * =(262,144×2+131,072)/1024=640KB —— 双缓冲意味着每核同时驻留 2 块输入
     * (512KB)+1 块输出(128KB); 该值对全部内核取最大者存 ctx->sramPeakKB,
     * 最终进 scoreboard sramPeakKB=640(实测)。 */
    {
        double peakKb = (chunkInputBytes * algorithm->bufferDepth + chunkOutputBytes) / 1024.0;
        if (peakKb > ctx->sramPeakKB)
            ctx->sramPeakKB = peakKb;
    }

    if (g_cosim_dbg && kernelIndex == 0) {
        fprintf(stderr,
                "DBG ki0 ptr=%p nkers=%d name='%.10s' kind=%d cubeF=%g vecF=%g inB=%g outB=%g g=%d P=%ld sizeK=%d\n",
                (void *)job->kernels, job->kernelCount, kernel->name, kernel->kind,
                kernel->cubeFlops, kernel->vectorFlops, kernel->loadBytes,
                kernel->storeBytes, groupCores, chunkCount, (int)sizeof(WorkKernel));
    }

    /* 双缓冲需要记录“几块之前的计算任务”; 规模过大时退化为无历史 */
    trackComputeHistory =
        (groupCores > 0 && chunkCount >= 1 && chunkCount <= 4096 &&
         (long long)groupCores * chunkCount <= 4000000);
    if (trackComputeHistory) {
        chunkComputeHistory = (int *)malloc(sizeof(int) *
                                            (size_t)(groupCores * chunkCount));
    }
    prevCpuTask = (int *)malloc(sizeof(int) * (size_t)(groupCores > 0 ? groupCores : 1));
    for (coreIndex = 0; coreIndex < groupCores; coreIndex++)
        prevCpuTask[coreIndex] = -1;

    /* chunk-major 交织: 同一轮 chunk 同时为所有核下发任务, 避免共享 MTE2/CPU
       队列中后核任务整体排在前核之后造成的核间串行 */
    /* **【GEMM_1】循环结构: 外层 32 块(chunkIndex 0..31)×内层 8 核(coreIndex
     * 0..7); chunk-major 意味着"第 c 块的 8 个核先于第 c+1 块的核"入队, 使
     * 卡级共享的 AI CPU/MTE2 队列上同块核间紧密相邻 —— 实测 CPU 串行位置
     * (c×8+k)×40(首块 core0 0-40、core7 280-320, 第二块 core0 320-360 …),
     * 首块载入 core0 40-299、core7 2371-2630(258.45/259 cyc, 受各自 CPU 与
     * 卡级 MTE2 队尾共同排队)。
     * chunkFloor: 仅第 0 块取 kernelStartFloorCycle(=0, 前驱就绪时刻),
     * 其余块 floor=0 —— 单内核时块间衔接完全靠任务依赖链与队尾, 不再整体对齐。 */
    for (chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
        for (coreIndex = 0; coreIndex < groupCores; coreIndex++) {
            int npuIndex = kernel->npuIndex;
            double chunkFloor = (chunkIndex == 0) ? kernelStartFloorCycle : 0.0;
            int cpuTask, bufferFreeTask = -1, loadTask, lastTask;

            /* ① AI CPU 下发(核上流水串行) */
            /* **【GEMM_1】CPU 任务: 固定 40.0 cyc(常量, 与 aicpuFreq 无关),
             * 依赖 prevCpuTask[core](本核上一块 CPU, 保证核内下发顺序)与
             * chunkFloor(仅首块)。AI CPU 是卡级共享单元(unit=0) ⇒ 全卡 256 个
             * CPU 任务在一个队列上串行, 合计 40×256=10,240 cyc(=ctrlBusy 实测),
             * 相对 431,913 仅 ~2.4%, 不构成瓶颈。 */
            cpuTask = createMicroTask(ctx, ROLE_AI_CPU_ISSUE, npuIndex, coreIndex,
                                      kernelIndex, chunkIndex, 40.0, 0,
                                      prevCpuTask[coreIndex], -1, chunkFloor);
            prevCpuTask[coreIndex] = cpuTask;

            /* ② MTE2 载入: 依赖 本 chunk 下发 完成 且 缓冲槽空闲
                (前 bufferDepth 个 chunk 的计算已把缓冲腾出) */
            /* **【GEMM_1】双缓冲依赖(bufferDepth=2):
             * chunkIndex≥2 时 bufferFreeTask=chunkComputeHistory[core×32+
             * (chunkIndex-2)](同一核、早 2 块那次的 Cube 任务下标), 于是
             * LD(c) 的依赖B=Cube(c-2) 结束 ⇒ 缓冲槽 c 释放; 对 core0:
             *   LD(c0) 依赖A=CPU(c0)end=40 ⇒ 40-299(258.45→259);
             *   LD(c1) 依赖A=CPU(c1)end=360(无缓冲依赖)⇒ 360-619? 实际被卡级
             *   MTE2 队尾再延;
             *   LD(c2) 依赖B=Cube(c0)end=13,868 ⇒ 最早 13,868 才能载入
             *   (下一块搬运完整藏进前一块的 13,422 cyc 计算窗口 —— 这就是
             *   掩盖率≈99.61% 的直接来源)。
             * 载入耗时 mte2LoadCycleCost(262,144B)≈258.45 cyc/次。 */
            if (chunkIndex >= algorithm->bufferDepth && trackComputeHistory)
                bufferFreeTask =
                    chunkComputeHistory[coreIndex * chunkCount +
                                        (chunkIndex - algorithm->bufferDepth)];
            loadTask = createMicroTask(ctx, ROLE_MTE2_LOAD, npuIndex, coreIndex,
                                       kernelIndex, chunkIndex,
                                       mte2LoadCycleCost(ctx, chunkInputBytes),
                                       chunkInputBytes, cpuTask, bufferFreeTask, 0);
            lastTask = loadTask;

            /* ③ 计算(按内核种类展开 Cube/Vector/混合链) */
            /* **【GEMM_1】kind=KERNEL_CUBE ⇒ 走第一分支(GEMM_1 唯一路径):
             *   LD → MTE1(ROLE_MTE1_MOVE, 每核独占 unit=4+3c, 耗时≈146.07) →
             *   Cube(unit=5+3c, 每块≈13,421.77);
             * 第二分支 KERNEL_VECTOR 与第三分支 KERNEL_MIXED 均不触发 ⇒
             * 没有任何 Vector 任务 ⇒ utilVec=0、Vector 泳道行全 0(正常)。
             * MTE1 与 Cube 都是"每核独占", 8 核同块并行; 单核 32 块 Cube
             * 首尾相接(Cube 单元队尾即上一块结束), 每核连续 429,504 cyc。
             * 每块 Cube 时间 ≈13,421.77(floor 到 end 为 13,422 cyc),
             * 32×8 块合计忙时(未取整)≈3,435,973 → utilCube=99.44%。 */
            if (kernel->kind == KERNEL_CUBE) {
                int mte1Task = createMicroTask(ctx, ROLE_MTE1_MOVE, npuIndex, coreIndex,
                                               kernelIndex, chunkIndex,
                                               mte1MoveCycleCost(ctx, chunkInputBytes),
                                               0, loadTask, -1, 0);
                int cubeTask = createMicroTask(ctx, ROLE_CUBE_COMPUTE, npuIndex, coreIndex,
                                               kernelIndex, chunkIndex,
                                               cubeComputeCycleCost(ctx, chunkCubeFlops),
                                               0, mte1Task, -1, 0);
                if (trackComputeHistory)
                    chunkComputeHistory[coreIndex * chunkCount + chunkIndex] = cubeTask;
                lastTask = cubeTask;
            } else if (kernel->kind == KERNEL_VECTOR) {
                int vectorTask = createMicroTask(ctx, ROLE_VECTOR_COMPUTE, npuIndex,
                                                 coreIndex, kernelIndex, chunkIndex,
                                                 vectorComputeCycleCost(ctx, chunkVectorFlops),
                                                 0, loadTask, -1, 0);
                if (trackComputeHistory)
                    chunkComputeHistory[coreIndex * chunkCount + chunkIndex] = vectorTask;
                lastTask = vectorTask;
            } else if (kernel->kind == KERNEL_MIXED) {
                /* 混合: Cube 计算随后接 Vector 计算 */
                int mte1Task = createMicroTask(ctx, ROLE_MTE1_MOVE, npuIndex, coreIndex,
                                               kernelIndex, chunkIndex,
                                               mte1MoveCycleCost(ctx, chunkInputBytes),
                                               0, loadTask, -1, 0);
                int cubeTask = createMicroTask(ctx, ROLE_CUBE_COMPUTE, npuIndex, coreIndex,
                                               kernelIndex, chunkIndex,
                                               cubeComputeCycleCost(ctx, chunkCubeFlops),
                                               0, mte1Task, -1, 0);
                int vectorTask = createMicroTask(ctx, ROLE_VECTOR_COMPUTE, npuIndex,
                                                 coreIndex, kernelIndex, chunkIndex,
                                                 vectorComputeCycleCost(ctx, chunkVectorFlops),
                                                 0, loadTask, -1, 0);
                if (trackComputeHistory)
                    chunkComputeHistory[coreIndex * chunkCount + chunkIndex] = cubeTask;
                lastTask = cubeTask > vectorTask ? cubeTask : vectorTask;
            }

            /* ④ MTE3 写出(若有输出) */
            /* **【GEMM_1】GEMM 有输出 ⇒ chunkOutputBytes=131,072B>0.5 恒成立,
             * 每块都写回: 耗时 mte3StoreCycleCost≈149.23 cyc(ceil 150),
             * 卡级共享(unit=2) ⇒ 256 次在一条队列串行 ≈38,202 cyc
             * (=utilMte3 8.84% 的分子)。依赖 lastTask=本块 Cube ⇒ 写回不会与
             * 本块计算重叠, 只与后续块计算/搬运重叠。
             * 末块(第 32 块)core7 的 ST 结束时刻即全仿真 endCycle=431,913。
             * 阈值 0.5B 的意义: 输出为 0 的纯载入内核(Vector 类 read-only)
             * 跳过 ST, 不产生写回任务。 */
            if (chunkOutputBytes > 0.5) {
                int storeTask = createMicroTask(ctx, ROLE_MTE3_STORE, npuIndex, coreIndex,
                                                kernelIndex, chunkIndex,
                                                mte3StoreCycleCost(ctx, chunkOutputBytes),
                                                chunkOutputBytes, lastTask, -1, 0);
                lastTask = storeTask;
            }
        }
    }
    free(prevCpuTask);
    if (chunkComputeHistory)
        free(chunkComputeHistory);

    /* 记录内核的最后任务(取结束最晚者; 用于跨内核依赖/波形链) */
    /* **【GEMM_1】向后扫描任务表找"本内核结束最晚的任务": GEMM_1 中即末块
     * (chunk31)core7 的 MTE3 写回任务, end=431,913 —— kernelLastTaskIndex[0]
     * 指向它; 由于 op0-GEMM 无后继内核, 该索引只作为整次仿真的端到端终点
     * (globalMaxEnd=431,913 → result->totalCycles)。多内核时它同时是
     * 依赖边"前驱就绪时刻"的取数点。 */
    {
        int taskScanIndex;
        for (taskScanIndex = ctx->taskCount - 1; taskScanIndex >= 0; taskScanIndex--) {
            if (ctx->tasks[taskScanIndex].kernelIndex == kernelIndex &&
                ctx->tasks[taskScanIndex].endCycle >= lastEndCycle) {
                lastEndCycle = ctx->tasks[taskScanIndex].endCycle;
                kernelLastTask = taskScanIndex;
            }
        }
    }
    ctx->kernelLastTaskIndex[kernelIndex] = kernelLastTask;

    if (g_cosim_dbg && kernelIndex < 9) {
        long long kernelMinStart = -1, kernelMaxEnd = 0;
        int taskScanIndex;
        for (taskScanIndex = ctx->taskCount - 1; taskScanIndex >= 0; taskScanIndex--) {
            if (ctx->tasks[taskScanIndex].kernelIndex == kernelIndex) {
                if (kernelMinStart < 0 ||
                    ctx->tasks[taskScanIndex].startCycle < kernelMinStart)
                    kernelMinStart = ctx->tasks[taskScanIndex].startCycle;
                if (ctx->tasks[taskScanIndex].endCycle > kernelMaxEnd)
                    kernelMaxEnd = ctx->tasks[taskScanIndex].endCycle;
            }
        }
        fprintf(stderr,
                "DBG kern%d %s g=%d P=%ld cbDur=%.0f cbSum=%.0f wall=%lld span=%lld\n",
                kernelIndex, kernel->name, groupCores, chunkCount,
                cubeComputeCycleCost(ctx, chunkCubeFlops),
                chunkCount * cubeComputeCycleCost(ctx, chunkCubeFlops),
                kernelMaxEnd, (kernelMinStart >= 0 ? kernelMaxEnd - kernelMinStart : -1));
    }
}

/* ============================================================================
 * 仿真上下文初始化 / 释放
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】simulationInit —— 单元表/累计器初始化(GEMM_1 数字)
 * 布局公式: unitsPerNpu = ROLE_MTE1_MOVE(4) + coresPerNpu×3 —— 4 个卡级共享
 * 单元(AI CPU/MTE2/MTE3/HCCS, 编号 0..3)+ 每核 3 个独占单元(MTE1/Cube/Vector,
 * 编号 4..27, 共 8×3=24) ⇒ GEMM_1: unitsPerNpu=28, totalUnits=1×28=28。
 * 初始化并清零:
 *   unitTailCycle[28]: 每单元队尾(初 0 ⇒ 首任务无排队); calloc 清零即"空队列";
 *   coreBusyCycles[1×8]: 每核忙时累计(初 0);
 *   kernelLastTaskIndex[1]: 每内核"最后任务"下标, 初 -1(未有任务)。
 * 硬件参数缓存(避免层层解引用): coreClockGhz=1.0, hbmBandwidthTBs=1.2,
 *   l0BandwidthTBsPerCore=2.0, hccsBandwidthGBs=392(GEMM_1 不用), cubePeakTflops
 *   =320, vectorPeakTflops=16 —— 与 decodeHardwareConfig 的解码值一一对应。
 * 忙时累计字段(cubeBusyCycles 等)全部随 memset(ctx) 归零, 供 createMicroTask
 * 逐任务累加。
 * ==========================================================================*/
static int simulationInit(SimulationContext *ctx, EngineJob *job,
                          const AlgorithmConfig *algorithm) {
    int kernelIndex;

    memset(ctx, 0, sizeof(*ctx));
    ctx->job = job;
    ctx->algorithm = algorithm;
    ctx->coresPerNpu = job->hardware.coresPerNpu;
    ctx->npuCount = job->hardware.numNpu;
    ctx->unitsPerNpu = ROLE_MTE1_MOVE + ctx->coresPerNpu * 3;   /* 4 卡级 + 每核3 */
    ctx->totalUnits = ctx->npuCount * ctx->unitsPerNpu;

    ctx->unitTailCycle = (long long *)calloc((size_t)ctx->totalUnits, sizeof(long long));
    ctx->coreBusyCycles = (double *)calloc((size_t)(ctx->npuCount * ctx->coresPerNpu),
                                           sizeof(double));
    ctx->kernelLastTaskIndex =
        (int *)malloc(sizeof(int) * (size_t)(job->kernelCount > 0 ? job->kernelCount : 1));
    for (kernelIndex = 0; kernelIndex < job->kernelCount; kernelIndex++)
        ctx->kernelLastTaskIndex[kernelIndex] = -1;

    ctx->coreClockGhz = job->hardware.coreClockGhz;
    ctx->hbmBandwidthTBs = job->hardware.hbmBandwidthTBs;
    ctx->l0BandwidthTBsPerCore = job->hardware.l0BandwidthTBsPerCore;
    ctx->hccsBandwidthGBs = job->hardware.hccsBandwidthGBs;
    ctx->cubePeakTflops = job->hardware.cubePeakTflops;
    ctx->vectorPeakTflops = job->hardware.vectorPeakTflops;

    logxLog(LOG_LEVEL_DEBUG, "simulationInit: %d卡×%d核 单元/卡=%d",
            ctx->npuCount, ctx->coresPerNpu, ctx->unitsPerNpu);
    return 0;
}

static void simulationFree(SimulationContext *ctx) {
    free(ctx->unitTailCycle);
    free(ctx->coreBusyCycles);
    free(ctx->kernelLastTaskIndex);
    free(ctx->tasks);
    memset(ctx, 0, sizeof(*ctx));
}

/* ============================================================================
 * ops 模式驱动(多卡联合): 按依赖拓扑反复挑选“就绪内核”逐个调度
 * 就绪规则: 所有前驱内核已调度;
 * 挑选策略: issueOrder='c' 关键路径优先(剩余计算量最大者), 否则按序号就绪顺序。
 * 跨卡依赖自动插入 HCCS 通信任务。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】runOperatorGraph —— 单内核场景实际只跑一轮
 * 数据结构: scheduled[1](char 标记是否已调度, calloc=全 0)、remaining=1、
 * stalledGuard=0(死锁保护计数)。
 * while 第一轮扫描: kernel0 的 predecessorCount=0 ⇒ allPredsDone=1(无前驱
 * 即"就绪"); issueOrder="dfs"、首字母非 'c' ⇒ 走"就绪顺序"分支
 * bestKernel=kernelIndex=0(不是关键路径优先)⇒ 立即选中。
 * 依赖就绪计算: 前驱循环不执行 ⇒ dependencyFloor 保持 0 —— scheduleOneKernel
 * (kernel0, floor=0) ⇒ chunk0 的 CPU 从 0 开始;
 * 跨卡分支(前驱 npu≠本内核 npu 才插 HCCS)不触发 —— 单卡无跨卡 ⇒ commBusy=0。
 * 置 scheduled[0]=1、remaining=0 ⇒ while 条件不成立、退出 —— 整段只跑了 1 轮,
 * "就绪内核循环"在无依赖单内核下的行为即如此。
 * ==========================================================================*/
static void runOperatorGraph(SimulationContext *ctx) {
    EngineJob *job = ctx->job;
    int kernelCount = job->kernelCount;
    char *scheduled = (char *)calloc((size_t)(kernelCount > 0 ? kernelCount : 1), 1);
    int remaining = kernelCount;
    int stalledGuard = 0;

    logxTraceEnter(LOG_LEVEL_DEBUG, "runOperatorGraph", "内核数=%d", kernelCount);

    while (remaining > 0) {
        int progressed = 0;
        int bestKernel = -1;
        double bestWork = -1;
        int kernelIndex;

        /* 扫描就绪内核 */
        for (kernelIndex = 0; kernelIndex < kernelCount; kernelIndex++) {
            WorkKernel *candidate;
            int allPredsDone = 1;
            int predecessorIndex;
            if (scheduled[kernelIndex])
                continue;
            candidate = &job->kernels[kernelIndex];
            for (predecessorIndex = 0;
                 predecessorIndex < candidate->predecessorCount;
                 predecessorIndex++) {
                if (!scheduled[candidate->predecessors[predecessorIndex]]) {
                    allPredsDone = 0;
                    break;
                }
            }
            if (!allPredsDone)
                continue;
            if (ctx->algorithm->issueOrder[0] == 'c') {
                /* 关键路径优先: 剩余计算量大的先调度 */
                double work = candidate->cubeFlops + candidate->vectorFlops;
                if (work > bestWork) {
                    bestWork = work;
                    bestKernel = kernelIndex;
                }
            } else {
                bestKernel = kernelIndex;  /* 就绪顺序: 取序号最小者 */
                break;
            }
        }

        if (bestKernel < 0) {
            /* 理论上的死锁(DAG 成环等): 最多容忍 3 轮无进展 */
            if (++stalledGuard > 3)
                break;
            break;
        }
        kernelIndex = bestKernel;
        {
            WorkKernel *kernel = &job->kernels[kernelIndex];
            double dependencyFloor = 0;
            int predecessorIndex;

            /* 计算所有前驱带来的就绪时刻(跨卡时加上 HCCS 通信时间) */
            for (predecessorIndex = 0;
                 predecessorIndex < kernel->predecessorCount;
                 predecessorIndex++) {
                int predKernel = kernel->predecessors[predecessorIndex];
                double predEndCycle = 0;
                int predLastTask = ctx->kernelLastTaskIndex[predKernel];
                if (predLastTask >= 0)
                    predEndCycle = (double)ctx->tasks[predLastTask].endCycle;

                if (job->kernels[predKernel].npuIndex != kernel->npuIndex) {
                    /* 跨卡: 源卡发 HCCS, 时延 = 字节/带宽 + 时延×跳数 */
                    double commBytes = job->kernels[predKernel].storeBytes;
                    double commDuration;
                    int commTask;
                    if (commBytes < 1)
                        commBytes = 1024 * 1024;
                    commDuration =
                        cyclesForByteTransfer(commBytes, ctx->hccsBandwidthGBs * 1e9,
                                              ctx->coreClockGhz)
                        + cyclesForMicroseconds(job->hardware.hccsLatencyUs,
                                                ctx->coreClockGhz)
                          * computeTopologyHops(ctx->npuCount, job->hardware.topology,
                                                job->kernels[predKernel].npuIndex,
                                                kernel->npuIndex);
                    commTask = createMicroTask(ctx, ROLE_HCCS_COMM,
                                               job->kernels[predKernel].npuIndex, -1,
                                               predKernel, -1, commDuration, commBytes,
                                               predLastTask, -1, 0);
                    predEndCycle = (double)ctx->tasks[commTask].endCycle;
                }
                if (predEndCycle > dependencyFloor)
                    dependencyFloor = predEndCycle;
            }
            scheduleOneKernel(ctx, kernelIndex, dependencyFloor);
            scheduled[kernelIndex] = 1;
            progressed = 1;
            remaining--;
        }
        if (!progressed) {
            if (++stalledGuard > 3)
                break;
        } else {
            stalledGuard = 0;
        }
    }
    free(scheduled);

    logxTraceLeave(LOG_LEVEL_DEBUG, "runOperatorGraph", "剩余未调度=%d", remaining);
}

/* ============================================================================
 * 模型(PP 分卡)模式驱动: 每卡按波形数重复执行本阶段内核链,
 * 波形边界向下一级发送激活(HCCS), 得到每阶段的总时长。
 * ==========================================================================*/
static void runModelPipeline(SimulationContext *ctx, double *stageDuration,
                             int pipelineStageCount) {
    EngineJob *job = ctx->job;
    int stage, wave, kernelIndex;

    logxTraceEnter(LOG_LEVEL_DEBUG, "runModelPipeline", "阶段数=%d 波形数=%d",
                   pipelineStageCount, job->microBatchWaves);

    for (stage = 0; stage < pipelineStageCount; stage++) {
        double waveChainEnd = 0;
        int firstKernel = -1, lastKernel = -1;

        /* 该阶段(算卡)上的内核区间 */
        for (kernelIndex = 0; kernelIndex < job->kernelCount; kernelIndex++) {
            if (job->kernels[kernelIndex].npuIndex == stage) {
                if (firstKernel < 0)
                    firstKernel = kernelIndex;
                lastKernel = kernelIndex;
            }
        }
        if (firstKernel < 0) {
            stageDuration[stage] = 0;
            continue;
        }

        for (wave = 0; wave < job->microBatchWaves; wave++) {
            /* 波形内: 顺序调度本阶段全部内核(用上一任务结束时刻串起流水) */
            for (kernelIndex = firstKernel; kernelIndex <= lastKernel; kernelIndex++) {
                int lastTask;
                scheduleOneKernel(ctx, kernelIndex, waveChainEnd);
                lastTask = ctx->kernelLastTaskIndex[kernelIndex];
                if (lastTask >= 0)
                    waveChainEnd = (double)ctx->tasks[lastTask].endCycle;
            }
            /* 波形边界: 向后级发送激活(最后一级除外) */
            if (stage < pipelineStageCount - 1) {
                double bytesPerElement =
                    job->model.bytesPerElement >= 1.5 ? 2.0 : 1.0;
                double activationBytes = job->model.seqLen * job->model.hiddenDim
                                         * bytesPerElement;
                double commDuration =
                    cyclesForByteTransfer(activationBytes, ctx->hccsBandwidthGBs * 1e9,
                                          ctx->coreClockGhz)
                    + cyclesForMicroseconds(job->hardware.hccsLatencyUs,
                                            ctx->coreClockGhz);
                int commTask = createMicroTask(ctx, ROLE_HCCS_COMM, stage, -1,
                                               lastKernel, -1, commDuration,
                                               activationBytes,
                                               ctx->kernelLastTaskIndex[lastKernel],
                                               -1, 0);
                waveChainEnd = (double)ctx->tasks[commTask].endCycle;
            }
        }
        stageDuration[stage] = waveChainEnd;
    }

    logxTraceLeave(LOG_LEVEL_DEBUG, "runModelPipeline", "done");
}

/* ============================================================================
 * 仿真结果汇总: 计算端到端时间/掩盖率/卡间负载偏差
 * ==========================================================================*/
typedef struct {
    long long endCycle;      /* 该卡最后任务结束时刻 */
    double cubeCycles;       /* 该卡 Cube 忙时累计   */
    double vectorCycles;     /* 该卡 Vector 忙时累计 */
    double mte2Cycles;       /* 该卡 MTE2 忙时累计   */
    double mte3Cycles;       /* 该卡 MTE3 忙时累计   */
    double aicpuCycles;      /* 该卡 AI CPU 忙时累计 */
    double commCycles;       /* 该卡 HCCS 忙时累计   */
} NpuTimeSummary;

/* ============================================================================
 * **【GEMM_1】computeAggregateMetrics —— 掩盖率(1024 箱)与逐卡忙时/负载偏差
 * 第一遍(perNpu[NpuTimeSummary]): 逐任务累加每卡 endCycle(该卡最晚结束)与
 * 六类角色忙时(mte2/mte3/aicpu/comm/cube/vector, 只加 durationCycles);
 * GEMM_1 单卡: perNpu[0].endCycle=431,913, cubeCycles≈3,435,973 等;
 * globalMaxEnd=431,913 → ops 模式的 result->totalCycles。
 * 掩盖率算法(1024 时间箱, 思想: 把"计算盖住的搬运比例"≈掩盖率):
 *   ① 每卡画一条长度 1024 的位图 computedBin; 对每个 Cube/Vector 任务, 把
 *      [start×1024/cardEnd, end×1024/cardEnd] 内的箱标记为"计算占用";
 *      —— GEMM_1: Cube 占 99.4% 时间 ⇒ 绝大多数箱为 1;
 *   ② 对每个 MTE2/MTE3 任务, 把其生命周期映射到同一批箱, 统计其中被计算
 *      占用的箱数占比 busyBins/totalBins, 用该占比 × 任务字节 计入
 *      maskedDmaBytes(字节加权, 与时间重叠比例成正比);
 *   ③ result->dmaBytes=Σ任务字节=100,663,296; maskedBytes≈100,270,000 ⇒
 *      memoryMaskRate=masked/dma≈0.9961(≈99.61%, 即搬运几乎全程藏在计算里);
 * 负载偏差段: 仅 model 模式有 stageDuration; ops 传 stageCount=0/stageDuration=
 * NULL ⇒ duration[1] 补零 ⇒ mean=0 ⇒ loadBalanceDeviation=0(单卡无偏差)。
 * 参数 activeCards 在本函数内实际未用((void)activeCards), 仅留接口。
 * ==========================================================================*/
static void computeAggregateMetrics(SimulationContext *ctx, const double *stageDuration,
                                    int stageCount, int activeCards,
                                    AlgorithmResult *result, long long *globalMaxEnd) {
    int taskIndex;
    NpuTimeSummary *perNpu;
    long long maxEnd = 0;

    perNpu = (NpuTimeSummary *)calloc((size_t)ctx->npuCount, sizeof(NpuTimeSummary));

    /* 第一遍: 每卡结束时刻与资源忙时 */
    for (taskIndex = 0; taskIndex < ctx->taskCount; taskIndex++) {
        MicroTask *task = &ctx->tasks[taskIndex];
        NpuTimeSummary *summary = &perNpu[task->npuIndex];
        if (task->endCycle > summary->endCycle)
            summary->endCycle = task->endCycle;
        if (task->endCycle > maxEnd)
            maxEnd = task->endCycle;
        switch (task->role) {
            case ROLE_MTE2_LOAD:      summary->mte2Cycles += task->durationCycles; break;
            case ROLE_MTE3_STORE:     summary->mte3Cycles += task->durationCycles; break;
            case ROLE_AI_CPU_ISSUE:   summary->aicpuCycles += task->durationCycles; break;
            case ROLE_HCCS_COMM:      summary->commCycles += task->durationCycles; break;
            case ROLE_CUBE_COMPUTE:   summary->cubeCycles += task->durationCycles; break;
            case ROLE_VECTOR_COMPUTE: summary->vectorCycles += task->durationCycles; break;
        }
    }
    *globalMaxEnd = maxEnd;

    /* 访存掩盖率: 按 1024 时间箱, 看 DMA(载入/写出)被计算覆盖的比例 */
    {
        const int binCount = 1024;
        unsigned char *computedBin = (unsigned char *)calloc(
            (size_t)(ctx->npuCount * binCount), 1);
        double totalDmaBytes = 0, maskedDmaBytes = 0;

        /* 先标记计算占用的时间箱 */
        for (taskIndex = 0; taskIndex < ctx->taskCount; taskIndex++) {
            MicroTask *task = &ctx->tasks[taskIndex];
            if (task->role == ROLE_CUBE_COMPUTE || task->role == ROLE_VECTOR_COMPUTE) {
                long long cardEnd = perNpu[task->npuIndex].endCycle > 0
                                        ? perNpu[task->npuIndex].endCycle : 1;
                long long binFrom = (task->startCycle * binCount) / cardEnd;
                long long binTo = (task->endCycle * binCount) / cardEnd;
                long long binIndex;
                if (binFrom < 0) binFrom = 0;
                if (binTo > binCount - 1) binTo = binCount - 1;
                for (binIndex = binFrom; binIndex <= binTo; binIndex++)
                    computedBin[task->npuIndex * binCount + binIndex] = 1;
            }
        }
        /* DMA 任务按覆盖比例计入被掩盖字节 */
        for (taskIndex = 0; taskIndex < ctx->taskCount; taskIndex++) {
            MicroTask *task = &ctx->tasks[taskIndex];
            if (task->role == ROLE_MTE2_LOAD || task->role == ROLE_MTE3_STORE) {
                double bytes = task->byteCount;
                long long cardEnd, binFrom, binTo, busyBins = 0, totalBins, binIndex;
                if (bytes <= 0)
                    continue;
                cardEnd = perNpu[task->npuIndex].endCycle > 0
                              ? perNpu[task->npuIndex].endCycle : 1;
                totalDmaBytes += bytes;
                binFrom = (task->startCycle * binCount) / cardEnd;
                binTo = (task->endCycle * binCount) / cardEnd;
                if (binFrom < 0) binFrom = 0;
                if (binTo > binCount - 1) binTo = binCount - 1;
                totalBins = binTo - binFrom + 1;
                if (totalBins < 1)
                    totalBins = 1;
                for (binIndex = binFrom; binIndex <= binTo; binIndex++) {
                    if (computedBin[task->npuIndex * binCount + binIndex])
                        busyBins++;
                }
                maskedDmaBytes += bytes * ((double)busyBins / totalBins);
            }
        }
        result->dmaBytes = totalDmaBytes;
        result->maskedBytes = maskedDmaBytes;
        free(computedBin);
    }

    /* 卡间/阶段负载偏差(相对标准差) */
    {
        double *duration = (double *)calloc((size_t)(stageCount > 0 ? stageCount : 1),
                                            sizeof(double));
        double mean = 0, variance = 0, stdDev = 0;
        int stageIndex;
        for (stageIndex = 0; stageIndex < stageCount; stageIndex++) {
            duration[stageIndex] = (stageDuration && stageDuration[stageIndex] > 0)
                                       ? stageDuration[stageIndex] : 0;
        }
        for (stageIndex = 0; stageIndex < stageCount; stageIndex++)
            mean += duration[stageIndex];
        if (stageCount > 0)
            mean /= stageCount;
        for (stageIndex = 0; stageIndex < stageCount; stageIndex++) {
            double delta = duration[stageIndex] - mean;
            variance += delta * delta;
        }
        if (stageCount > 1)
            variance /= stageCount;
        else
            variance = 0;
        stdDev = sqrt(variance);
        result->loadBalanceDeviation = mean > 1 ? stdDev / mean : 0;
        free(duration);
    }

    (void)activeCards;
    free(perNpu);
}

/* ============================================================================
 * **【GEMM_1】simulateAlgorithm —— ops 单算法仿真走法与指标口径
 * 准备: result 清零并拷 algoKey/algoName; simulationInit 建 28 单元;
 * activeCardCount: 扫内核的 npuIndex 收集用到的卡 ⇒ GEMM_1 只见到 0 号卡 ⇒ 1。
 * 模式分支: STIMULUS_OPERATORS ⇒ stageCount=1、runOperatorGraph(1 轮)、
 * serialEquivalentCycles/pipelineFillCycles=0、duty 全 1.0;
 * computeAggregateMetrics(..., NULL, 0, ...) 后 ops 用 globalMaxEnd=431,913
 * 覆写 result->totalCycles(不再用阶段公式)。
 * 利用率(分母 utilizationDenominator = activeCardCount×coresPerNpu×T
 * = 1×8×431,913 = 3,455,304 核·cycle):
 *   cubeUtilizationPct = 100×3,435,973/3,455,304 ≈ 99.44(命中理论 99.44%);
 *   vectorUtilizationPct = 0/分母 = 0;
 *   mteUtilizationPct   = 100×(mte2+mte3)/(2×activeCards×T)
 *                       = 100×104,366/863,826 ≈ 12.08(2 是"两台引擎"归一);
 *   mte2UtilizationPct  = 100×66,163/(1×431,913) ≈ 15.32;
 *   mte3UtilizationPct  = 100×38,202/431,913   ≈ 8.84;
 * bubblePercent = 100×(1-(cube+vec)/分母) = 100×(1-0.994406) ≈ 0.559;
 * memoryMaskRate 由 computeAggregateMetrics 填 ≈0.9961;
 * sramPeakKB=640(来自 scheduleOneKernel 的峰值段);
 * lowerBoundCycles=job->lowerBoundCycles=429,497(建模期 Tlb)。
 * 评分 score=100×(0.35·efficiency+0.20·(1-bubble)+0.30·mask+0.15·(1-lb)):
 *   efficiency=Tlb/T=429,497/431,913≈0.9944(截断 ≤1);
 *   bubbleFraction≈0.00559; maskRatio≈0.9961; loadBalance=0;
 *   ⇒ 100×(0.34804+0.19888+0.29883+0.15)≈99.575(实测 score=99.5751)。
 * 瓶颈判定见 engine.h【GEMM_1】块 ⇒ BOUND_COMPUTE(0.5 阈值语义同前)。
 * ==========================================================================*/

/* ============================================================================
 * 单算法仿真入口(simulateAlgorithm, engine.h 导出)
 * 运行完整仿真后聚合出 AlgorithmResult(评分/瓶颈/利用率等)。
 * ==========================================================================*/
void simulateAlgorithm(EngineJob *job, const AlgorithmConfig *algorithm,
                       AlgorithmResult *result) {
    SimulationContext ctx;
    double stageDuration[256], perWaveDuration[256];
    int stageCount = 0;
    double maxPerWaveDuration = 0, sumPerWaveDuration = 0;
    long long globalMaxEnd = 0;
    int activeCardCount = 0;
    int kernelIndex, stageIndex;

    memset(result, 0, sizeof(*result));
    strncpy(result->algoKey, algorithm->algoKey, 23);
    result->algoKey[23] = '\0';
    strncpy(result->algoName, algorithm->algoName, 79);
    result->algoName[79] = '\0';

    logxTraceEnter(LOG_LEVEL_INFO, "simulateAlgorithm", "算法=%s",
                   algorithm->algoName);

    simulationInit(&ctx, job, algorithm);

    /* 统计实际用到的算卡数 */
    {
        char cardSeen[256];
        memset(cardSeen, 0, sizeof(cardSeen));
        for (kernelIndex = 0; kernelIndex < job->kernelCount; kernelIndex++) {
            int npuIndex = job->kernels[kernelIndex].npuIndex;
            if (npuIndex < 0)
                npuIndex = 0;
            if (npuIndex < 64 && !cardSeen[npuIndex]) {
                cardSeen[npuIndex] = 1;
                activeCardCount++;
            }
        }
        if (activeCardCount < 1)
            activeCardCount = 1;
    }

    if (job->stimulus.mode == STIMULUS_MODEL) {
        /* ---- 模型模式: PP 多卡流水 + 多波形 ---- */
        int pipelineStageCount = job->pipelineActive
                                     ? (job->model.pipelineParallel > job->hardware.numNpu
                                            ? job->hardware.numNpu
                                            : job->model.pipelineParallel)
                                     : 1;
        int waveCount = job->microBatchWaves;
        stageCount = pipelineStageCount;
        runModelPipeline(&ctx, stageDuration, pipelineStageCount);

        if (waveCount < 1)
            waveCount = 1;
        for (stageIndex = 0; stageIndex < pipelineStageCount; stageIndex++) {
            perWaveDuration[stageIndex] = stageDuration[stageIndex] / waveCount;
            if (perWaveDuration[stageIndex] > maxPerWaveDuration)
                maxPerWaveDuration = perWaveDuration[stageIndex];
            sumPerWaveDuration += perWaveDuration[stageIndex];
        }
        /* 端到端: 流水稳态公式 T=(waves-1)*max + Σ */
        result->totalCycles = (waveCount - 1) * maxPerWaveDuration + sumPerWaveDuration;
        result->serialEquivalentCycles = sumPerWaveDuration * waveCount;
        result->pipelineFillCycles = result->totalCycles - maxPerWaveDuration;

        /* 各卡负载占比(按每阶段完整波形链时长) */
        for (stageIndex = 0; stageIndex < pipelineStageCount; stageIndex++) {
            double duty = result->totalCycles > 0
                              ? stageDuration[stageIndex] / result->totalCycles : 0;
            if (stageIndex == 0) {
                result->npuDutyMax = duty;
                result->npuDutyMin = duty;
            } else {
                if (duty > result->npuDutyMax)
                    result->npuDutyMax = duty;
                if (duty < result->npuDutyMin)
                    result->npuDutyMin = duty;
            }
            result->npuDutyAvg += duty;
        }
        if (pipelineStageCount > 0)
            result->npuDutyAvg /= pipelineStageCount;
        for (stageIndex = 0; stageIndex < stageCount; stageIndex++)
            stageDuration[stageIndex] = perWaveDuration[stageIndex];
        /* 平衡统计用单波形时长 */
    } else {
        /* ---- ops 模式: 多卡联合, 端到端=全局最后任务结束 ---- */
        stageCount = activeCardCount;
        runOperatorGraph(&ctx);
        result->serialEquivalentCycles = 0;
        result->pipelineFillCycles = 0;
        result->npuDutyMax = result->npuDutyMin = result->npuDutyAvg = 1.0;
    }

    computeAggregateMetrics(&ctx,
                            job->stimulus.mode == STIMULUS_MODEL ? stageDuration : NULL,
                            job->stimulus.mode == STIMULUS_MODEL ? stageCount : 0,
                            activeCardCount, result, &globalMaxEnd);

    if (job->stimulus.mode != STIMULUS_MODEL)
        result->totalCycles = (double)globalMaxEnd;

    /* ---- 利用率 / 气泡 / 掩盖率 / 评分 / 瓶颈 ---- */
    {
        double totalCycles = result->totalCycles;
        double utilizationDenominator;
        double efficiency, bubbleFraction, maskRatio, loadBalance;
        double cubeFrac, memFrac, controlFrac, stallFrac;

        if (totalCycles < 1)
            totalCycles = 1;
        utilizationDenominator = (double)activeCardCount * job->hardware.coresPerNpu
                                 * totalCycles;

        result->cubeBusyCycles = ctx.cubeBusyCycles;
        result->vectorBusyCycles = ctx.vectorBusyCycles;
        result->mte2BusyCycles = ctx.mte2BusyCycles;
        result->mte3BusyCycles = ctx.mte3BusyCycles;
        result->aicpuBusyCycles = ctx.aicpuBusyCycles;
        result->commBusyCycles = ctx.commBusyCycles;
        result->stallCycles = ctx.stallCycles;

        result->cubeUtilizationPct = utilizationDenominator > 0
            ? 100.0 * ctx.cubeBusyCycles / utilizationDenominator : 0;
        result->vectorUtilizationPct = utilizationDenominator > 0
            ? 100.0 * ctx.vectorBusyCycles / utilizationDenominator : 0;
        result->mteUtilizationPct = utilizationDenominator > 0
            ? 100.0 * (ctx.mte2BusyCycles + ctx.mte3BusyCycles)
              / (2.0 * activeCardCount * totalCycles) : 0;
        result->mte2UtilizationPct = utilizationDenominator > 0
            ? 100.0 * ctx.mte2BusyCycles / (activeCardCount * totalCycles) : 0;
        result->mte3UtilizationPct = utilizationDenominator > 0
            ? 100.0 * ctx.mte3BusyCycles / (activeCardCount * totalCycles) : 0;

        result->loadDmaBytes = ctx.mte2LoadBytes;
        result->storeDmaBytes = ctx.mte3StoreBytes;
        result->totalCommBytes = ctx.commBytes;

        result->bubblePercent = utilizationDenominator > 0
            ? 100.0 * (1.0 - (ctx.cubeBusyCycles + ctx.vectorBusyCycles)
                       / utilizationDenominator) : 100.0;
        result->memoryMaskRate = result->dmaBytes > 0
            ? result->maskedBytes / result->dmaBytes : 0;
        result->sramPeakKB = ctx.sramPeakKB;
        result->lowerBoundCycles = job->lowerBoundCycles > 0
            ? job->lowerBoundCycles : totalCycles;

        /* 综合优度评分: 效率35% + 气泡20% + 掩盖率30% + 均衡15% */
        efficiency = result->lowerBoundCycles / totalCycles;
        if (efficiency > 1) efficiency = 1;
        if (efficiency < 0) efficiency = 0;
        loadBalance = result->loadBalanceDeviation;
        if (loadBalance < 0) loadBalance = 0;
        if (loadBalance > 1) loadBalance = 1;
        bubbleFraction = result->bubblePercent / 100.0;
        if (bubbleFraction > 1) bubbleFraction = 1;
        maskRatio = result->memoryMaskRate;
        if (maskRatio > 1) maskRatio = 1;
        result->compositeScore = 100.0 * (0.35 * efficiency
                                          + 0.20 * (1.0 - bubbleFraction)
                                          + 0.30 * maskRatio
                                          + 0.15 * (1.0 - loadBalance));
        if (result->compositeScore > 100)
            result->compositeScore = 100;
        if (result->compositeScore < 0)
            result->compositeScore = 0;

        /* 瓶颈判定: 控制/开销 -> 计算 -> 访存 */
        cubeFrac = ctx.cubeBusyCycles / utilizationDenominator;
        memFrac = (ctx.mte2BusyCycles + ctx.mte3BusyCycles)
                  / (2.0 * activeCardCount * totalCycles);
        controlFrac = ctx.aicpuBusyCycles / (activeCardCount * totalCycles);
        stallFrac = ctx.stallCycles / utilizationDenominator;
        if (controlFrac > 0.5 && controlFrac > cubeFrac && controlFrac > memFrac)
            result->bottleneck = BOUND_OVERHEAD;
        else if (stallFrac > 0.5 && stallFrac > cubeFrac && stallFrac > memFrac)
            result->bottleneck = BOUND_OVERHEAD;
        else if (cubeFrac >= memFrac)
            result->bottleneck = BOUND_COMPUTE;
        else
            result->bottleneck = BOUND_MEMORY;
    }

    logxLog(LOG_LEVEL_INFO,
            "算法「%s」仿真完成: 总Cycles=%.0f 评分=%.1f 气泡=%.1f%% 掩盖率=%.1f%%",
            algorithm->algoName, result->totalCycles, result->compositeScore,
            result->bubblePercent, result->memoryMaskRate * 100.0);
    logxTraceLeave(LOG_LEVEL_INFO, "simulateAlgorithm", "总Cycles=%.0f",
                   result->totalCycles);

    simulationFree(&ctx);
}

/* ============================================================================
 * **【GEMM_1】泳道输出几何(G EMM_1 view={core0:0,core1:7,bins:80})
 * 行集合 = 4 卡级行(AI CPU/MTE2 入/MTE3 出/HCCS)+ 每核 3 行(MTE1/Cube/Vector)
 * × 8 核 = 28 行(实测 rows=28); GEMM_1 各行的典型形态:
 *   AI CPU 行: 只在开头一段 1000(下发集中在前 10,240 cyc);
 *   MTE2/MTE3 行: 稀疏 1000 条(载入/写回 ~259/150 cyc 的短任务散布全时间轴,
 *     被 Cube 行淹没 ⇒ 掩盖率高);
 *   HCCS 行: 全 0(单卡无通信, 正常);
 *   Core* MTE1: 每块前 ~147 cyc 短忙条;
 *   Core* Cube: 几乎连续满格 1000(32 块首尾相接)⇒ Compute-bound 的图面证据;
 *   Core* Vector: 全 0(纯 KERNEL_CUBE, 正常)。
 * 分箱: bins=80 ⇒ 每格宽 431,913/80≈5,399 cyc; 格值=忙碌占比‰。
 * 气泡: 8 条(每核首块 Cube 等 MTE1 的 0→446/705/…/2259 缺口), 均标注
 *   "等待 L1->L0 (MTE1) 上载完成"(s=缺口起点,e=缺口终点,w=位置,y=原因);
 * segs: 1,280 个任务采样(segsTotal=1280), 每任务 {s,e,r,c,k}(起/止/角色/
 * 核/内核); dag: total=1, 节点 op0-GEMM kind=Cube deps=[](无依赖)。
 * ==========================================================================*/

/* ============================================================================
 * 泳道详情输出(仅单个算卡的视图): 行值=忙碌占比‰
 *
 * 【泳道图怎么读(结果页 5.2 视图)】
 *   横轴=时间(目标卡总周期 T 等分为 timeBinCount 格); 纵轴=一行一个硬件单元:
 *     行顺序 = AI CPU / MTE2入 / MTE3出 / HCCS + 每核(MTE1, Cube, Vector) 各一行,
 *     因此总行数 = 4 + 核数×3(如 32 核共 100 行)。
 *   每格数值 0..1000 = 该时间片该单元的忙碌占比‰(任意任务覆盖即置满 1000)。
 *   阅读判定:
 *     · Cube 行满格(≈1000 连续) => Compute-bound, 计算是瓶颈;
 *     · Cube 行大段空白而 MTE2/MTE3 仍忙 => 搬运未与计算重叠(memory-bound/重叠差);
 *     · AI CPU 行长期忙 => 指令下发开销大(overhead-bound), 一般只在开头一小片;
 *     · 各核 Cube 行深浅长短不一 => 核间负载不均;
 *     · Vector 行全空: 纯 KERNEL_CUBE(GEMM/Conv) 无向量任务, 属正常;
 *     · HCCS 行全空: 单卡无跨卡通信, 属正常。
 *   空白大段会附带"气泡标注"(s/e=起止, w=位置, y=原因), 原因与依赖一一对应:
 *     等待 MTE2 搬运/MTE1 上载/前序结果回写/AI CPU 下发/跨卡 HCCS/双缓冲未释放。
 * ==========================================================================*/

/* 输出一行泳道数据: label 行名, role+core 决定取哪些任务, binCount 为分箱数 */
/* **【GEMM_1】一行=一个硬件单元: emitSwimlaneRow 只取 npuIndex 命中且
 * role==参数 role(核级行再要求 coreIndex 相等)的任务; GEMM_1 中每行取到
 * 8~64 个任务不等(CPU 行 256、MTE1/Cube 行 256…)。binFrom/To 用
 * start×binCount/totalCycles 把任务的 [start,end] 投到 80 个格子上, 每格
 * 累计 +1000(上限 1000)⇒ 格值=该格内忙碌占比‰(满格 1000=整格被任务覆盖,
 * 如 Cube 行的连续 1000; 半格则按重叠比例折减——本实现凡有覆盖即加满,
 * 属近似)。输出 {l:行名, t:role 号(前端按此上色), v:[…80 个格值]}。 */
static void emitSwimlaneRow(StringBuffer *out, int isFirstRow, const char *rowLabel,
                            const MicroTask *tasks, int taskCount,
                            int npuIndex, MicroTaskRole role, int coreIndex,
                            int binCount, long long totalCycles) {
    int *binValue = (int *)calloc((size_t)binCount, sizeof(int));
    int taskIndex, binColumn;

    for (taskIndex = 0; taskIndex < taskCount; taskIndex++) {
        const MicroTask *task = &tasks[taskIndex];
        long long binFrom, binTo;
        int binCursor;
        if (task->npuIndex != npuIndex || task->role != role)
            continue;
        if (coreIndex >= 0 && task->coreIndex != coreIndex)
            continue;
        binFrom = (task->startCycle * binCount) / totalCycles;
        binTo = (task->endCycle * binCount) / totalCycles;
        if (binFrom < 0) binFrom = 0;
        if (binTo >= binCount) binTo = binCount - 1;
        for (binCursor = (int)binFrom; binCursor <= (int)binTo; binCursor++) {
            double occupancy = 1.0;
            if (binValue[binCursor] < 1000)
                binValue[binCursor] += (int)(occupancy * 1000.0);
            if (binValue[binCursor] > 1000)
                binValue[binCursor] = 1000;
        }
    }

    if (!isFirstRow)
        strbufAppendChar(out, ',');
    strbufAppend(out, "{\"l\":");
    strbufAppendEscapedString(out, rowLabel);
    strbufAppendFormat(out, ",\"t\":%d,\"v\":[", (int)role);
    for (binColumn = 0; binColumn < binCount; binColumn++) {
        if (binColumn)
            strbufAppendChar(out, ',');
        strbufAppendFormat(out, "%d", binValue[binColumn]);
    }
    strbufAppend(out, "]}");
    free(binValue);
}

/* 输出一个气泡标注段(空缺起止 + 位置/原因) */
static void emitBubbleMark(StringBuffer *out, int isFirstBubble,
                           long long gapStart, long long gapEnd,
                           const char *where, const char *why) {
    if (!isFirstBubble)
        strbufAppendChar(out, ',');
    strbufAppendFormat(out, "{\"s\":%lld,\"e\":%lld,\"w\":", gapStart, gapEnd);
    strbufAppendEscapedString(out, where);
    strbufAppend(out, ",\"y\":");
    strbufAppendEscapedString(out, why);
    strbufAppendChar(out, '}');
}

/* 采样任务段(泳道 SVG 用) */
typedef struct {
    long long startCycle;   /* 开始 cycle        */
    long long endCycle;     /* 结束 cycle        */
    int  role;              /* 微任务角色        */
    int  coreIndex;         /* 所在核            */
    int  kernelIndex;       /* 所属内核          */
} SegmentSample;

/* 按开始时间排序采样段 */
static int compareSegmentByStart(const void *leftItem, const void *rightItem) {
    const SegmentSample *leftSegment = (const SegmentSample *)leftItem;
    const SegmentSample *rightSegment = (const SegmentSample *)rightItem;
    return leftSegment->startCycle > rightSegment->startCycle ? 1
           : (leftSegment->startCycle < rightSegment->startCycle ? -1 : 0);
}

/* ============================================================================
 * 泳道详情入口(simulateDetailLanes, engine.h 导出)
 * 输出 JSON: {algo,npu,T,rows[],bubbles[],rowsNote,segs[],segsTotal,dag{...}}
 * ==========================================================================*/
void simulateDetailLanes(EngineJob *job, const AlgorithmConfig *algorithm,
                         int npuIndex, int timeBinCount,
                         int firstCore, int lastCore, int algorithmIndex,
                         StringBuffer *out) {
    SimulationContext ctx;
    double stageDuration[256];
    long long npuEndCycle = 0;
    long long totalCycles;
    int taskIndex, coreNumber;

    logxTraceEnter(LOG_LEVEL_INFO, "simulateDetailLanes",
                   "算法=%s 卡=%d 核[%d..%d] 分箱=%d",
                   algorithm->algoName, npuIndex, firstCore, lastCore, timeBinCount);

    simulationInit(&ctx, job, algorithm);

    if (job->stimulus.mode == STIMULUS_MODEL) {
        int pipelineStageCount = job->pipelineActive
                                     ? (job->model.pipelineParallel > job->hardware.numNpu
                                            ? job->hardware.numNpu
                                            : job->model.pipelineParallel)
                                     : 1;
        runModelPipeline(&ctx, stageDuration, pipelineStageCount);
    } else {
        runOperatorGraph(&ctx);
    }

    /* 目标卡的结束时刻作为时间轴总长 */
    /* **【GEMM_1】npuEndCycle=0 号卡最后任务结束=431,913(与 simulateAlgorithm
     * 的 globalMaxEnd 同值); totalCycles=npuEndCycle 作为泳道时间轴全长 T,
     * 80 个分箱(=view.bins) ⇒ 每格≈5,399 cyc; T 同时是 emitSwimlaneRow 的
     * 归一化分母(把任务 start/end 投到 0..79 格)与气泡 minGap 的基准。 */
    for (taskIndex = 0; taskIndex < ctx.taskCount; taskIndex++) {
        if (ctx.tasks[taskIndex].npuIndex == npuIndex &&
            ctx.tasks[taskIndex].endCycle > npuEndCycle)
            npuEndCycle = ctx.tasks[taskIndex].endCycle;
    }
    if (npuEndCycle < 1)
        npuEndCycle = 1;
    totalCycles = npuEndCycle;

    strbufAppendFormat(out, "{\"algo\":%d,\"npu\":%d,\"T\":%lld,\"rows\":[",
                       algorithmIndex, npuIndex, totalCycles);

    /* 卡级共享单元行 */
    emitSwimlaneRow(out, 1, "AI CPU",  ctx.tasks, ctx.taskCount, npuIndex,
                    ROLE_AI_CPU_ISSUE, -1, timeBinCount, totalCycles);
    emitSwimlaneRow(out, 0, "MTE2 入", ctx.tasks, ctx.taskCount, npuIndex,
                    ROLE_MTE2_LOAD, -1, timeBinCount, totalCycles);
    emitSwimlaneRow(out, 0, "MTE3 出", ctx.tasks, ctx.taskCount, npuIndex,
                    ROLE_MTE3_STORE, -1, timeBinCount, totalCycles);
    emitSwimlaneRow(out, 0, "HCCS",    ctx.tasks, ctx.taskCount, npuIndex,
                    ROLE_HCCS_COMM, -1, timeBinCount, totalCycles);

    /* 核级单元行(每个核的 MTE1/Cube/Vector) */
    for (coreNumber = firstCore;
         coreNumber <= lastCore && coreNumber < ctx.coresPerNpu;
         coreNumber++) {
        char rowLabel[64];
        snprintf(rowLabel, sizeof(rowLabel), "Core%d MTE1", coreNumber);
        emitSwimlaneRow(out, 0, rowLabel, ctx.tasks, ctx.taskCount, npuIndex,
                        ROLE_MTE1_MOVE, coreNumber, timeBinCount, totalCycles);
        snprintf(rowLabel, sizeof(rowLabel), "Core%d Cube", coreNumber);
        emitSwimlaneRow(out, 0, rowLabel, ctx.tasks, ctx.taskCount, npuIndex,
                        ROLE_CUBE_COMPUTE, coreNumber, timeBinCount, totalCycles);
        snprintf(rowLabel, sizeof(rowLabel), "Core%d Vector", coreNumber);
        emitSwimlaneRow(out, 0, rowLabel, ctx.tasks, ctx.taskCount, npuIndex,
                        ROLE_VECTOR_COMPUTE, coreNumber, timeBinCount, totalCycles);
    }

    /* ---- 气泡标注: 计算单元两次相邻任务间的大缺口(>minGap) ---- */
    /* **【GEMM_1】气泡判定与语义: 只扫描本卡、role=Cube/Vector、核在
     * [firstCore,lastCore] 的计算任务; unitLastEnd[unitLocal] 记该计算单元
     * 上一任务的结束时刻(按核分桶), 若 task.start - gapStart ≥ minGap
     * (=max(T/2000,200)=max(215.9,200)≈215) 就记一个气泡, 输出字段:
     *   s=气泡起点(上一计算任务结束)、e=气泡终点(本任务开始)、
     *   w=where(哪个单元/哪核/哪内核哪块)、y=why(按依赖任务类型归因):
     *   depRole=MTE1_MOVE ⇒ "等待 L1->L0 (MTE1) 上载完成"; 其余分支
     *   (MTE2 搬运/HCCS/CPU/双缓冲 Cube)在 GEMM_1 也都有对应文案。
     * GEMM_1 实测 bubbles=8: 每核首块 Cube 的 0→446/…/2259 段(等首块
     * MTE1 上载), 此后双缓冲让搬运藏进计算, 不再产生 ≥215 cyc 的大缺口。
     * 上限 80 条防膨胀。 */
    strbufAppend(out, "],\"bubbles\":[");
    {
        int unitsPerNpu = ctx.unitsPerNpu;
        long long *unitLastEnd = (long long *)calloc((size_t)unitsPerNpu,
                                                     sizeof(long long));
        int bubbleCount = 0;
        long long minGap = totalCycles / 2000;

        if (minGap < 200)
            minGap = 200;
        for (taskIndex = 0; taskIndex < ctx.taskCount; taskIndex++) {
            MicroTask *task = &ctx.tasks[taskIndex];
            long long gapStart;
            int unitLocal, depRole = -1;
            long long bestDepEnd = -1;

            if (task->npuIndex != npuIndex)
                continue;
            if (task->role != ROLE_CUBE_COMPUTE && task->role != ROLE_VECTOR_COMPUTE)
                continue;
            if (task->coreIndex < firstCore || task->coreIndex > lastCore)
                continue;

            unitLocal = task->unitIndex - npuIndex * unitsPerNpu;
            gapStart = unitLastEnd[unitLocal];
            if (task->startCycle - gapStart >= minGap && bubbleCount < 80) {
                /* 归因: 看依赖任务类型 */
                const char *reason = "调度间隙(下一任务未就绪)";
                char where[96];
                if (task->depTaskIndexA >= 0) {
                    long long depEnd = ctx.tasks[task->depTaskIndexA].endCycle;
                    if (depEnd > bestDepEnd) {
                        bestDepEnd = depEnd;
                        depRole = ctx.tasks[task->depTaskIndexA].role;
                    }
                }
                if (task->depTaskIndexB >= 0) {
                    long long depEnd = ctx.tasks[task->depTaskIndexB].endCycle;
                    if (depEnd > bestDepEnd) {
                        bestDepEnd = depEnd;
                        depRole = ctx.tasks[task->depTaskIndexB].role;
                    }
                }
                if (depRole == ROLE_MTE2_LOAD)
                    reason = "数据未就绪: 等待 MTE2 搬运(带宽/前序DMA排队)";
                else if (depRole == ROLE_MTE1_MOVE)
                    reason = "等待 L1->L0 (MTE1) 上载完成";
                else if (depRole == ROLE_MTE3_STORE)
                    reason = "等待前序结果回写 (MTE3)";
                else if (depRole == ROLE_AI_CPU_ISSUE)
                    reason = "AI CPU 指令下发/调度延迟";
                else if (depRole == ROLE_HCCS_COMM)
                    reason = "等待 HCCS 跨卡数据到达";
                else if (depRole == ROLE_CUBE_COMPUTE)
                    reason = "双缓冲未释放: 等待前一轮计算结束";
                else if (depRole == ROLE_VECTOR_COMPUTE)
                    reason = "双缓冲未释放: 等待前一轮向量计算结束";

                snprintf(where, sizeof(where), "%s @Core%d [%s c%d]",
                         task->role == ROLE_CUBE_COMPUTE ? "Cube" : "Vector",
                         task->coreIndex,
                         (task->kernelIndex >= 0 && task->kernelIndex < job->kernelCount)
                             ? job->kernels[task->kernelIndex].name : "-",
                         task->chunkIndex);
                emitBubbleMark(out, bubbleCount == 0 ? 1 : 0,
                               gapStart, task->startCycle, where, reason);
                bubbleCount++;
            }
            unitLastEnd[unitLocal] = task->endCycle;
        }
        free(unitLastEnd);
    }

    strbufAppend(out, "],\"rowsNote\":\"行值=忙碌占比‰\uFF080-1000\uFF09; 深色=忙碌, 空白=气泡\"");

    /* ---- 采样任务段(泳道 SVG 用, 按开始时间取前 800 条) ---- */
    /* **【GEMM_1】segs 采样: 本卡任务总数 taskOnCardCount=1,280 > 800 ⇒
     * stride=1280/800=1(等间隔抽样=全采, 容量截到 800), 每条 {s,e,r,c,k}
     * =任务的 开始/结束 cycle、role(0..6)、core、kernelIndex; 按开始时间
     * 排序(qsort)后进 JSON; 实测 segsTotal=1280(全量计数, 非保留数)。
     * 1,280 = 32 块 × 8 核 × 5 任务/份(CPU+LD+MTE1+Cube+ST), 与任务表
     * createMicroTask 的总创建数一致。 */
    {
        int taskOnCardCount = 0, keptCount = 0, seenCount = 0, stride;
        int capacity;
        SegmentSample *samples;

        for (taskIndex = 0; taskIndex < ctx.taskCount; taskIndex++) {
            if (ctx.tasks[taskIndex].npuIndex == npuIndex)
                taskOnCardCount++;
        }
        stride = taskOnCardCount > 800 ? taskOnCardCount / 800 : 1;
        capacity = taskOnCardCount / stride + 2;
        if (capacity > 800)
            capacity = 800;
        samples = (SegmentSample *)malloc(sizeof(SegmentSample) *
                                          (size_t)(capacity > 0 ? capacity : 1));
        for (taskIndex = 0; taskIndex < ctx.taskCount; taskIndex++) {
            if (ctx.tasks[taskIndex].npuIndex != npuIndex)
                continue;
            if ((seenCount % stride) == 0 && keptCount < capacity) {
                samples[keptCount].startCycle = ctx.tasks[taskIndex].startCycle;
                samples[keptCount].endCycle = ctx.tasks[taskIndex].endCycle;
                samples[keptCount].role = ctx.tasks[taskIndex].role;
                samples[keptCount].coreIndex = ctx.tasks[taskIndex].coreIndex;
                samples[keptCount].kernelIndex = ctx.tasks[taskIndex].kernelIndex;
                keptCount++;
            }
            seenCount++;
        }
        qsort(samples, (size_t)keptCount, sizeof(SegmentSample), compareSegmentByStart);

        strbufAppend(out, ",\"segs\":[");
        for (taskIndex = 0; taskIndex < keptCount; taskIndex++) {
            if (taskIndex)
                strbufAppendChar(out, ',');
            strbufAppendFormat(out, "{\"s\":%lld,\"e\":%lld,\"r\":%d,\"c\":%d,\"k\":%d}",
                               samples[taskIndex].startCycle, samples[taskIndex].endCycle,
                               samples[taskIndex].role, samples[taskIndex].coreIndex,
                               samples[taskIndex].kernelIndex);
        }
        strbufAppendFormat(out, "],\"segsTotal\":%d", taskOnCardCount);
        free(samples);
    }

    /* ---- 调度后逻辑 DAG(内核级采样前 64 节点, 供拓扑视图) ---- */
    /* **【GEMM_1】dag 输出: totalKernels=1 ⇒ nodes 只有 op0-GEMM
     * {i:0, name:"op0-GEMM", npu:0, kind:"Cube", deps:[]}; "deps" 取
     * kernel->predecessors(GEMM_1 为空); 模型模式的隐含串行边分支(depCount==0
     * 且上一内核同卡)不触发 —— ops 模式不建立该隐含边, 与单内核事实一致。 */
    strbufAppend(out, ",\"dag\":{");
    {
        int totalKernels = job->kernelCount;
        int shownKernels = totalKernels < 64 ? totalKernels : 64;
        int shownKernelIndex;

        strbufAppendFormat(out, "\"total\":%d,\"nodes\":[", totalKernels);
        for (shownKernelIndex = 0; shownKernelIndex < shownKernels;
             shownKernelIndex++) {
            WorkKernel *kernel = &job->kernels[shownKernelIndex];
            const char *kindLabel =
                kernel->kind == KERNEL_VECTOR ? "Vector"
                : (kernel->kind == KERNEL_DMA ? "MTE2"
                   : (kernel->kind == KERNEL_MIXED ? "Cube" : "Cube"));
            int depCount = 0, depIndex;
            if (shownKernelIndex)
                strbufAppendChar(out, ',');
            strbufAppendFormat(out, "{\"i\":%d,\"name\":", shownKernelIndex);
            strbufAppendEscapedString(out, kernel->name);
            strbufAppendFormat(out, ",\"npu\":%d,\"kind\":", kernel->npuIndex);
            strbufAppendEscapedString(out, kindLabel);
            strbufAppend(out, ",\"deps\":[");
            for (depIndex = 0; depIndex < kernel->predecessorCount; depIndex++) {
                if (depCount)
                    strbufAppendChar(out, ',');
                depCount++;
                strbufAppendFormat(out, "%d", kernel->predecessors[depIndex]);
            }
            /* 模型模式同卡相邻内核存在隐含串行依赖(未显式建边) */
            if (depCount == 0 && shownKernelIndex > 0 &&
                job->kernels[shownKernelIndex - 1].npuIndex == kernel->npuIndex &&
                job->stimulus.mode == STIMULUS_MODEL) {
                strbufAppendFormat(out, "%d", shownKernelIndex - 1);
            }
            strbufAppend(out, "]}");
        }
        strbufAppend(out, "]}");
    }
    strbufAppendChar(out, '}');
    logxTraceLeave(LOG_LEVEL_INFO, "simulateDetailLanes", "T=%lld cycles", totalCycles);
    simulationFree(&ctx);
}

