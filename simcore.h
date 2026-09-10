/* ============================================================================
 * simcore.h - 软硬协同调度仿真平台 C 引擎: 公共类型与常量
 *
 * 本头文件被 cfg.c / work.c / sim.c / run.c / server.c 共同引用, 定义:
 *   1) 各类仿真对象的数据结构(硬件/模型/激励/算法/内核);
 *   2) 内核种类、微任务角色、激励来源等枚举常量;
 *   3) 各类规模上限常量。
 *
 * 注意事项:
 *   - 面向 MSVC C 编译(cl /std:c11 /utf-8), 不使用 VLA/C99 变长数组;
 *   - 枚举的【数值顺序】是外部 JSON 输出契约的一部分(详情泳道的
 *     角色编号等会被前端引用), 新增枚举可以, 但禁止改动既有数值。
 * ==========================================================================*/
#ifndef SIMCORE_H
#define SIMCORE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ============================================================================
 * **【GEMM_1】** 贯穿用例总览(批注基准) —— 读下面任何字段/公式前先对照这里
 * ----------------------------------------------------------------------------
 * 贯穿用例 GEMM_1: 单个 GEMM 算子, 用来把本工程每一处数据结构、单位换算、
 * 切块与指标公式落到一个可复算的具体例子上。
 *   · 激励(src 节, ops 模式): {mode:"ops", layout:"dag", repeat:1, prec:2,
 *     autoNpu:0, ops:[{op:"gemm",label:"GEMM",m:4096,n:4096,k:4096}], edges:[]}
 *     —— 单算子、无依赖边、autoNpu=0 ⇒ 建模只产出 1 个内核 op0-GEMM,
 *       且固定落在 0 号卡(npuIndex=0); prec=2 ⇒ bytesPerElement=2(FP16, bpp=2)。
 *   · 硬件(hw 节, 只列与 GEMM_1 计算相关的项): npus=1, cores=8, freq=1.0GHz,
 *     cubeTf=320, vecTf=16, ubKb=256, hbmBw=1.2TB/s, l0Bw=2.0TB/s/核,
 *     mte1Lat=15, mte2Lat=40, mte3Lat=40(cyc), aicpuFreq=2.0
 *     (AI CPU 下发按固定 40 cyc 建模, 与 aicpuFreq 无关)。
 *   · 并行观: TP/DP/EP/CP/PP 概念上都=1 —— 单卡、无 PP 波形、无跨卡,
 *     本批注只观察"1卡×8核内 AI CPU / MTE1 / MTE2 / MTE3 / Cube / Vector
 *     的调度与掩盖"。
 *   · 算法/激励默认值(每处代码会再核对): coresPerKernel(group=0)→auto=8核;
 *     bufferDepth(dbuf 缺省)=2; maxChunksPerCore(maxChunk 缺省)=320;
 *     ubCapacityFraction(ubFrac 缺省)=0.66; tileSplit 缺省=32。
 *   · 全链路推导结果(GEMM_1 用例实测, _rn2/GEMM_1.json 复跑确认):
 *       内核数 nkers=1; cubeFlops=2·4096³≈1.3743895e11;
 *       loadBytes=(4096·4096+4096·4096)×2=67,108,864B(=MK+NK 两输入);
 *       storeBytes=4096·4096×2=33,554,432B(=MN 输出);
 *       Tlb=1.3743895e11/(320×1e3)≈429,497 cyc;
 *       切块 P=min(ceil(8,388,608/173,015)=49, 320, 32)=32;
 *       chunkInputBytes=67,108,864/(8×32)=262,144B(256KB, >ubCap≈173KB,
 *       属 tileSplit=32 造成的近似超容); chunkOutputBytes=33,554,432/256=131,072B;
 *       每 chunk Cube≈13,422 cyc; 总周期≈431,913 cyc(Tlb+0.56%);
 *       Cube 利用率≈99.44%; Vector=0; utilMte2≈15.32%; utilMte3≈8.84%;
 *       dmaBusy≈104,366; 掩盖率≈99.61%; SRAM 峰值=(262,144×2+131,072)/1024=640KB;
 *       泳道 rows=28(4卡级+8核×3), bubbles=8, segs=1280(=32×8×5)。
 * ==========================================================================*/

/* ============================================================================
 * 规模上限常量
 * ==========================================================================*/
#define MAX_ALGORITHMS         8   /* 一次请求可同时对比的调度算法个数上限 */
#define MAX_PREDECESSORS       8   /* 单个内核允许的依赖(前驱)数上限       */
#define MAX_PIPELINE_STAGES 4096   /* PP 流水阶段(参与切分的算卡)数上限     */

/* ============================================================================
 * 内核计算种类(KernelKind)
 * 每个逻辑内核(WorkKernel)属于其中一种; memset 清零后默认值为 KERNEL_CUBE。
 * ==========================================================================*/
typedef enum {
    KERNEL_CUBE   = 0,   /* 纯矩阵计算(Cube): GEMM / Conv / 各类投影      */
    KERNEL_VECTOR = 1,   /* 纯向量计算(Vector): LayerNorm / Softmax / GELU */
    KERNEL_MIXED  = 2,   /* 混合内核: Cube 计算 + Vector 计算先后执行      */
    KERNEL_DMA    = 3,   /* 纯搬运内核                                     */
    KERNEL_COMM   = 4,   /* 跨卡通信内核(预留)                             */
    KERNEL_KIND_COUNT    /* 种类计数(兼作数组大小用)                       */
} KernelKind;

/* ============================================================================
 * 微任务角色(MicroTaskRole)
 * 仿真的最小执行单元为微任务(Task), 每个微任务承担一种硬件角色。
 * 数值顺序同时决定了“单位(unit)表”的排布方式(见 sim.c), 勿改动!
 *
 * **【GEMM_1】角色含义与"卡级共享(0..3) vs 每核独占(4..6)"布局**
 * 角色分两族, 直接决定硬件单元表怎么建(sim.c simulationInit/computeUnitIndex):
 *   · 卡级共享(0..3): AI CPU / MTE2 / MTE3 / HCCS —— 每张卡各 1 个单元、
 *     全卡一个队列; 同卡上任意核产生的这类任务都在这一个队列上串行排队
 *     (模拟"带宽/端口/引擎是卡的公共资源")。GEMM_1 中:
 *       ROLE_AI_CPU_ISSUE(0) 每 chunk 固定 40 cyc 的指令下发/点火, 全卡排队,
 *       32chunk×8核=256 次 ⇒ ctrlBusy 实测 10,240=40×256;
 *       ROLE_MTE2_LOAD(1) 把输入 A/B 从 HBM 载入片上, 卡级共享 ⇒ 256 次载入
 *       在一条队列上串行, 每次≈258.45 cyc(218.45 带宽 + 40 lat), 合计
 *       ≈66,163 cyc ⇒ utilMte2 实测 15.32%;
 *       ROLE_MTE3_STORE(2) 结果 C 写回 HBM, 256 次每次≈149.23 cyc
 *       (109.23+40), 合计≈38,202 cyc ⇒ utilMte3 实测 8.84%;
 *       ROLE_HCCS_COMM(3) 仅跨卡依赖才产生 —— GEMM_1 单卡无跨卡, 恒空。
 *   · 每核独占(4..6): MTE1 / Cube / Vector —— 每核各 3 个单元, 核间并行。
 *       ROLE_MTE1_MOVE(4) L1→L0 上载喂 Cube(只喂输入, 不搬输出),
 *       每核每 chunk≈146.07 cyc(131.07+15), GEMM_1 中 8 核并行开展;
 *       ROLE_CUBE_COMPUTE(5) 真正的矩阵计算, 每核每 chunk≈13,421.77 cyc,
 *       32 块首尾相接占满几乎全部时间 ⇒ Cube 利用率 99.44%;
 *       ROLE_VECTOR_COMPUTE(6) 元素级/混合内核才有 —— GEMM_1 内核 kind=
 *       KERNEL_CUBE 纯 Cube, Vector 分支不触发 ⇒ utilVec=0、Vector 泳道全空属正常。
 * ==========================================================================*/
typedef enum {
    ROLE_AI_CPU_ISSUE  = 0,  /* AI CPU 指令下发 / 控制(卡级共享)      */
    ROLE_MTE2_LOAD     = 1,  /* MTE2 载入: L2/HBM -> L1/UB(卡级共享)  */
    ROLE_MTE3_STORE    = 2,  /* MTE3 写出: L1/UB -> HBM(卡级共享)     */
    ROLE_HCCS_COMM     = 3,  /* HCCS 卡间通信(卡级共享)               */
    ROLE_MTE1_MOVE     = 4,  /* MTE1 上载: L1 -> L0(每核独占)         */
    ROLE_CUBE_COMPUTE  = 5,  /* Cube 计算(每核独占)                   */
    ROLE_VECTOR_COMPUTE= 6,  /* Vector 计算(每核独占)                 */
    ROLE_COUNT               /* 角色计数                               */
} MicroTaskRole;

/* ============================================================================
 * 硬件配置(HardwareConfig): 与页面"硬件配置"一一对应
 *
 * **【GEMM_1】仅列与本次计算相关的字段, 逐一说明(量纲/JSON键/来源/用在哪):
 *   · numNpu/coresPerNpu: 算卡数与每卡核数(JSON npus/cores)。GEMM_1=1卡×8核;
 *     coresPerNpu 还是"整卡峰值→每核算力"的除法分母(见 cubePeakTflops)。
 *   · coreClockGhz: 仿真主时钟(JSON freq)。GEMM_1=1.0; 一切"秒→cycle"的换算
 *     都乘它(cyclesForByteTransfer/cyclesForFlops/…)。带宽类换算的分子"字节/秒"
 *     除以它即得"每 cycle 可搬字节"; 算力类换算见 cubePeakTflops。
 *   · cubePeakTflops(JSON cubeTf): 每卡 Cube 峰值 TFLOPS, 1T=1e12FLOP/s。
 *     GEMM_1=320 ⇒ 全卡每 cycle(1GHz)=320e12/1e9=320,000 FLOPs;
 *     每核峰值=320e12/8=4e13 FLOP/s ⇒ 每核每 cycle=40,000 FLOPs ⇒ 每 chunk
 *     536,870,912FLOPs 需 13,421.77 cyc。理论下界 Tlb 的除法分母用 ×1e3
 *     (见 work.c: flops×freq/(cubeTf×1e3)=429,497)。
 *   · vectorPeakTflops(JSON vecTf): 每卡 Vector 峰值。GEMM_1=16 但纯 GEMM
 *     无向量任务 ⇒ 仅用于把 vectorFlops=0 折算成 0 cyc, 不出现在结果里。
 *   · ubSizeKb(JSON ubKb): 每核 UB 容量 KB。GEMM_1=256; 与 ubCapacityFraction
 *     一起定"单块在飞预算" ubCapacityBytes=256×1024×0.66≈173,015B(软上限),
 *     是切块数 chunkCount 容量维度的来源。
 *   · hbmBandwidthTBs(JSON hbmBw): HBM 带宽 TB/s。GEMM_1=1.2; MTE2 载入与
 *     MTE3 写出都按它折算: 调用时换算成 B/s = ×1e12, 每 chunk 载入
 *     262,144B ⇒ 262144/(1.2e12/1e9)=218.45 cyc(再+40 lat)。
 *   · l0BandwidthTBsPerCore(JSON l0Bw): MTE1(L1→L0)每核带宽 TB/s。GEMM_1=2.0;
 *     每 chunk 262,144B ⇒ 262144/(2e12/1e9)=131.07 cyc(再+15 lat)。
 *   · mte1LatencyCycles/mte2LatencyCycles/mte3LatencyCycles(JSON mte1Lat/
 *     mte2Lat/mte3Lat): 各搬运引擎的固定启动/时延开销, 与带宽时间相加才是
 *     mteX 任务的完整耗时。GEMM_1=15/40/40(逐任务加计, 是全工程"逐任务时延
 *     显式计费"的来源之一, 使实测总周期 431,913 > 理论下界 429,497)。
 *   · (未列字段 l0SizeKb、l2SizeMb、hbmSizeGb、l2BandwidthTBs、aicBus 相关键、
 *     hccs/roce 相关键 与 GEMM_1 的 Cube 流水无关: l2Mb/hbmGb 只用于 HBM 容量
 *     告警, aicBus 键解码后未参与本用例; hccs/roce 键仅跨卡通信分支使用。)
 * ==========================================================================*/
typedef struct {
    int    numNpu;                  /* 算卡(NPU)数量                       */
    int    coresPerNpu;             /* 每张算卡上的 AI Core 核数           */
    char   topology[16];            /* 卡间互联拓扑: ring/linear/mesh/torus/full */
    double coreClockGhz;            /* AI Core 主频 GHz(仿真主时钟)        */
    double aicpuClockGhz;           /* AI CPU 主频 GHz                      */
    double cubePeakTflops;          /* 每卡 Cube 峰值算力 TFLOPS(FP16)     */
    double vectorPeakTflops;        /* 每卡 Vector 峰值算力 TFLOPS         */
    double l0SizeKb;                /* 每核 L0 Buffer 容量 KB              */
    double ubSizeKb;                /* 每核 UB(Unified Buffer, L1/UB SRAM)容量 KB
                                       (默认256)。与算法级 ubCapacityFraction(ubFrac)
                                       相乘即"单块在飞数据预算" ubCapacityBytes =
                                       ubSizeKb×1024×ubCapacityFraction, 是切块数与
                                       每块大小的容量基础(公式与代码变量见 AlgorithmConfig
                                       上方说明及 sim.c scheduleOneKernel)。改它=改硬件口径,
                                       改 ubFrac=同一硬件下更激进取用 UB。 */
    double l2SizeMb;                /* 卡内共享 L2 Cache 容量 MB           */
    double hbmSizeGb;               /* 每卡 HBM 显存容量 GB                */
    double l2BandwidthTBs;          /* L2 Cache 带宽 TB/s                  */
    double hbmBandwidthTBs;         /* HBM 带宽 TB/s(MTE2/MTE3 均视为 HBM 方向) */
    double l0BandwidthTBsPerCore;   /* MTE1(L1->L0) 每核数据带宽 TB/s      */
    double mte1LatencyCycles;       /* MTE1 时延(cycles)                   */
    double mte2LatencyCycles;       /* MTE2 时延(cycles)                   */
    double mte3LatencyCycles;       /* MTE3 时延(cycles)                   */
    double aicAicpuBandwidthGBs;    /* AI Core <-> AI CPU 总线带宽 GB/s    */
    double aicAicpuLatencyUs;       /* AI Core <-> AI CPU 总线时延 us      */
    double hccsBandwidthGBs;        /* 卡间 HCCS 单跳带宽 GB/s             */
    double hccsLatencyUs;           /* 卡间 HCCS 单跳时延 us               */
    double roceBandwidthGbps;       /* 机间 RoCE 网络带宽 Gbps             */
    double roceLatencyUs;           /* 机间 RoCE 网络时延 us               */
} HardwareConfig;

/* ============================================================================
 * 模型配置(ModelConfig): 激励来源为"模型生成"时生效(见页面"模型配置")
 * ==========================================================================*/
typedef struct {
    char   presetName[32];      /* 模型预设: llama3-70b / deepseek-v3 / custom */
    double hiddenDim;           /* 隐层维度 h                                 */
    double numLayers;           /* Transformer 层数 L                         */
    double numHeads;            /* 注意力头数 H                               */
    double numKvHeads;          /* KV 头数 Hkv(MHA=H; GQA/MLA 更小)           */
    double vocabSize;           /* 词表大小 V                                 */
    double seqLen;              /* 序列长度 s                                 */
    double microBatch;          /* MicroBatch 大小(JSON键 mb; 近似 PP 波形数 waves) */
    double bytesPerElement;     /* 每元素字节数: 2=FP16/BF16, 1=FP8           */
    int    tensorParallel;      /* TP 张量并行度                              */
    int    pipelineParallel;    /* PP 流水线并行度                            */
    int    dataParallel;        /* DP 数据并行度                              */
    int    contextParallel;     /* CP 上下文(序列)并行度                      */
    int    expertParallel;      /* EP 专家并行度                              */
    int    enableMoe;           /* 是否启用 MoE 架构                          */
    double numExperts;          /* MoE: 专家总数 E                            */
    double topKSelected;        /* MoE: TopK 激活的专家数                     */
    double expertFfnDim;        /* MoE: 每个专家的 FFN 宽度                   */
    int    maxTileSplitsPerKernel; /* 每算子/内核的 Tile 切分数上限(激励级, JSON键
                                      "tileSplit", 默认32, ≥1)。
                                      与算法级 maxChunksPerCore 同作用: 对最终切块数
                                      chunkCount 做上限钳制, 两处取更严格者(通常本值
                                      更小先命中); 调大=允许该算子切得更细(每块更小),
                                      调小=粗切。 */
    char   zeroStage[8];        /* ZeRO 优化策略: off/1/2/3                  */
    char   actCheckpoint[8];    /* 激活重算: none/selective/full             */
    char   cpuOffload[8];       /* CPU Offload: none/optimizer/params        */
} ModelConfig;

/* ============================================================================
 * 激励来源(StimulusMode)与激励配置(StimulusConfig)
 * ==========================================================================*/
typedef enum {
    STIMULUS_MODEL     = 0,   /* 激励来自"模型生成": 按模型结构逐层建内核 */
    STIMULUS_OPERATORS = 1    /* 激励来自"算子编排": 单算子/链式/DAG       */
} StimulusMode;

typedef struct {
    StimulusMode mode;      /* 激励来源选择                                */
    int repeat;             /* ops 模式: 整张算子图重复次数(>=1; DAG 须为1) */
} StimulusConfig;

/* ============================================================================
 * **【GEMM_1】AlgorithmConfig 在本用例的实际取值(默认值来源见 cfg.c decodeOneAlgorithm)
 *   · algoKey="dfs" / algoName="默认参数调度" / issueOrder="dfs"(GEMM_1.json 给出);
 *     issueOrder[0]!='c' ⇒ runOperatorGraph 用"就绪顺序"(序号最小先调度)
 *     而非关键路径优先分支。
 *   · coresPerKernel=group=0(auto) ⇒ sim.c 解析 groupCores=coresPerNpu=8(整卡);
 *     每核摊薄 loadBytes/8=8,388,608B、storeBytes/8=4,194,304B。
 *   · bufferDepth=dbuf 缺省2 ⇒ 双缓冲: chunk 序号 c 的载入须等 c-2 的计算完成。
 *   · maxChunksPerCore=maxChunk 缺省320 ⇒ 容量算出的理想块数 49<320, 不构成钳制。
 *   · ubCapacityFraction=ubFrac 缺省0.66 ⇒ ubCapacityBytes=256×1024×0.66≈173,015B。
 *   · prefetchWindow=lookahead 缺省0: 预留, 仿真不使用。
 *   切块钳制链(代码见 sim.c scheduleOneKernel): 理想块数 ceil(49) →
 *   min(49, maxChunk=320)=49 → min(49, tileSplit=32)=32 ⇒ P=32。
 * ==========================================================================*/

/* ============================================================================
 * 单条调度算法配置(AlgorithmConfig): 与页面"调度算法"卡片一一对应
 *
 * 【切块(Tile/K-chunk)参数语义 —— 阅读本结构前先看这里】
 * 仿真把每个内核的数据按"块"(chunk/Tile)切小搬运到片上。切块由以下参数共同决定,
 * 公式与代码变量(sim.c scheduleOneKernel)的对应关系如下:
 *
 *   groupCores        = coresPerKernel(0=全部核)  ← 参与切块的核数
 *   ubCapacityBytes   = ubSizeKb×1024×ubCapacityFraction
 *                                              ← 每块"在飞数据"预算(软上限)
 *   每核最大驻留字节   = max(loadBytes/groupCores, storeBytes/groupCores)
 *                                              ← 每核输入/输出字节的较大者
 *                                                (代码 perCoreInput/perCoreOutput,
 *                                                取大者存于 chunkBytesLimit)
 *   理想切块数        = ceil(每核最大驻留字节 / ubCapacityBytes)
 *                                              ← 按容量"想切几块"
 *   最终切块数(代码变量 chunkCount)
 *                     = min(理想切块数, maxChunksPerCore, maxTileSplitsPerKernel)
 *                                              ← 数量上限两道钳制
 *   每核每块输入/输出/计算量
 *     (代码 chunkInputBytes / chunkOutputBytes / chunkCubeFlops)
 *                     = 内核总字节(FLOPs) ÷ (groupCores × chunkCount)
 *                                              ← 每核每块大小(=结果,非输入)
 *
 * 三个"旋钮"的分工:
 *   · ubCapacityFraction(ubFrac): 每块尺寸软上限的百分比 -> 想切碎就调小;
 *   · maxChunksPerCore(maxChunk): 每核每内核最多切几块(调度级刹车);
 *   · maxTileSplitsPerKernel(tileSplit, 激励级, 见 ModelConfig): 每算子最多切几块(业务级刹车)。
 *   ubCapacityBytes 限"每块能多大", maxChunk/tileSplit 限"最多切几块":
 *   若理想切块数 ≤ maxChunk 则每块 ≤ ubCapacityBytes(容量主导);
 *   若理想切块数 > maxChunk 则被强切粗(每块 > ubCapacityBytes, 属近似超容, 仿真照跑
 *   并体现在 SRAM 峰值与时间上)。
 * 仿真刻意不提供"显式 tile 大小/形状"参数: WorkKernel 只保留总字节/FLOPs、
 * 不携带 M×N×K 几何, 且真机上 tile 尺寸本也由容量 Tiling 推导(见 sim.c scheduleOneKernel)。
 * ==========================================================================*/
typedef struct {
    char   algoKey[24];         /* 算法标识: naive / dfs / cp            */
    char   algoName[80];        /* 算法显示名                            */
    char   issueOrder[12];      /* 指令重排策略: dfs=就绪顺序 / cp=关键路径优先 */
    int    coresPerKernel;      /* 每个内核占用核数, 0=auto=占满全卡;
                                    参与切块(每核分摊数据量=总字节/groupCores)
                                    与并行度 */
    int    bufferDepth;         /* 软件流水缓冲深度 1..4(dbuf: 双/三/四级缓冲)。
                                   每轮块循环(块序号 chunkIndex)的载入须等
                                   序号 chunkIndex-bufferDepth 的块计算完
                                   (缓冲槽释放)才开始; =1 时搬运-计算近似串行,
                                   ≥2 才可能把下一块搬运藏进当前块计算。 */
    int    maxChunksPerCore;    /* 每核每内核的 K-chunk(Tile) 数上限(默认320,1~4096)。
                                   是切块数量钳制之一: 小于容量算出的理想切块数时,
                                   最终切块数 chunkCount 被强制为 maxChunksPerCore
                                   (每块将超过每块在飞预算 ubCapacityBytes, 近似超容)。 */
    double ubCapacityFraction;  /* UB 可被占用比例(0.05..0.95, 默认0.66)。
                                   单块在飞数据预算 ubCapacityBytes =
                                   ubSizeKb×1024×该值。 */
    int    prefetchWindow;      /* 预留: 指令预取窗口(解码但仿真未使用)     */
} AlgorithmConfig;

/* ============================================================================
 * 详情视图配置(DetailViewConfig): 决定结果页"泳道图"的抽样范围
 * ==========================================================================*/
typedef struct {
    int algorithmIndex;   /* 需要输出详细泳道的算法下标              */
    int npuIndex;         /* 需要输出详细泳道的算卡号                */
    int firstCore;        /* 泳道展示的核范围起点                    */
    int lastCore;         /* 泳道展示的核范围终点                    */
    int timeBinCount;     /* 时间轴分箱数(泳道分辨率, 60..3000)      */
} DetailViewConfig;

/* ============================================================================
 * **【GEMM_1】WorkKernel 在 GEMM 语义下的字段含义(建模见 work.c appendOperatorKernel)
 * 每个字段都是"总账"(整内核、切块前), 之后 sim.c scheduleOneKernel 再按
 * groupCores×chunkCount 均分给每核每块。
 *   · kind: 内核计算种类。GEMM/Conv 归 KERNEL_CUBE(=0), 只生成 Cube 任务链
 *     (MTE2载入→MTE1上载→Cube→MTE3写出), 不产生 Vector 任务 —— GEMM_1 的
 *     utilVec=0、Vector 泳道全空都源于此。
 *   · cubeFlops: 总 Cube 计算量(未切块)。GEMM C=A×B 一次乘加=2 FLOPs,
 *     ⇒ cubeFlops=2·m·n·k=2·4096³≈1.3743895e11; 参与 理论下界 Tlb
 *     (=flops×freq/(cubeTf×1e3)≈429,497) 与每块计算量 chunkCubeFlops
 *     =cubeFlops/(8×32)=536,870,912。
 *   · loadBytes: 总输入搬运字节。GEMM 读 A(M×K)与 B(K×N):
 *     loadBytes=(m·k+n·k)×bpp=(4096·4096+4096·4096)×2=67,108,864B。
 *     每核摊薄 8,388,608B, 每核每块 chunkInputBytes=262,144B。
 *   · storeBytes: 总输出搬运字节。GEMM 写 C(M×N):
 *     storeBytes=m·n×bpp=4096·4096×2=33,554,432B; 每核每块=131,072B。
 *   · npuIndex: 所在算卡。GEMM_1: autoNpu=0 且 op 无 npu 键 ⇒ npuIndex=0
 *     (work.c: autoAssignNpu 为假时一律 0 号卡), 单卡场景无跨卡依赖。
 *   · predecessorCount/predecessors: DAG 依赖。GEMM_1 edges=[] ⇒
 *     predecessorCount=0 ⇒ 内核"立即可调度"(runOperatorGraph 第一轮选中)。
 * ==========================================================================*/

/* ============================================================================
 * 逻辑内核(WorkKernel): 建模阶段生成、调度/仿真阶段消费的基本单位
 * 每个内核是一类计算+搬运的抽象, 仿真时再按缓冲深度切成 K-chunk。
 * ==========================================================================*/
typedef struct {
    char   name[96];         /* 内核名(算子实例标识, 如 op0-GEMM / L1-QKV) */
    KernelKind kind;         /* 内核计算种类                                */
    double cubeFlops;        /* 总 Cube 计算量(未切块前)                    */
    double vectorFlops;      /* 总 Vector 计算量(未切块前)                  */
    double loadBytes;        /* 总输入搬运字节                              */
    double storeBytes;       /* 总输出搬运字节                              */
    int    npuIndex;         /* 所在算卡编号                                */
    int    commBytes;        /* 预留: 若>0 表示内核结束需跨卡通信的字节数   */
    int    predecessorCount;                       /* 依赖(前驱)内核个数   */
    int    predecessors[MAX_PREDECESSORS];         /* 前驱内核下标数组     */
} WorkKernel;

#endif /* SIMCORE_H */
