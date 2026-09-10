/* ============================================================================
 * cfg.c - 请求配置解码与校验
 *
 * 职责: 把前端 POST 上来的请求 JSON 解码为结构化 EngineJob(硬件/模型/
 * 激励/算法/视图), 完成默认值填充、范围钳制与"跨页一致性校验"。
 * 本文件只负责解码与换算, 不做建模(建模见 work.c)。
 *
 * 换算工具(cyclesFor*): 把 字节/带宽、微秒、FLOPs 统一折算为仿真主时钟
 * 的 cycle 数, 供 sim.c 使用。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】本文件视角(解码与换算)
 * cfg.c 是 GEMM_1 全部"配置输入"的落地点: GEMM_1.json 的 hw/src/algos/view
 * 四节在这里被解码为 EngineJob 的 hardware/model/stimulus/algorithms/detailView。
 * 本批注对与 GEMM_1 计算有关的四件事逐一展开:
 *   ① decodeHardwareConfig: 硬件默认值与 JSON 覆盖(npus/cores/ubKb/hbmBw/l0Bw/
 *      cubeTf/vecTf/mte1/2/3Lat/freq), 并说明 TB/s→B/s(×1e12)、KB→B(×1024)
 *      等"单位换算"发生在哪里(换算函数本身见下);
 *   ② decodeOneAlgorithm: 算法缺省值来源 —— group=0(auto 8核)、dbuf=2、
 *      maxChunk=320、ubFrac=0.66、lookahead=0(GEMM_1 的 algos 只给 id/name/order,
 *      其余四值全靠这里的缺省);
 *   ③ decodeJobConfig: 激励/模型节读取 —— ops 模式下 tp/pp/dp… 概念=1 不参与建模,
 *      tileSplit 缺省 32(GEMM_1 切块 P=32 的第三道钳制来源), view 缺省与覆盖;
 *   ④ cyclesForByteTransfer / cyclesForMicroseconds / cyclesForFlops:
 *      三种"秒/字节/FLOPs → 主时钟 cycle"换算, 给出单位推导与 GEMM_1 代值。
 * ==========================================================================*/
#include "engine.h"

/* ============================================================================
 * 单位换算工具
 * ==========================================================================*/

/* 搬运时延: T_cycles = bytes / (带宽字节每秒 / 主频) = bytes × 主频 / 带宽
 * 即把 "字节/带宽" 的秒数按主时钟频率换算成 cycles。 */
/* ============================================================================
 * **【GEMM_1】cyclesForByteTransfer —— 搬运耗时换算(带宽→cycle)逐量说明
 * 形参: bytes=搬运字节数; bandwidthBytesPerSec=带宽(单位: 字节/秒, 注意调用方
 * 必须先完成 TB/s→B/s 的 ×1e12); clockGhz=主时钟(仿真频率, GEMM_1=1.0)。
 * 单位推导: 带宽每秒可搬 bandwidthBytesPerSec/1e9 字节/cycle(1GHz 基准) ⇒
 *   用时(cycle)= bytes ÷ (带宽字节每秒 ÷ 频率赫兹)
 *             = bytes × clockGhz×1e9 / bandwidthBytesPerSec。
 * GEMM_1 代值示例(MTE2 载入每 chunk 输入 262,144B, HBM 1.2TB/s=1.2e12B/s):
 *   262144/(1.2e12/(1.0×1e9))=262144/1200≈218.45 cyc —— 这是"纯带宽账",
 *   每任务还要在此之上加 mte2LatencyCycles=40(见 sim.c mte2LoadCycleCost);
 *   同一公式给 MTE3 写出(131,072B→109.23)与 MTE1(按 2e12B/s 每核带宽)用,
 *   只是带宽参数与附加 lat 不同。
 * 防御分支: bytes≤0 或带宽≤0 直接返回 0(GEMM_1 中输出字节恒>0, 不会走 0)。
 * ==========================================================================*/
double cyclesForByteTransfer(double bytes, double bandwidthBytesPerSec, double clockGhz) {
    if (bytes <= 0 || bandwidthBytesPerSec <= 0)
        return 0;
    return bytes / (bandwidthBytesPerSec / (clockGhz * 1e9));
}

/* 时延(us) -> cycles: us × freq(GHz) × 1000 */
/* ============================================================================
 * **【GEMM_1】cyclesForMicroseconds —— 固定时延(us)→cycle
 * 推导: 1us = 1e-6 s; 主频 f(GHz)=f×1e9 cycle/s ⇒ us 数 × f×1e9 × 1e-6
 * = microseconds × clockGhz × 1000。
 * 使用点: 仅跨卡通信(HCCS 单跳时延 hccsLatencyUs)与 AI Core↔AI CPU 总线时延
 * 会调用它。GEMM_1 单卡、无跨卡 ⇒ 本函数在贯穿用例里不被触发(0 次调用),
 * 保留它只为说明"us 型时延"与"cycle 型时延(mte*Lat)"在仿真里的两种建模口径:
 * mte* 直接以 cycle 计数, hccs/aicBus 以 us×freq 折算。
 * ==========================================================================*/
double cyclesForMicroseconds(double microseconds, double clockGhz) {
    return microseconds * clockGhz * 1000.0;
}

/* 计算时延: 把 groupCores 个核共同分担 flops, 按"单核摊薄算力"折算 cycles。
 * peakTflops 为整卡峰值, 每核算力 = peakTflops×1e12 / coresPerNpu。 */
/* ============================================================================
 * **【GEMM_1】cyclesForFlops —— 计算耗时换算(FLOPs→cycle)逐量说明
 * 形参: flops=本任务 FLOPs; peakTflops=整卡峰值(TFLOPS, 1T=1e12FLOP/s);
 *   coresPerNpu=每卡核数(把"整卡峰值"摊到单核的除法分母);
 *   clockGhz=主频; groupCores=承担本任务的实际核数(本函数把 flops 均分给它们)。
 * 局部量推导:
 *   perCoreFlops = flops / groupCores —— 每核要算的 FLOPs;
 *   perCorePeak  = peakTflops×1e12 / coresPerNpu —— 每核算力(FLOP/s):
 *     GEMM_1: 320e12/8 = 4e13 FLOP/s, 即每核每 cycle 4e4 FLOP(1GHz);
 *   cycles = perCoreFlops/perCorePeak × clockGhz×1e9 —— (FLOPs÷FLOP/s)×频率。
 * GEMM_1 代值(Cube 每核每块 chunkCubeFlops=536,870,912, groupCores=1):
 *   536,870,912/4e13×1e9 ≈ 13,421.77 cyc/块; ×32块/核=429,497 cyc/核=理论下界。
 * 使用点: sim.c cubeComputeCycleCost/vectorComputeCycleCost(GEMM_1 只有 Cube
 * 路径; vectorFlops=0, Vector 分支不触发)。
 * ==========================================================================*/
double cyclesForFlops(double flops, double peakTflops, int coresPerNpu,
                      double clockGhz, int groupCores) {
    double perCoreFlops, perCorePeak;
    if (flops <= 0 || peakTflops <= 0)
        return 0;
    if (coresPerNpu < 1)
        coresPerNpu = 1;
    if (groupCores < 1)
        groupCores = 1;
    perCoreFlops = flops / groupCores;
    perCorePeak  = peakTflops * 1e12 / coresPerNpu;
    return perCoreFlops / perCorePeak * clockGhz * 1e9;
}

/* 卡间拓扑最短跳数: 决定跨卡通信一次要走几跳
 *   full=全互联(1 跳), ring=环(双向最近), mesh/torus=近似网格,
 *   其余(如 linear)按编号差线性跳数。 */
int computeTopologyHops(int npuCount, const char *topology, int fromNpu, int toNpu) {
    int ringDistance, colCount, rowHop;
    int effectiveNpuCount;

    if (fromNpu == toNpu)
        return 0;
    effectiveNpuCount = npuCount < 1 ? 1 : npuCount;
    ringDistance = fromNpu > toNpu ? fromNpu - toNpu : toNpu - fromNpu;

    if (!strcmp(topology, "full"))
        return 1;
    if (!strcmp(topology, "ring"))
        return ringDistance < effectiveNpuCount - ringDistance
                   ? ringDistance : effectiveNpuCount - ringDistance;
    if (!strcmp(topology, "mesh") || !strcmp(topology, "torus")) {
        colCount = 1;
        while (colCount * colCount < effectiveNpuCount)
            colCount++;
        rowHop = fromNpu / colCount > toNpu / colCount
                     ? fromNpu / colCount - toNpu / colCount
                     : toNpu / colCount - fromNpu / colCount;
        if (fromNpu % colCount != toNpu % colCount)
            rowHop++;   /* 不在同一列时需多走一步(近似) */
        return rowHop < 1 ? 1 : rowHop;
    }
    return ringDistance;
}

/* ============================================================================
 * 硬件配置解码(HardwareConfig)
 * 先写满默认值, 再覆盖请求 JSON 中给出的字段, 最后做范围钳制。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】decodeHardwareConfig 三段式填充(默认 → JSON 覆盖 → 钳制)与单位口径
 * 读取顺序决定"缺省兜底": memset 清零 → 函数开头"默认规格"段写满内置默认 →
 * 若请求有 hw 节, 逐键 jsonGet* 覆盖; 若某键缺失则保留默认。最后"范围钳制"段收尾。
 * 关键: 本函数只存"原始单位"数值 —— 带宽存 TB/s、容量存 KB/MB/GB、时延存
 * cycle/us —— 具体 ×1e12(TB/s→B/s)、×1024(KB→B)的换算不在解码期做,
 * 而在使用点(sim.c 调 cyclesForByteTransfer 时 ×1e12、scheduleOneKernel 算
 * ubCapacityBytes 时 ×1024)。这是 GEMM_1 所有数值的量纲源头:
 *   npus=1(卡)  cores=8(核/卡)  freq=1.0GHz  aicpuFreq=2.0(仅解码, 下发仍 40cyc)
 *   cubeTf=320(卡级 TFLOPS)  vecTf=16  l0Kb=64(不用)  ubKb=256(切块容量根)
 *   hbmBw=1.2(TB/s; MTE2/3 带宽)  l0Bw=2.0(TB/s/核; MTE1 带宽)
 *   mte1Lat=15 / mte2Lat=40 / mte3Lat=40(cyc, 逐任务加在带宽账上)
 *   其余(l2Mb/hbmGb/l2Bw/aicBus/hccs/roce)只服务于跨卡/告警, GEMM_1 不触达。
 * 钳制要点: 核数 1..256、主频≤0 回退 1.0、cubeTf≤0 回退 1、hbmBw≤0 回退 0.1、
 * ubKb≤8 抬到 8(GEMM_1 全部在合法区, 不触发)。
 * ==========================================================================*/
static void decodeHardwareConfig(HardwareConfig *hardware, JsonValue *requestJson) {
    JsonValue *hwJson;

    memset(hardware, 0, sizeof(*hardware));

    /* ---- 默认规格(与页面预设一致) ----
     * **【GEMM_1】内置默认与 GEMM_1.json 覆盖对照(默认→用例实际值):
     *   numNpu=4→1; coresPerNpu=32→8; coreClockGhz=1.0→1.0(未覆盖, 仍 1.0);
     *   cubePeakTflops=320→320; vectorPeakTflops=16→16; ubSizeKb=256→256;
     *   hbmBandwidthTBs=1.2→1.2; l0BandwidthTBsPerCore=2.0→2.0;
     *   mte1/2/3LatencyCycles=15/40/40→15/40/40; topology="ring"(单卡无意义)。
     *   默认值本身取自"页面预设"的典型 910B 规格 —— 这批默认正是 GEMM_1 想要
     *   考察的"1卡8核 Cube320T/Vector16T/UB256K/HBM1.2T"的等价物, 故用例把
     *   覆盖集中在 npus=1/cores=8/cubeTf=320 三处以构造"1 卡 + 大 Cube"。 */
    hardware->numNpu = 4;
    hardware->coresPerNpu = 32;
    strcpy(hardware->topology, "ring");
    hardware->coreClockGhz = 1.0;
    hardware->aicpuClockGhz = 2.0;
    hardware->cubePeakTflops = 320;
    hardware->vectorPeakTflops = 16;
    hardware->l0SizeKb = 64;
    hardware->ubSizeKb = 256;
    hardware->l2SizeMb = 192;
    hardware->hbmSizeGb = 64;
    hardware->l2BandwidthTBs = 3.0;
    hardware->hbmBandwidthTBs = 1.2;
    hardware->l0BandwidthTBsPerCore = 2.0;
    hardware->mte1LatencyCycles = 15;
    hardware->mte2LatencyCycles = 40;
    hardware->mte3LatencyCycles = 40;
    hardware->aicAicpuBandwidthGBs = 64;
    hardware->aicAicpuLatencyUs = 0.8;
    hardware->hccsBandwidthGBs = 392;
    hardware->hccsLatencyUs = 1.0;
    hardware->roceBandwidthGbps = 400;
    hardware->roceLatencyUs = 2.5;

    /* ---- 覆盖请求字段(缺失则保留默认) ----
     * **【GEMM_1】jsonGetXxx(...,默认值) 逐键覆盖清单(键→字段→GEMM_1 取值):
     *   "npus"→numNpu=1; "cores"→coresPerNpu=8; "net"→topology(字符串拷入,
     *     注意 16 字节缓冲与手工 '\0');
     *   "freq"→coreClockGhz=1.0; "aicpuFreq"→aicpuClockGhz=2.0(GEMM_1 未用);
     *   "cubeTf"→cubePeakTflops=320; "vecTf"→vectorPeakTflops=16;
     *   "l0Kb"→l0SizeKb=64; "ubKb"→ubSizeKb=256(切块预算根);
     *   "l2Mb"/"hbmGb"/"l2Bw"→只服务容量告警与 L2 说明, GEMM_1 不参与仿真;
     *   "hbmBw"→hbmBandwidthTBs=1.2(MTE2/MTE3 时间账的带宽, 单位 TB/s,
     *     调用 cyclesForByteTransfer 时才 ×1e12 变 B/s);
     *   "l0Bw"→l0BandwidthTBsPerCore=2.0(MTE1 每核带宽, 同 ×1e12 用法);
     *   "mte1Lat"→15 / "mte2Lat"→40 / "mte3Lat"→40(直接以 cycle 计,
     *     在 sim.c 的 mte1/2/3*CycleCost 里与带宽账相加);
     *   "aicBusBw"/"aicBusLatUs"/"hccsBw"/"hccsLatUs"/"roceBw"/"roceLatUs"
     *   → 仅跨卡/告警, GEMM_1 单卡不触达。 */
    hwJson = jsonGetMember(requestJson, "hw");
    if (!hwJson)
        return;

    hardware->numNpu = jsonGetInt(hwJson, "npus", hardware->numNpu);
    hardware->coresPerNpu = jsonGetInt(hwJson, "cores", hardware->coresPerNpu);
    {
        const char *topologyText = jsonGetString(hwJson, "net", "ring");
        strncpy(hardware->topology, topologyText, 15);
        hardware->topology[15] = '\0';
    }
    hardware->coreClockGhz      = jsonGetNumber(hwJson, "freq",      hardware->coreClockGhz);
    hardware->aicpuClockGhz     = jsonGetNumber(hwJson, "aicpuFreq", hardware->aicpuClockGhz);
    hardware->cubePeakTflops    = jsonGetNumber(hwJson, "cubeTf",    hardware->cubePeakTflops);
    hardware->vectorPeakTflops  = jsonGetNumber(hwJson, "vecTf",     hardware->vectorPeakTflops);
    hardware->l0SizeKb          = jsonGetNumber(hwJson, "l0Kb",      hardware->l0SizeKb);
    hardware->ubSizeKb          = jsonGetNumber(hwJson, "ubKb",      hardware->ubSizeKb);
    hardware->l2SizeMb          = jsonGetNumber(hwJson, "l2Mb",      hardware->l2SizeMb);
    hardware->hbmSizeGb         = jsonGetNumber(hwJson, "hbmGb",     hardware->hbmSizeGb);
    hardware->l2BandwidthTBs    = jsonGetNumber(hwJson, "l2Bw",      hardware->l2BandwidthTBs);
    hardware->hbmBandwidthTBs   = jsonGetNumber(hwJson, "hbmBw",     hardware->hbmBandwidthTBs);
    hardware->l0BandwidthTBsPerCore = jsonGetNumber(hwJson, "l0Bw",  hardware->l0BandwidthTBsPerCore);
    hardware->mte1LatencyCycles = jsonGetNumber(hwJson, "mte1Lat",   hardware->mte1LatencyCycles);
    hardware->mte2LatencyCycles = jsonGetNumber(hwJson, "mte2Lat",   hardware->mte2LatencyCycles);
    hardware->mte3LatencyCycles = jsonGetNumber(hwJson, "mte3Lat",   hardware->mte3LatencyCycles);
    hardware->aicAicpuBandwidthGBs = jsonGetNumber(hwJson, "aicBusBw",    hardware->aicAicpuBandwidthGBs);
    hardware->aicAicpuLatencyUs    = jsonGetNumber(hwJson, "aicBusLatUs", hardware->aicAicpuLatencyUs);
    hardware->hccsBandwidthGBs  = jsonGetNumber(hwJson, "hccsBw",    hardware->hccsBandwidthGBs);
    hardware->hccsLatencyUs     = jsonGetNumber(hwJson, "hccsLatUs", hardware->hccsLatencyUs);
    hardware->roceBandwidthGbps = jsonGetNumber(hwJson, "roceBw",    hardware->roceBandwidthGbps);
    hardware->roceLatencyUs     = jsonGetNumber(hwJson, "roceLatUs", hardware->roceLatencyUs);

    /* ---- 范围钳制(防止非法输入) ---- */
    if (hardware->numNpu < 1) hardware->numNpu = 1;
    if (hardware->numNpu > 256) hardware->numNpu = 256;
    if (hardware->coresPerNpu < 1) hardware->coresPerNpu = 1;
    if (hardware->coresPerNpu > 256) hardware->coresPerNpu = 256;
    if (hardware->coreClockGhz <= 0) hardware->coreClockGhz = 1.0;
    if (hardware->cubePeakTflops <= 0) hardware->cubePeakTflops = 1;
    if (hardware->vectorPeakTflops <= 0) hardware->vectorPeakTflops = 0.1;
    if (hardware->hbmBandwidthTBs <= 0) hardware->hbmBandwidthTBs = 0.1;
    if (hardware->ubSizeKb <= 8) hardware->ubSizeKb = 8;   /* UB 最小 8KB */
}

/* ============================================================================
 * 单条调度算法解码(默认值 + 范围钳制)
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】decodeOneAlgorithm —— 单条算法的缺省值来源(GEMM_1 全部命中缺省)
 * 调用点: decodeJobConfig 第 6 节, 对 algos 数组逐条解析。GEMM_1 的 algos 只给
 * {id:"dfs", name:"默认参数调度", order:"dfs"}, 于是:
 *   id→algoKey="dfs"(缺省键 "cp" 不触发);
 *   name 缺省回退 id; order 缺省回退 "dfs";
 *   group 缺省 0 ⇒ coresPerKernel=0 ⇒ sim.c 里 groupCores=coresPerNpu=8(auto);
 *   dbuf 缺省 2 ⇒ bufferDepth=2(双缓冲);
 *   maxChunk 缺省 320 ⇒ maxChunksPerCore=320(1..4096 钳制不触发);
 *   ubFrac 缺省 0.66 ⇒ ubCapacityFraction=0.66(0.05..0.95 钳制不触发);
 *   lookahead 缺省 0 ⇒ prefetchWindow=0(预留)。
 * 缺省值 320/0.66/2/0 就是 GEMM_1 切块链的默认"旋钮"位置; 若界面没给任何
 * 数值, 本函数就是让"只用缺省也能跑出 49→32 块"的落点。
 * ==========================================================================*/
static void decodeOneAlgorithm(AlgorithmConfig *algorithm, JsonValue *algorithmJson) {
    const char *key, *displayName, *orderText;

    /* 与原实现保持一致: 算法键/名缺省时统一回退为 "cp" */
    key = jsonGetString(algorithmJson, "id", "cp");
    strncpy(algorithm->algoKey, key, 23);
    algorithm->algoKey[23] = '\0';

    displayName = jsonGetString(algorithmJson, "name", key);
    strncpy(algorithm->algoName, displayName, 79);
    algorithm->algoName[79] = '\0';

    orderText = jsonGetString(algorithmJson, "order", "dfs");
    strncpy(algorithm->issueOrder, orderText, 11);
    algorithm->issueOrder[11] = '\0';

    algorithm->coresPerKernel     = jsonGetInt(algorithmJson, "group", 0);
    algorithm->bufferDepth        = jsonGetInt(algorithmJson, "dbuf", 2);
    algorithm->maxChunksPerCore   = jsonGetInt(algorithmJson, "maxChunk", 320);
    algorithm->ubCapacityFraction = jsonGetNumber(algorithmJson, "ubFrac", 0.66);
    algorithm->prefetchWindow     = jsonGetInt(algorithmJson, "lookahead", 0);

    if (algorithm->bufferDepth < 1) algorithm->bufferDepth = 1;
    if (algorithm->bufferDepth > 4) algorithm->bufferDepth = 4;
    if (algorithm->maxChunksPerCore < 1) algorithm->maxChunksPerCore = 1;
    if (algorithm->maxChunksPerCore > 4096) algorithm->maxChunksPerCore = 4096;
    if (algorithm->ubCapacityFraction <= 0.05) algorithm->ubCapacityFraction = 0.05;
    if (algorithm->ubCapacityFraction > 0.95) algorithm->ubCapacityFraction = 0.95;
}

/* ============================================================================
 * 请求解码主入口(EngineJob)
 * 返回 0 表示成功; 失败时返回 1 并在 job->errorMessage 中写明原因。
 * ==========================================================================*/
/* ============================================================================
 * **【GEMM_1】decodeJobConfig 主流程(GEMM_1 视角逐段)
 * 1. decodeHardwareConfig —— 见上(1卡×8核/1GHz/320T)。
 * 2/4. 模型默认与模型结构读取: ops 模式下这些值照读但 buildWorkloadKernels 不建模,
 *    对 GEMM_1 唯一有意义的模型字段是 maxTileSplitsPerKernel(tileSplit=32)。
 * 3. 激励来源: src.mode="ops" ⇒ job->stimulus.mode=STIMULUS_OPERATORS;
 *    repeat 缺省 1(钳 1..64)。
 * 5. 并行策略 tp/pp/dp/cp/ep: GEMM_1 未给 ⇒ 全取缺省 1(概念上 TP=DP=EP=CP=PP=1,
 *    与贯穿用例定义一致); moe/zero/act/offload 全部走缺省关闭。
 * 6. 算法列表: algos 非空(1 条) ⇒ 不走内置缺省回退, 直接 decodeOneAlgorithm ×1。
 * 7. view: GEMM_1 给了 {algo:0,npu:0,core0:0,core1:7,bins:80} ⇒ detailView
 *    覆盖成 0/0/0/7/80(泳道取 0 号卡 8 核、80 个时间分箱)。
 * 8. 跨页一致性: parallelProduct=1×1×1×1×1=1 ≤ npus=1; PP=1≤L; TP=1≤8;
 *    algos 的 group=0≤8 —— 全部通过 ⇒ GEMM_1 解码成功、无告警。
 * ==========================================================================*/
int decodeJobConfig(EngineJob *job, JsonValue *requestJson) {
    ModelConfig *model;
    JsonValue *srcJson;
    int success = 0;

    logxTraceEnter(LOG_LEVEL_INFO, "decodeJobConfig", NULL);

    memset(job, 0, sizeof(*job));

    /* ---- 1. 硬件配置 ---- */
    decodeHardwareConfig(&job->hardware, requestJson);

    /* ---- 2. 模型默认参数(不论激励模式都会填充, ops 模式不参与建模) ---- */
    model = &job->model;
    model->hiddenDim       = 4096;
    model->numLayers       = 32;
    model->numHeads        = 32;
    model->numKvHeads      = 8;
    model->vocabSize       = 128256;
    model->seqLen          = 4096;
    model->microBatch      = 1;
    model->bytesPerElement = 2;
    model->tensorParallel  = 1;
    model->pipelineParallel = 1;
    model->dataParallel    = 1;
    model->contextParallel = 1;
    model->expertParallel  = 1;
    strcpy(model->presetName, "custom");
    strcpy(model->zeroStage, "off");
    strcpy(model->actCheckpoint, "none");
    strcpy(model->cpuOffload, "none");
    job->ffnWidth      = 14336;      /* 自定义模型 FFN 宽度默认值 */
    job->microBatchWaves = 1;
    job->detailView.algorithmIndex = 0;
    job->detailView.npuIndex       = 0;
    job->detailView.firstCore      = 0;
    job->detailView.lastCore       = 15;
    job->detailView.timeBinCount   = 520;

    /* ---- 3. 激励来源 ---- */
    srcJson = jsonGetMember(requestJson, "src");
    if (!srcJson) {
        strcpy(job->errorMessage, "缺少 src 配置节");
        logxLog(LOG_LEVEL_ERROR, "请求缺少 src 配置节, 解码失败");
        goto done;
    }
    {
        const char *modeText = jsonGetString(srcJson, "mode", "model");
        job->stimulus.mode = strcmp(modeText, "ops") ? STIMULUS_MODEL : STIMULUS_OPERATORS;
    }
    job->stimulus.repeat = jsonGetInt(srcJson, "repeat", 1);
    if (job->stimulus.repeat < 1)
        job->stimulus.repeat = 1;
    if (job->stimulus.repeat > 64)
        job->stimulus.repeat = 64;

    /* ---- 4. 模型结构参数(两模式都会读取; ops 模式下被忽略) ---- */
    model->hiddenDim       = jsonGetNumber(srcJson, "h",       model->hiddenDim);
    model->numLayers       = jsonGetNumber(srcJson, "L",       model->numLayers);
    model->numHeads        = jsonGetNumber(srcJson, "heads",   model->numHeads);
    model->numKvHeads      = jsonGetNumber(srcJson, "kvHeads", model->numKvHeads);
    model->vocabSize       = jsonGetNumber(srcJson, "V",       model->vocabSize);
    model->seqLen          = jsonGetNumber(srcJson, "s",       model->seqLen);
    model->microBatch      = jsonGetNumber(srcJson, "mb",      model->microBatch);
    model->bytesPerElement = jsonGetNumber(srcJson, "prec",    model->bytesPerElement);
    if (model->microBatch < 1)
        model->microBatch = 1;
    if (model->microBatch > 256)
        model->microBatch = 256;
    job->ffnWidth = jsonGetNumber(srcJson, "ff", job->ffnWidth);
    if (job->ffnWidth <= 0)
        job->ffnWidth = 1;

    /* ---- 5. 并行策略 / MoE / 优化手段 ---- */
    model->tensorParallel   = jsonGetInt(srcJson, "tp", model->tensorParallel);
    model->pipelineParallel = jsonGetInt(srcJson, "pp", model->pipelineParallel);
    model->dataParallel     = jsonGetInt(srcJson, "dp", model->dataParallel);
    model->contextParallel  = jsonGetInt(srcJson, "cp", model->contextParallel);
    model->expertParallel   = jsonGetInt(srcJson, "ep", model->expertParallel);
    {
        const char *zeroText = jsonGetString(srcJson, "zero", "off");
        strncpy(model->zeroStage, zeroText, 7);
        model->zeroStage[7] = '\0';
    }
    {
        const char *actText = jsonGetString(srcJson, "act", "none");
        strncpy(model->actCheckpoint, actText, 7);
        model->actCheckpoint[7] = '\0';
    }
    {
        const char *offloadText = jsonGetString(srcJson, "offload", "none");
        strncpy(model->cpuOffload, offloadText, 7);
        model->cpuOffload[7] = '\0';
    }
    {
        const char *presetText = jsonGetString(srcJson, "model", "custom");
        strncpy(model->presetName, presetText, 31);
        model->presetName[31] = '\0';
    }
    /* MoE 与 Tile 切分(界面新增参数) */
    model->enableMoe = jsonGetInt(srcJson, "moe", 0);
    model->numExperts = jsonGetNumber(srcJson, "experts", 0);
    if (model->numExperts < 0) model->numExperts = 0;
    model->topKSelected = jsonGetNumber(srcJson, "topk", 0);
    if (model->topKSelected < 0) model->topKSelected = 0;
    model->expertFfnDim = jsonGetNumber(srcJson, "ffn_exp", 0);
    if (model->expertFfnDim < 0) model->expertFfnDim = 0;
    model->maxTileSplitsPerKernel = jsonGetInt(srcJson, "tileSplit", 32);
    if (model->maxTileSplitsPerKernel < 1) model->maxTileSplitsPerKernel = 1;

    /* 基础范围钳制 */
    if (model->tensorParallel < 1)   model->tensorParallel = 1;
    if (model->pipelineParallel < 1) model->pipelineParallel = 1;
    if (model->numHeads < 1)         model->numHeads = 1;
    if (model->numKvHeads < 1)       model->numKvHeads = 1;
    if (model->seqLen < 1)           model->seqLen = 1;
    if (model->hiddenDim < 1)        model->hiddenDim = 1;
    if (model->numLayers < 1)        model->numLayers = 1;

    /* ---- 6. 调度算法列表 ---- */
    {
        JsonValue *algosJson = jsonGetMember(requestJson, "algos");
        int rawCount = jsonArrayLength(algosJson);   /* 缺失/空数组时为 0 */
        int algorithmIndex;

        if (rawCount > MAX_ALGORITHMS)
            rawCount = MAX_ALGORITHMS;
        if (rawCount < 1) {
            /* 未提供算法时给一个内置默认算法, 保证始终可运行 */
            strcpy(job->algorithms[0].algoKey, "cp");
            strcpy(job->algorithms[0].algoName, "关键路径模调度");
            job->algorithms[0].coresPerKernel   = 0;
            job->algorithms[0].bufferDepth      = 3;
            job->algorithms[0].maxChunksPerCore = 320;
            job->algorithms[0].ubCapacityFraction = 0.66;
            job->algorithms[0].prefetchWindow   = 0;
            job->algorithmCount = 1;
        } else {
            job->algorithmCount = rawCount;
            for (algorithmIndex = 0; algorithmIndex < rawCount; algorithmIndex++)
                decodeOneAlgorithm(&job->algorithms[algorithmIndex],
                                   algosJson->u.array.items[algorithmIndex]);
        }
    }

    /* ---- 7. 泳道详情视图参数 ---- */
    {
        JsonValue *viewJson = jsonGetMember(requestJson, "view");
        if (viewJson) {
            job->detailView.algorithmIndex = jsonGetInt(viewJson, "algo", 0);
            job->detailView.npuIndex       = jsonGetInt(viewJson, "npu", 0);
            job->detailView.firstCore      = jsonGetInt(viewJson, "core0", 0);
            job->detailView.lastCore       = jsonGetInt(viewJson, "core1", 15);
            job->detailView.timeBinCount   = jsonGetInt(viewJson, "bins", 520);
        }
        if (job->detailView.lastCore < job->detailView.firstCore)
            job->detailView.lastCore = job->detailView.firstCore;
        if (job->detailView.timeBinCount < 60)
            job->detailView.timeBinCount = 60;
        if (job->detailView.timeBinCount > 3000)
            job->detailView.timeBinCount = 3000;
        if (job->detailView.npuIndex < 0)
            job->detailView.npuIndex = 0;
        if (job->detailView.npuIndex >= job->hardware.numNpu)
            job->detailView.npuIndex = job->hardware.numNpu - 1;
    }

    /* ============ 8. 跨页一致性校验 ============ */
    {
        const HardwareConfig *hardware = &job->hardware;
        const ModelConfig    *modelCfg = &job->model;
        long long parallelProduct =
            (long long)modelCfg->tensorParallel * modelCfg->pipelineParallel *
            modelCfg->dataParallel * modelCfg->contextParallel * modelCfg->expertParallel;
        int algorithmIndex;

        /* PP 不能超过层数: 每个流水级至少 1 层 */
        if (modelCfg->pipelineParallel > (int)modelCfg->numLayers) {
            snprintf(job->errorMessage, sizeof(job->errorMessage),
                     "PP=%d 超过模型层数 L=%d, 每个流水级至少需要 1 层(请减小 PP 或增大 L)",
                     modelCfg->pipelineParallel, (int)modelCfg->numLayers);
            goto done;
        }
        /* TP 不能超过单卡核数 */
        if (modelCfg->tensorParallel > hardware->coresPerNpu) {
            snprintf(job->errorMessage, sizeof(job->errorMessage),
                     "TP=%d 超过每卡 AI Core 核数 %d(张量并行度不能大于单卡核数)",
                     modelCfg->tensorParallel, hardware->coresPerNpu);
            goto done;
        }
        /* 并行域乘积不能超过算卡数 */
        if (parallelProduct > hardware->numNpu) {
            snprintf(job->errorMessage, sizeof(job->errorMessage),
                     "并行域乘积 TP·PP·DP·CP·EP = %lld 超过算卡数 %d(例如 %d 张卡配 TP=%d 不合法), 请调整硬件算卡数或并行参数",
                     parallelProduct, hardware->numNpu, hardware->numNpu,
                     modelCfg->tensorParallel);
            goto done;
        }
        /* EP>1 但未启用 MoE: 告警不阻断 */
        if (modelCfg->expertParallel > 1 && !(modelCfg->enableMoe && modelCfg->numExperts > 0)) {
            snprintf(job->warningMessage, sizeof(job->warningMessage),
                     "EP=%d 但模型未启用 MoE, 专家并行参数被忽略", modelCfg->expertParallel);
            logxLog(LOG_LEVEL_WARN, "%s", job->warningMessage);
        }
        /* 各调度算法的核绑定不能超过单卡核数 */
        for (algorithmIndex = 0; algorithmIndex < job->algorithmCount; algorithmIndex++) {
            AlgorithmConfig *algorithm = &job->algorithms[algorithmIndex];
            if (algorithm->coresPerKernel > hardware->coresPerNpu) {
                snprintf(job->errorMessage, sizeof(job->errorMessage),
                         "调度算法「%s」使用核数 %d 超过每卡核数 %d, 请调整核绑定策略或硬件核数",
                         algorithm->algoName, algorithm->coresPerKernel,
                         hardware->coresPerNpu);
                goto done;
            }
        }
    }

    success = 1;

done:
    if (success) {
        logxLog(LOG_LEVEL_INFO,
                "配置解码成功: %d卡×%d核 拓扑=%s 主频=%.2fGHz 激励=%s",
                job->hardware.numNpu, job->hardware.coresPerNpu,
                job->hardware.topology, job->hardware.coreClockGhz,
                job->stimulus.mode == STIMULUS_MODEL ? "模型生成" : "算子编排");
    }
    logxTraceLeave(LOG_LEVEL_INFO, "decodeJobConfig",
                   success ? "成功" : "失败: %s", job->errorMessage);
    return success ? 0 : 1;
}
