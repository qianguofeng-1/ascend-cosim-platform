/* ============================================================================
 * run.c - 顶层编排: 一次 /api/run 请求的处理主流程
 *
 *   解码配置 -> 建模生成内核 -> 逐个调度算法跑 Cycle 级仿真
 *   -> 组装"meta + topo + scoreboard + verdict + detail"结果 JSON。
 *
 * 本文件不修改前端可见的 JSON 契约(键名与取值口径不变), 仅负责编排与拼装。
 * ==========================================================================*/
#include "engine.h"
#include <time.h>

/* ============================================================================
 * **【GEMM_1】run.c 视角(结果 JSON 拼装)
 * GEMM_1 请求(单算法 dfs)在本文件的五步:
 *   解码(decodeJobConfig)→ 建模(buildWorkloadKernels)→ 1 次
 *   simulateAlgorithm → winner=0(唯一算法)→ 组装 JSON。
 * 结果 JSON 里每个数字都来自 AlgorithmResult 的某个字段(映射表见 engine.h
 * 【GEMM_1】块), 本文件只做格式搬运, 不改口径; 下面在每处组装点标注
 * "该数值取自哪个 result 字段、GEMM_1 实测值多少"。
 * meta 节的关键值: mode="ops", nkers=1, waves=1, ppActive=0,
 *   Tlb=429,497(job->lowerBoundCycles), freq=1, warn="";
 * scoreboard[0](dfs): totalCycles=431,913 totalUs=431.913(=cycles/(freq×1000)),
 *   score=99.5751, bound="Compute-bound 计算受限";
 * detail 节来自 simulateDetailLanes(0 号卡, 80 分箱): rows=28, bubbles=8,
 *   segsTotal=1280, dag.total=1。
 * ==========================================================================*/

/* ============================================================================
 * 受限类型 -> 显示名(写入 scoreboard.bound)
 * ==========================================================================*/
static const char *bottleneckDisplayName(BottleneckKind bottleneck) {
    switch (bottleneck) {
        case BOUND_COMPUTE:  return "Compute-bound 计算受限";
        case BOUND_MEMORY:   return "Memory-bound 访存受限";
        case BOUND_OVERHEAD: return "Overhead-bound 控制/开销受限";
        default:             return "Balanced 均衡";
    }
}

/* ============================================================================
 * 拓扑 JSON(5.0 视图): 卡列表 + 按拓扑类型生成的互联边
 * ==========================================================================*/
static void appendTopologyJson(EngineJob *job, StringBuffer *out) {
    const HardwareConfig *hardware = &job->hardware;
    int linkCount = 0;
    int npuIndex;

    strbufAppend(out, "\"topo\":{");
    strbufAppendFormat(out, "\"npus\":%d,\"cores\":%d,\"net\":",
                       hardware->numNpu, hardware->coresPerNpu);
    strbufAppendEscapedString(out, hardware->topology);
    strbufAppend(out, ",\"cards\":[");
    for (npuIndex = 0; npuIndex < hardware->numNpu; npuIndex++) {
        if (npuIndex)
            strbufAppendChar(out, ',');
        strbufAppendFormat(out,
            "{\"npu\":%d,\"cubeTf\":%g,\"vecTf\":%g,\"l0Kb\":%g,\"ubKb\":%g,\"l2Mb\":%g,"
            "\"hbmGb\":%g,\"hbmBw\":%g,\"l2Bw\":%g,\"aicBusBw\":%g,\"aicBusLatUs\":%g}",
            npuIndex, hardware->cubePeakTflops, hardware->vectorPeakTflops,
            hardware->l0SizeKb, hardware->ubSizeKb, hardware->l2SizeMb,
            hardware->hbmSizeGb, hardware->hbmBandwidthTBs, hardware->l2BandwidthTBs,
            hardware->aicAicpuBandwidthGBs, hardware->aicAicpuLatencyUs);
    }
    strbufAppend(out, "],\"links\":[");

    if (!strcmp(hardware->topology, "ring") || hardware->numNpu <= 2) {
        /* 环: 每个节点只向“序号更大”的邻居(末节点连回 0)发一条边 */
        for (npuIndex = 0; npuIndex < hardware->numNpu; npuIndex++) {
            int nextNpu = (npuIndex + 1) % hardware->numNpu;
            if (nextNpu == npuIndex)
                continue;
            if (!(nextNpu > npuIndex ||
                  (npuIndex == hardware->numNpu - 1 && nextNpu == 0)))
                continue;
            if (linkCount)
                strbufAppendChar(out, ',');
            linkCount++;
            strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                               npuIndex, nextNpu,
                               hardware->hccsBandwidthGBs, hardware->hccsLatencyUs);
        }
    } else if (!strcmp(hardware->topology, "linear")) {
        /* 直线: npuIndex -> npuIndex+1 */
        for (npuIndex = 0; npuIndex + 1 < hardware->numNpu; npuIndex++) {
            if (linkCount)
                strbufAppendChar(out, ',');
            linkCount++;
            strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                               npuIndex, npuIndex + 1,
                               hardware->hccsBandwidthGBs, hardware->hccsLatencyUs);
        }
    } else if (!strcmp(hardware->topology, "mesh") ||
               !strcmp(hardware->topology, "torus")) {
        /* 网格/环面: 向下、向右两条邻边; torus 另加行列回绕边 */
        int colCount = 1;
        int rowCount;
        int nodeIndex;
        while (colCount * colCount < hardware->numNpu)
            colCount++;
        rowCount = (hardware->numNpu + colCount - 1) / colCount;
        for (nodeIndex = 0; nodeIndex < hardware->numNpu; nodeIndex++) {
            int rowIndex = nodeIndex / colCount;
            int colIndex = nodeIndex % colCount;
            if (rowIndex + 1 < rowCount &&
                nodeIndex + colCount < hardware->numNpu) {
                if (linkCount) strbufAppendChar(out, ',');
                linkCount++;
                strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                                   nodeIndex, nodeIndex + colCount,
                                   hardware->hccsBandwidthGBs, hardware->hccsLatencyUs);
            }
            if (colIndex < colCount - 1 && nodeIndex + 1 < hardware->numNpu) {
                if (linkCount) strbufAppendChar(out, ',');
                linkCount++;
                strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                                   nodeIndex, nodeIndex + 1,
                                   hardware->hccsBandwidthGBs, hardware->hccsLatencyUs);
            }
            if (!strcmp(hardware->topology, "torus")) {   /* 回绕边 */
                if (rowIndex == rowCount - 1 && nodeIndex - colCount >= 0) {
                    if (linkCount) strbufAppendChar(out, ',');
                    linkCount++;
                    strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                                       nodeIndex, nodeIndex - (rowCount - 1) * colCount,
                                       hardware->hccsBandwidthGBs, hardware->hccsLatencyUs);
                }
                if (colIndex == colCount - 1) {
                    if (linkCount) strbufAppendChar(out, ',');
                    linkCount++;
                    strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                                       nodeIndex, nodeIndex - (colCount - 1),
                                       hardware->hccsBandwidthGBs, hardware->hccsLatencyUs);
                }
            }
        }
    } else {
        /* 其余(含 full): 全互联两两成边 */
        int fromNpu, toNpu;
        for (fromNpu = 0; fromNpu < hardware->numNpu; fromNpu++) {
            for (toNpu = fromNpu + 1; toNpu < hardware->numNpu; toNpu++) {
                if (linkCount) strbufAppendChar(out, ',');
                linkCount++;
                strbufAppendFormat(out, "{\"a\":%d,\"b\":%d,\"bwGB\":%g,\"latUs\":%g}",
                                   fromNpu, toNpu, hardware->hccsBandwidthGBs,
                                   hardware->hccsLatencyUs);
            }
        }
    }
    strbufAppend(out,
        "],\"notes\":\"控制通路 Host--PCIe--AI CPU--TS--AI Core; "
        "数据通路 HBM<->L2<->L1/UB<->L0<->Cube/Vector; 卡间 HCCS/RoCE\"}");
}

/* ============================================================================
 * 裁判结论文本(5.4 视图)
 * ==========================================================================*/
static void appendVerdictText(EngineJob *job, const AlgorithmResult *results,
                              int algorithmCount, int winnerIndex, StringBuffer *text) {
    const AlgorithmResult *winner = &results[winnerIndex];
    const char *topology = job->hardware.topology;
    int resultIndex;

    if (job->stimulus.mode == STIMULUS_MODEL) {
        strbufAppendFormat(text,
            "硬件=算卡×AI Core(NPU数%d, 每卡%d核, %s互连, Cube %gTF/卡), "
            "激励=模型前向(经%d类内核×%d层切分/并行)",
            job->hardware.numNpu, job->hardware.coresPerNpu, topology,
            job->hardware.cubePeakTflops, 9, (int)job->model.numLayers);
    } else {
        strbufAppendFormat(text,
            "硬件=算卡×AI Core(%d卡×%d核, %s互连), 激励=自定义算子/计算图(%d内核)",
            job->hardware.numNpu, job->hardware.coresPerNpu, topology, job->kernelCount);
    }
    strbufAppendFormat(text, ", 对比%d种调度: 「%s」综合优度最高(%.1f分/100)",
                       algorithmCount, winner->algoName, winner->compositeScore);

    for (resultIndex = 0; resultIndex < algorithmCount; resultIndex++) {
        const AlgorithmResult *other = &results[resultIndex];
        if (resultIndex == winnerIndex)
            continue;
        strbufAppendFormat(text,
            "; 较「%s」: 气泡占比 %.1f%%vs%.1f%%, 访存掩盖率 %.1f%%vs%.1f%%, Cube利用率 %.1f%%vs%.1f%%",
            other->algoName, winner->bubblePercent, other->bubblePercent,
            100 * winner->memoryMaskRate, 100 * other->memoryMaskRate,
            winner->cubeUtilizationPct, other->cubeUtilizationPct);
    }
    strbufAppend(text, "。归因: ①Tile粒度/并行核数差异使权重分块搬运分散于更多核; ");
    strbufAppend(text, "②指令(微任务)重排与多级缓冲深度决定计算-搬运重叠窗口; ");
    strbufAppend(text, "③访存-计算重叠度高者 DMA 被计算掩盖, 气泡与长尾等待更少。");
}

/* ============================================================================
 * 请求处理入口(runEngineRequest, engine.h 导出)
 * 正常: 返回 0, out 内容为 {"ok":1,...} 完整结果 JSON;
 * 失败: 返回非 0, out 内容为 {"ok":0,"err":"原因"}。
 * ==========================================================================*/
int runEngineRequest(JsonValue *requestJson, StringBuffer *out) {
    EngineJob job;
    AlgorithmResult results[MAX_ALGORITHMS];
    int winnerIndex;
    int algorithmIndex;
    clock_t startClock = clock();

    logxTraceEnter(LOG_LEVEL_INFO, "runEngineRequest", "开始处理请求");

    /* ---- 1. 配置解码 ---- */
    if (decodeJobConfig(&job, requestJson)) {
        strbufAppend(out, "{\"ok\":0,\"err\":");
        strbufAppendEscapedString(out, job.errorMessage);
        strbufAppend(out, "}");
        logxLog(LOG_LEVEL_ERROR, "配置解码失败: %s", job.errorMessage);
        logxTraceLeave(LOG_LEVEL_INFO, "runEngineRequest", "配置解码失败");
        return 1;
    }

    /* ---- 2. 建模 ---- */
    if (buildWorkloadKernels(&job, requestJson)) {
        strbufAppend(out, "{\"ok\":0,\"err\":");
        strbufAppendEscapedString(out, job.errorMessage);
        strbufAppend(out, "}");
        logxLog(LOG_LEVEL_ERROR, "建模失败: %s", job.errorMessage);
        freeJobResources(&job);
        logxTraceLeave(LOG_LEVEL_INFO, "runEngineRequest", "建模失败");
        return 1;
    }

    if (g_cosim_dbg) {
        fprintf(stderr, "DBG run: after work_build nkers=%d cubeF0=%g inB0=%g kers0=%p\n",
                job.kernelCount,
                job.kernels ? job.kernels[0].cubeFlops : 0,
                job.kernels ? job.kernels[0].loadBytes : 0,
                (void *)job.kernels);
    }

    /* ---- 3. 各调度算法逐一仿真 ---- */
    for (algorithmIndex = 0; algorithmIndex < job.algorithmCount; algorithmIndex++) {
        simulateAlgorithm(&job, &job.algorithms[algorithmIndex],
                          &results[algorithmIndex]);
    }

    /* ---- 4. 找出综合优度最高的算法(裁判胜者) ---- */
    /* **【GEMM_1】算法数=1 ⇒ winnerIndex=0(循环不执行); 多算法时取
     * compositeScore 最大者。verdict.winner 输出其 algoKey("dfs")。 */
    winnerIndex = 0;
    for (algorithmIndex = 1; algorithmIndex < job.algorithmCount; algorithmIndex++) {
        if (results[algorithmIndex].compositeScore > results[winnerIndex].compositeScore)
            winnerIndex = algorithmIndex;
    }

    /* ---- 5. 组装结果 JSON ---- */
    strbufAppend(out, "{\"ok\":1");
    strbufAppend(out, ",\"meta\":{\"mode\":");
    strbufAppendEscapedString(out, job.stimulus.mode == STIMULUS_MODEL ? "model" : "ops");
    strbufAppend(out, ",\"srcSummary\":");
    {
        /* 激励摘要文本(展示用, 不参与计算) */
        StringBuffer summary;
        strbufInit(&summary);
        if (job.stimulus.mode == STIMULUS_MODEL) {
            strbufAppendFormat(&summary, "模型激励 %s: h=%g L=%g s=%g mb=%g %s | 并行 TP%d/PP%d/DP%d",
                               job.model.presetName, job.model.hiddenDim, job.model.numLayers,
                               job.model.seqLen, job.model.microBatch,
                               job.model.bytesPerElement >= 1.5 ? "FP16" : "FP8",
                               job.model.tensorParallel, job.model.pipelineParallel,
                               job.model.dataParallel);
            if (job.model.expertParallel > 1)
                strbufAppendFormat(&summary, "/EP%d", job.model.expertParallel);
            if (job.model.contextParallel > 1)
                strbufAppendFormat(&summary, "/CP%d", job.model.contextParallel);
            strbufAppend(&summary,
                " | 前向单步近似(权重 HBM 流式加载, 激活重算/Offload 影响未建模)");
        } else {
            strbufAppendFormat(&summary, "算子激励: 自定义算子/计算图, 共 %d 内核(节点), 每图重复 %d 次",
                               job.kernelCount, job.stimulus.repeat);
        }
        strbufAppendEscapedString(out, summary.data);
        strbufFree(&summary);
    }
    strbufAppendFormat(out, ",\"nkers\":%d,\"waves\":%d,\"ppActive\":%d,\"Tlb\":%g,\"freq\":%g,\"warn\":",
                       job.kernelCount, job.microBatchWaves, job.pipelineActive,
                       job.lowerBoundCycles, job.hardware.coreClockGhz);
    strbufAppendEscapedString(out, job.warningMessage);
    strbufAppend(out, "},");

    /* ---- 5.0 拓扑 ---- */
    /* **【GEMM_1】topo 节: cards 只含 0 号卡(cubeTf=320,vecTf=16,l0Kb=64,
     * ubKb=256,l2Mb=192,hbmGb=64,hbmBw=1.2,l2Bw=3,aicBusBw=64,aicBusLatUs=0.8);
     * links 为空(1 卡不成环)⇒ GEMM_1 无跨卡边。 */
    appendTopologyJson(&job, out);
    strbufAppend(out, ",\"scoreboard\":[");

    /* ---- 5.4 评分板 ---- */
    /* **【GEMM_1】scoreboard 每字段 ← AlgorithmResult(实测值列在右):
     * id/name ← algoKey/algoName("dfs"/"默认参数调度");
     * totalCycles ← totalCycles(431,913); totalUs ← totalCycles/(freq×1000)
     *   =431.913us; bubblePct ← bubblePercent(0.559);
     * maskRate ← memoryMaskRate(0.9961); sramPeakKB ← sramPeakKB(640);
     * lbDev ← loadBalanceDeviation(0);
     * utilCube/utilVec/utilMte/utilMte2/utilMte3 ← 99.44/0/12.08/15.32/8.84;
     * dmaBytes/maskedBytes ← dmaBytes/maskedBytes(1.00663e8/1.0027e8);
     * ldBytes/stBytes ← loadDmaBytes/storeDmaBytes(6.71089e7/3.35544e7);
     * stallCyc ← stallCycles(10,820); cubeBusy/vecBusy ← 3.43597e6/0;
     * dmaBusy ← mte2BusyCycles+mte3BusyCycles(104,366); ctrlBusy ←
     * aicpuBusyCycles(10,240); commBusy ← commBusyCycles(0);
     * theorySerialCyc/pipeCyc ← ops 模式恒 0; dutyMax/Min/Avg ← ops 恒 1.0;
     * score ← compositeScore(99.5751); bound ← bottleneck 显示名
     *   ("Compute-bound 计算受限", 见 bottleneckDisplayName)。 */
    for (algorithmIndex = 0; algorithmIndex < job.algorithmCount; algorithmIndex++) {
        const AlgorithmResult *result = &results[algorithmIndex];
        if (algorithmIndex)
            strbufAppendChar(out, ',');
        strbufAppend(out, "{\"id\":");
        strbufAppendEscapedString(out, result->algoKey);
        strbufAppend(out, ",\"name\":");
        strbufAppendEscapedString(out, result->algoName);
        strbufAppendFormat(out, ",\"totalCycles\":%g", result->totalCycles);
        strbufAppendFormat(out, ",\"totalUs\":%g",
                           result->totalCycles / (job.hardware.coreClockGhz * 1000.0));
        strbufAppendFormat(out, ",\"bubblePct\":%g", result->bubblePercent);
        strbufAppendFormat(out, ",\"maskRate\":%g", result->memoryMaskRate);
        strbufAppendFormat(out, ",\"sramPeakKB\":%g", result->sramPeakKB);
        strbufAppendFormat(out, ",\"lbDev\":%g", result->loadBalanceDeviation);
        strbufAppendFormat(out, ",\"utilCube\":%g", result->cubeUtilizationPct);
        strbufAppendFormat(out, ",\"utilVec\":%g", result->vectorUtilizationPct);
        strbufAppendFormat(out, ",\"utilMte\":%g", result->mteUtilizationPct);
        strbufAppendFormat(out, ",\"utilMte2\":%g", result->mte2UtilizationPct);
        strbufAppendFormat(out, ",\"utilMte3\":%g", result->mte3UtilizationPct);
        strbufAppendFormat(out, ",\"dmaBytes\":%g,\"maskedBytes\":%g",
                           result->dmaBytes, result->maskedBytes);
        strbufAppendFormat(out, ",\"ldBytes\":%g,\"stBytes\":%g",
                           result->loadDmaBytes, result->storeDmaBytes);
        strbufAppendFormat(out, ",\"stallCyc\":%g", result->stallCycles);
        strbufAppendFormat(out, ",\"cubeBusy\":%g", result->cubeBusyCycles);
        strbufAppendFormat(out, ",\"vecBusy\":%g", result->vectorBusyCycles);
        strbufAppendFormat(out, ",\"dmaBusy\":%g",
                           result->mte2BusyCycles + result->mte3BusyCycles);
        strbufAppendFormat(out, ",\"ctrlBusy\":%g", result->aicpuBusyCycles);
        strbufAppendFormat(out, ",\"commBusy\":%g", result->commBusyCycles);
        strbufAppendFormat(out, ",\"theorySerialCyc\":%g", result->serialEquivalentCycles);
        strbufAppendFormat(out, ",\"pipeCyc\":%g", result->pipelineFillCycles);
        strbufAppendFormat(out, ",\"dutyMax\":%g,\"dutyMin\":%g,\"dutyAvg\":%g",
                           result->npuDutyMax, result->npuDutyMin, result->npuDutyAvg);
        strbufAppendFormat(out, ",\"score\":%g", result->compositeScore);
        strbufAppend(out, ",\"bound\":");
        strbufAppendEscapedString(out, bottleneckDisplayName(result->bottleneck));
        strbufAppend(out, "}");
    }
    strbufAppend(out, "]");

    /* ---- 裁判结论 ---- */
    /* **【GEMM_1】verdict.winner=winnerIndex 的 algoKey="dfs"; text 由
     * appendVerdictText 生成, 取 winner->compositeScore(99.6/100)、
     * bubblePercent、memoryMaskRate、cubeUtilizationPct 等, 归因段为固定
     * 文案(①Tile/核数 ②缓冲深度 ③掩盖率) —— GEMM_1 单算法时无对比对手,
     * "较「…」"循环不输出。 */
    strbufAppend(out, ",\"verdict\":{\"winner\":");
    strbufAppendEscapedString(out, results[winnerIndex].algoKey);
    strbufAppend(out, ",\"text\":");
    {
        StringBuffer verdict;
        strbufInit(&verdict);
        appendVerdictText(&job, results, job.algorithmCount, winnerIndex, &verdict);
        strbufAppendEscapedString(out, verdict.data);
        strbufFree(&verdict);
    }
    strbufAppend(out, "}");

    /* ---- 泳道详情 ---- */
    /* **【GEMM_1】detailAlgo=view.algo=0(在算法数内)⇒ simulateDetailLanes
     * 跑同一算法 dfs, npu=0、core0..core1=0..7、bins=80 ⇒ 输出
     * {algo:0,npu:0,T:431913,rows[28],bubbles[8],rowsNote,segs[],segsTotal:1280,
     *  dag{total:1}} —— 供结果页泳道/SVG/拓扑三视图消费。 */
    strbufAppend(out, ",\"detail\":");
    {
        int detailAlgo = job.detailView.algorithmIndex;
        if (detailAlgo < 0 || detailAlgo >= job.algorithmCount)
            detailAlgo = winnerIndex;
        simulateDetailLanes(&job, &job.algorithms[detailAlgo],
                            job.detailView.npuIndex, job.detailView.timeBinCount,
                            job.detailView.firstCore, job.detailView.lastCore,
                            detailAlgo, out);
    }
    strbufAppend(out, "}");

    freeJobResources(&job);

    {
        double elapsedMs = 1000.0 * (double)(clock() - startClock) / CLOCKS_PER_SEC;
        logxLog(LOG_LEVEL_INFO,
                "请求处理完成: 算法数=%d 胜者=%s 耗时=%.1fms",
                job.algorithmCount,
                job.algorithmCount > 0 ? results[winnerIndex].algoName : "-",
                elapsedMs);
    }
    logxTraceLeave(LOG_LEVEL_INFO, "runEngineRequest", "ok");
    logxFlush();
    return 0;
}
