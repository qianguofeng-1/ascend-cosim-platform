/* ============================================================================
 * work.c - 建模: 把“激励”展开为逻辑内核序列(buildWorkloadKernels)
 *
 * 两种激励来源:
 *   (a) 模型生成(model): 按 Transformer 结构逐层生成 9 类算子内核,
 *       PP 阶段按层切分并分配到不同算卡; MoE 模式生成 Router+专家 FFN;
 *   (b) 算子编排(ops): 单算子 / 链式(自动 前一算子末内核 -> 后一算子首内核) /
 *       DAG(依赖边显式指定),
 *       每个算子的放置算卡可由 op.npu 指定, autoNpu=1 时自动轮询分配。
 *
 * 产物: EngineJob.kernels(WorkKernel 数组) —— sim.c 的调度与仿真输入。
 * ==========================================================================*/
#include "engine.h"

/* ============================================================================
 * **【GEMM_1】本文件视角(建模)
 * GEMM_1 的激励是 ops 模式的单 GEMM, 因此只会走到 buildOperatorGraphStimulus()
 * (model 分支不触发)。本文件对 GEMM_1 的产物是恰好 1 个 WorkKernel:
 *   name="op0-GEMM", kind=KERNEL_CUBE, npuIndex=0,
 *   cubeFlops=2·4096³≈1.3743895e11, loadBytes=67,108,864, storeBytes=33,554,432,
 *   predecessorCount=0(edges=[])。随后 buildWorkloadKernels 按"每卡并行上限"
 *   算出理论下界 Tlb≈429,497 cyc 存入 job->lowerBoundCycles(效率评分分母)。
 * 批注点: ① appendOperatorKernel 的 gemm 分支(m/n/k→rows/cols/reduceDim、
 * bpp、三条公式); ② 单内核无依赖边的放置与依赖处理(buildOperatorGraphStimulus);
 * ③ buildWorkloadKernels 的理论下界与告警段。
 * ==========================================================================*/

/* ============================================================================
 * 内部辅助类型
 * ==========================================================================*/

/* 内核动态数组: 建模过程中不断追加新内核 */
typedef struct {
    WorkKernel *items;     /* 内核数组(heap) */
    int         count;     /* 已有内核个数   */
    int         capacity;  /* 数组容量       */
} KernelBuilder;

/* 算子在展开图中的首个/末个内核下标(用于连依赖边) */
typedef struct {
    int firstKernelIndex;  /* 该算子展开出的第一个内核 */
    int lastKernelIndex;   /* 该算子展开出的最后一个内核 */
} OpKernelRange;

/* ---------------- 内核数组操作 ---------------- */

/* 追加一个内核(结构体值拷贝) */
static void builderAppendKernel(KernelBuilder *builder, WorkKernel kernel) {
    if (builder->count == builder->capacity) {
        builder->capacity = builder->capacity ? builder->capacity * 2 : 4096;
        builder->items = (WorkKernel *)realloc(builder->items,
                                               (size_t)builder->capacity * sizeof(WorkKernel));
    }
    builder->items[builder->count++] = kernel;
}

/* 给 dependentIndex 增加一条依赖: 它必须等 predecessorIndex 完成后才能开始。
 * 自动去重并限制前驱数量(MAX_PREDECESSORS)。 */
static void builderAddDependency(KernelBuilder *builder, int predecessorIndex, int dependentIndex) {
    WorkKernel *dependent;
    int predecessorSlot;

    /* 非法/自依赖/越界直接忽略 */
    if (dependentIndex == predecessorIndex || predecessorIndex < 0 || dependentIndex < 0)
        return;
    if (predecessorIndex >= builder->count || dependentIndex >= builder->count)
        return;
    dependent = &builder->items[dependentIndex];
    for (predecessorSlot = 0; predecessorSlot < dependent->predecessorCount;
         predecessorSlot++) {
        if (dependent->predecessors[predecessorSlot] == predecessorIndex)
            return;                     /* 已有该依赖, 去重 */
    }
    if (dependent->predecessorCount < MAX_PREDECESSORS)
        dependent->predecessors[dependent->predecessorCount++] = predecessorIndex;
}

/* 给内核设置名称(带长度保护) */
static void assignKernelName(WorkKernel *kernel, const char *name) {
    strncpy(kernel->name, name, sizeof(kernel->name) - 1);
    kernel->name[sizeof(kernel->name) - 1] = '\0';
}

/* ============================================================================
 * 单层 Transformer 展开为 9 类内核(非 MoE 的稠密 FFN 结构)
 * 内核顺序: QKV -> Attn(Mix) -> OutProj -> LN1 -> MLP-Up -> MLP-Gate
 *           -> GELU -> Down(MLP输出投影) -> LN2
 * ==========================================================================*/
static void buildTransformerLayerKernels(KernelBuilder *builder, int globalLayerNo,
                                         double hiddenDim, double seqLen,
                                         double numHeads, double numKvHeads,
                                         double ffnWidth, double bytesPerElement,
                                         int npuIndex, const char *nameTag) {
    double headDim = hiddenDim / numHeads;
    double activationBytes = seqLen * hiddenDim * bytesPerElement;
    WorkKernel kernel;
    char kernelName[160];

    /* 1) QKV 投影: 矩阵乘, 权重 + 激活输入 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 2 * seqLen * hiddenDim * (hiddenDim + 2 * numKvHeads * headDim);
    kernel.loadBytes = hiddenDim * (hiddenDim + 2 * numKvHeads * headDim) * bytesPerElement
                       + activationBytes;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-QKV", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 2) Attention(Mix): QK^T 与 AV 用 Cube, 掩码/Softmax 用 Vector */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_MIXED;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 4 * seqLen * seqLen * hiddenDim;
    kernel.vectorFlops = 10 * numKvHeads * seqLen * seqLen;
    kernel.loadBytes = seqLen * hiddenDim * bytesPerElement
                       + numKvHeads * seqLen * seqLen * bytesPerElement;
    kernel.storeBytes = numKvHeads * seqLen * seqLen * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-Attn", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 3) 输出投影 OutProj */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 2 * seqLen * hiddenDim * hiddenDim;
    kernel.loadBytes = hiddenDim * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-OutProj", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 4) 第一个 LayerNorm(LN1) */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_VECTOR;
    kernel.npuIndex = npuIndex;
    kernel.vectorFlops = 8 * seqLen * hiddenDim;
    kernel.loadBytes = seqLen * hiddenDim * bytesPerElement;
    kernel.storeBytes = seqLen * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-LN1", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 5)~6) MLP 上投影: Up / Gate 两个独立的矩阵乘 */
    {
        int projIndex;
        const char *projLabels[2] = { "Up", "Gate" };
        for (projIndex = 0; projIndex < 2; projIndex++) {
            memset(&kernel, 0, sizeof(kernel));
            kernel.kind = KERNEL_CUBE;
            kernel.npuIndex = npuIndex;
            kernel.cubeFlops = 2 * seqLen * hiddenDim * ffnWidth;
            kernel.loadBytes = hiddenDim * ffnWidth * bytesPerElement;
            snprintf(kernelName, sizeof(kernelName), "%sL%d-MLP-%s",
                     nameTag, globalLayerNo, projLabels[projIndex]);
            assignKernelName(&kernel, kernelName);
            builderAppendKernel(builder, kernel);
        }
    }

    /* 7) GELU 激活 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_VECTOR;
    kernel.npuIndex = npuIndex;
    kernel.vectorFlops = 8 * seqLen * ffnWidth;
    kernel.loadBytes = seqLen * ffnWidth * bytesPerElement;
    kernel.storeBytes = seqLen * ffnWidth * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-GELU", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 8) MLP 下投影 Down */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 2 * seqLen * ffnWidth * hiddenDim;
    kernel.loadBytes = ffnWidth * hiddenDim * bytesPerElement;
    kernel.storeBytes = seqLen * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-Down", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 9) 第二个 LayerNorm(LN2) */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_VECTOR;
    kernel.npuIndex = npuIndex;
    kernel.vectorFlops = 8 * seqLen * hiddenDim;
    kernel.loadBytes = seqLen * hiddenDim * bytesPerElement;
    kernel.storeBytes = seqLen * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "%sL%d-LN2", nameTag, globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);
}

/* ============================================================================
 * MoE 层展开(界面结构: QKV/Attn/OutProj/LN1/Router/专家FFN/LN2)
 * 近似假设: 全部专家权重载入(loadBytes 含 E 份), 计算量按 TopK 稀疏折算。
 * ==========================================================================*/
static void buildMoeLayerKernels(KernelBuilder *builder, int globalLayerNo,
                                 double hiddenDim, double seqLen,
                                 double numHeads, double numKvHeads,
                                 double numExperts, double topKSelected,
                                 double expertFfnDim, double bytesPerElement,
                                 int npuIndex) {
    double headDim = hiddenDim / numHeads;
    double activationBytes = seqLen * hiddenDim * bytesPerElement;
    WorkKernel kernel;
    char kernelName[160];

    /* 1) QKV 投影 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 2 * seqLen * hiddenDim * (hiddenDim + 2 * numKvHeads * headDim);
    kernel.loadBytes = hiddenDim * (hiddenDim + 2 * numKvHeads * headDim) * bytesPerElement
                       + activationBytes;
    snprintf(kernelName, sizeof(kernelName), "L%d-QKV", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 2) Attention(Mix) */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_MIXED;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 4 * seqLen * seqLen * hiddenDim;
    kernel.vectorFlops = 10 * numKvHeads * seqLen * seqLen;
    kernel.loadBytes = seqLen * hiddenDim * bytesPerElement
                       + numKvHeads * seqLen * seqLen * bytesPerElement;
    kernel.storeBytes = numKvHeads * seqLen * seqLen * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "L%d-Attn", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 3) 输出投影 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 2 * seqLen * hiddenDim * hiddenDim;
    kernel.loadBytes = hiddenDim * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "L%d-OutProj", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 4) LN1 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_VECTOR;
    kernel.npuIndex = npuIndex;
    kernel.vectorFlops = 8 * seqLen * hiddenDim;
    kernel.loadBytes = seqLen * hiddenDim * bytesPerElement;
    kernel.storeBytes = seqLen * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "L%d-LN1", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 5) Router: 为每个 token 在所有专家上打分 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 2 * seqLen * hiddenDim * numExperts;
    kernel.loadBytes = hiddenDim * numExperts * bytesPerElement;
    kernel.storeBytes = seqLen * numExperts * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "L%d-Router", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 6) MoE 专家 FFN(聚合近似: 全部专家权重载入, 计算按 topk 稀疏) */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_CUBE;
    kernel.npuIndex = npuIndex;
    kernel.cubeFlops = 6 * seqLen * topKSelected * hiddenDim * expertFfnDim;
    kernel.loadBytes = 3 * hiddenDim * expertFfnDim * numExperts * bytesPerElement;
    kernel.storeBytes = seqLen * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "L%d-MoE", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);

    /* 7) LN2 */
    memset(&kernel, 0, sizeof(kernel));
    kernel.kind = KERNEL_VECTOR;
    kernel.npuIndex = npuIndex;
    kernel.vectorFlops = 8 * seqLen * hiddenDim;
    kernel.loadBytes = seqLen * hiddenDim * bytesPerElement;
    kernel.storeBytes = seqLen * hiddenDim * bytesPerElement;
    snprintf(kernelName, sizeof(kernelName), "L%d-LN2", globalLayerNo);
    assignKernelName(&kernel, kernelName);
    builderAppendKernel(builder, kernel);
}

/* ============================================================================
 * 模型激励展开(顶层): 按 PP 阶段切分层区间, 逐层生成内核
 * ==========================================================================*/
static int buildModelStimulus(EngineJob *job, KernelBuilder *builder) {
    const ModelConfig *model = &job->model;
    double bytesPerElement = model->bytesPerElement >= 1.5 ? 2.0 : 1.0;
    double ffnWidth = job->ffnWidth;
    int pipelineParallel = model->pipelineParallel;
    int moeEnabled;
    int totalLayers, layersPerStage, stage, globalLayer;
    double microBatchValue;

    if (pipelineParallel > job->hardware.numNpu)   /* 阶段数不能超过算卡数 */
        pipelineParallel = job->hardware.numNpu;
    job->pipelineActive = pipelineParallel > 1 ? 1 : 0;

    microBatchValue = model->microBatch;
    if (microBatchValue < 1)
        microBatchValue = 1;
    job->microBatchWaves = (int)ceil(microBatchValue);
    if (job->microBatchWaves > 256)
        job->microBatchWaves = 256;

    /* MoE 生效需同时开启开关并给出专家数/TopK/专家FFN宽 */
    moeEnabled = (model->enableMoe && model->numExperts > 0 &&
                  model->topKSelected > 0 && model->expertFfnDim > 0);

    totalLayers = (int)model->numLayers;
    layersPerStage = (totalLayers + pipelineParallel - 1) / pipelineParallel;

    for (stage = 0; stage < pipelineParallel; stage++) {
        int layerBegin = stage * layersPerStage;
        int layerEnd = layerBegin + layersPerStage;
        if (layerEnd > totalLayers)
            layerEnd = totalLayers;
        for (globalLayer = layerBegin; globalLayer < layerEnd; globalLayer++) {
            if (moeEnabled) {
                buildMoeLayerKernels(builder, globalLayer + 1,
                                     model->hiddenDim, model->seqLen,
                                     model->numHeads, model->numKvHeads,
                                     model->numExperts, model->topKSelected,
                                     model->expertFfnDim, bytesPerElement, stage);
            } else {
                buildTransformerLayerKernels(builder, globalLayer + 1,
                                             model->hiddenDim, model->seqLen,
                                             model->numHeads, model->numKvHeads,
                                             ffnWidth, bytesPerElement, stage, "");
            }
        }
    }
    return 0;
}

/* ============================================================================
 * 算子编排: 单个算子 -> 内核(支持 GEMM/Conv/Attention/MoE/Transformer/...
 * 各类参数键与参考界面兼容, 缺失时取默认值)
 *
 * 【为什么"没有 gemm() 函数/GEMM 模块"】
 * 本平台是时序/资源近似仿真, 不执行算子数值计算: 每个算子在这里只被翻译成
 * 一份 WorkKernel 数据(种类 + FLOPs + 搬运字节 + 放置 + 依赖), 后续调度/仿真
 * (sim.c)按 kernel.kind 统一处理, 因此不存在 gemm/conv 各自的计算实现。
 *   · GEMM/Conv/注意力投影等一律归 KERNEL_CUBE(喂 Cube 的搬运+计算);
 *   · LN/Softmax/GELU 归 KERNEL_VECTOR; FlashAttention 类归 KERNEL_MIXED;
 *   · α/β 缩放、数据布局/精度等"数值语义"不影响 FLOPs/字节/依赖 => 不建模
 *     (β≠0 需读回旧 C 的流量也不会体现, 与 README"近似与未建模"口径一致)。
 * 用户可见的 "gemm" 只是 JSON 里的算子类型字符串(本文件唯一出现处)。
 * ==========================================================================*/
static void appendOperatorKernel(KernelBuilder *builder, JsonValue *opJson,
                                 int opIndex, int npuIndex, double bytesPerElement) {
    const char *opKind = jsonGetString(opJson, "op", "gemm");
    const char *opLabel = jsonGetString(opJson, "label", opKind);
    WorkKernel kernel;
    char kernelName[160];

    /* 注意: kernel.npuIndex 在各分支内赋值, 与原实现逐分支一致:
     * 其中“自定义(custom)”分支在原实现中从不写 npu(恒为 0),
     * 此处保留该历史行为以保证前后结果一致。 */
    memset(&kernel, 0, sizeof(kernel));

    if (!strcmp(opKind, "gemm")) {
        /* ================================================================
         * **【GEMM_1】gemm 分支变量与公式逐项(仅记账, 不执行数值计算)
         * 读入的三个 JSON 键(m/n/k 缺省都是 1024, GEMM_1 给 4096):
         *   matrixRows      = jsonGetNumber(op,"m",1024) = 4096 —— 输出 C 的行
         *                     (即 A 的行数), 语义矩阵乘 C(M×N)=A(M×K)·B(K×N);
         *   matrixColumns   = jsonGetNumber(op,"n",1024) = 4096 —— 输出 C 的列
         *                     (即 B 的列数);
         *   matrixReduceDim = jsonGetNumber(op,"k",1024) = 4096 —— 归约维 K
         *                     (A 的列 / B 的行, 逐 K 累加的部分和)。
         * bpp(bytesPerElement, 形参): 由外层 buildOperatorGraphStimulus 按
         *   src.prec 折算 —— prec=2(≥1.5) ⇒ 2.0B/元素(FP16)。GEMM_1=2.0。
         * 三条"总账"公式(结果进 WorkKernel, 切块在 sim.c 再做):
         *   kernel.cubeFlops = 2·rows·cols·reduceDim —— 一次乘加=2 FLOPs;
         *     = 2×4096×4096×4096 = 2×4096³ ≈ 1.3743895e11(理论下界分子);
         *   kernel.loadBytes = (rows·reduceDim + cols·reduceDim)×bpp —— 读
         *     A(M×K) 与 B(K×N) 两个输入矩阵(不含 C);
         *     = (4096·4096+4096·4096)×2 = 67,108,864B;
         *   kernel.storeBytes = rows·cols×bpp —— 写 C(M×N);
         *     = 4096·4096×2 = 33,554,432B;
         * 内核名 "op<opIndex>-<label>" ⇒ opIndex=0、label="GEMM"
         *   ⇒ "op0-GEMM"(泳道/seg 的 kernelIndex=0 都指向它)。
         * ================================================================ */
        /* 通用矩阵乘 C=A×B (α=1,β=0 语义): 只记账不计算——
         * 计算量 cubeFlops=2·rows·cols·reduceDim(一次乘加=2 FLOPs);
         * 载入 loadBytes=(rows·reduceDim+cols·reduceDim)×字节/元素(读 A、B);
         * 写出 storeBytes=rows·cols×字节/元素(写 C)。
         * rows/cols/reduceDim 取 JSON 键 m/n/k: M×N 为输出 C 的二维形状,
         * K 为 A×B 的归约维。 */
        double matrixRows = jsonGetNumber(opJson, "m", 1024);
        double matrixColumns = jsonGetNumber(opJson, "n", 1024);
        double matrixReduceDim = jsonGetNumber(opJson, "k", 1024);
        kernel.kind = KERNEL_CUBE;
        kernel.npuIndex = npuIndex;
        kernel.cubeFlops = 2 * matrixRows * matrixColumns * matrixReduceDim;
        kernel.loadBytes = (matrixRows * matrixReduceDim +
                            matrixColumns * matrixReduceDim) * bytesPerElement;
        kernel.storeBytes = matrixRows * matrixColumns * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-%s", opIndex, opLabel);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else if (!strcmp(opKind, "conv2d")) {
        /* 二维卷积: 由 H/W/F/Cin/Cout 推算输出特征图尺寸 */
        double batch = jsonGetNumber(opJson, "batch", 1);
        double inChannels = jsonGetNumber(opJson, "Cin", jsonGetNumber(opJson, "c", 256));
        double inHeight = jsonGetNumber(opJson, "H", jsonGetNumber(opJson, "h", 56));
        double inWidth  = jsonGetNumber(opJson, "W", jsonGetNumber(opJson, "w", 56));
        double filterSize = jsonGetNumber(opJson, "K", jsonGetNumber(opJson, "f", 3));
        double outChannels = jsonGetNumber(opJson, "Cout", jsonGetNumber(opJson, "k", 256));
        double padding = jsonGetNumber(opJson, "pad", 1);
        double stride = jsonGetNumber(opJson, "stride", 1);
        double outHeight = floor((inHeight + 2 * padding - filterSize) / stride) + 1;
        double outWidth  = floor((inWidth + 2 * padding - filterSize) / stride) + 1;
        if (outHeight < 1) outHeight = 1;
        if (outWidth < 1) outWidth = 1;
        kernel.kind = KERNEL_CUBE;
        kernel.npuIndex = npuIndex;
        kernel.cubeFlops = 2 * batch * outHeight * outWidth *
                           inChannels * outChannels * filterSize * filterSize;
        kernel.loadBytes = (inChannels * filterSize * filterSize * outChannels +
                            batch * inChannels * inHeight * inWidth) * bytesPerElement;
        kernel.storeBytes = batch * outChannels * outHeight * outWidth * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-%s", opIndex, opLabel);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else if (!strcmp(opKind, "flashattn") || !strcmp(opKind, "attention")) {
        /* 注意力(近似 FlashAttention): QK^T/AV 用 Cube, 其余用 Vector */
        double tokens = jsonGetNumber(opJson, "tokens", jsonGetNumber(opJson, "s", 4096));
        double numHeads = jsonGetNumber(opJson, "heads", 32);
        double headDim = jsonGetNumber(opJson, "d", 128);
        JsonValue *hiddenJson = jsonGetMember(opJson, "h");
        double hiddenDim = (hiddenJson && hiddenJson->type == JSON_NUMBER)
                               ? hiddenJson->u.numberValue : numHeads * headDim;
        double numKvHeads = jsonGetNumber(opJson, "kvHeads", numHeads);
        kernel.kind = KERNEL_MIXED;
        kernel.npuIndex = npuIndex;
        kernel.cubeFlops = 4 * tokens * tokens * hiddenDim;
        kernel.vectorFlops = 10 * numKvHeads * tokens * tokens;
        kernel.loadBytes = (hiddenDim + 2 * numKvHeads * (hiddenDim / numHeads) +
                            numKvHeads * tokens) * tokens * bytesPerElement;
        kernel.storeBytes = (hiddenDim + numKvHeads * tokens) * tokens * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-%s", opIndex, opLabel);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else if (!strcmp(opKind, "moe")) {
        /* MoE 专家混合层: 计算按 topk 稀疏, 权重按全部专家载入 */
        double tokens = jsonGetNumber(opJson, "tokens", jsonGetNumber(opJson, "s", 1024));
        double hiddenDim = jsonGetNumber(opJson, "h", 4096);
        double numExperts = jsonGetNumber(opJson, "experts", jsonGetNumber(opJson, "nExperts", 64));
        double topK = jsonGetNumber(opJson, "topk", 8);
        double ffnDim = jsonGetNumber(opJson, "ff", jsonGetNumber(opJson, "ffnExp", 2048));
        kernel.kind = KERNEL_CUBE;
        kernel.npuIndex = npuIndex;
        kernel.cubeFlops = 4 * tokens * topK * hiddenDim * ffnDim;
        kernel.loadBytes = (numExperts * 2 * hiddenDim * ffnDim + tokens * hiddenDim)
                           * bytesPerElement;
        kernel.storeBytes = tokens * hiddenDim * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-%s", opIndex, opLabel);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else if (!strcmp(opKind, "transformer") || !strcmp(opKind, "transformerblock")) {
        /* Transformer Block: 复用单层展开, 逐层追加; 层号用 opIndex 打底避免撞名 */
        double seqLen = jsonGetNumber(opJson, "s", 4096);
        double hiddenDim = jsonGetNumber(opJson, "h", 4096);
        double numHeads = jsonGetNumber(opJson, "heads",
                                        jsonGetNumber(opJson, "H", floor(hiddenDim / 128)));
        double numKvHeads = jsonGetNumber(opJson, "kvHeads",
                                          jsonGetNumber(opJson, "Hkv", numHeads));
        double ffnWidth = jsonGetNumber(opJson, "ff", jsonGetNumber(opJson, "ffn", hiddenDim * 3.5));
        double numLayers = jsonGetNumber(opJson, "layers", 1);
        int layerIndex;
        for (layerIndex = 0; layerIndex < (int)numLayers; layerIndex++) {
            buildTransformerLayerKernels(builder, opIndex * 100000 + layerIndex + 1,
                                         hiddenDim, seqLen, numHeads, numKvHeads,
                                         ffnWidth, bytesPerElement, npuIndex, opLabel);
        }

    } else if (!strcmp(opKind, "layernorm")) {
        double elemCount = jsonGetNumber(opJson, "num", jsonGetNumber(opJson, "N", 4096 * 4096));
        kernel.kind = KERNEL_VECTOR;
        kernel.npuIndex = npuIndex;
        kernel.vectorFlops = 8 * elemCount;
        kernel.loadBytes = elemCount * bytesPerElement;
        kernel.storeBytes = elemCount * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-LN", opIndex);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else if (!strcmp(opKind, "softmax")) {
        double elemCount = jsonGetNumber(opJson, "num", jsonGetNumber(opJson, "N", 4096 * 4096));
        kernel.kind = KERNEL_VECTOR;
        kernel.npuIndex = npuIndex;
        kernel.vectorFlops = 10 * elemCount;
        kernel.loadBytes = elemCount * bytesPerElement;
        kernel.storeBytes = elemCount * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-Softmax", opIndex);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else if (!strcmp(opKind, "gelu")) {
        double elemCount = jsonGetNumber(opJson, "num", jsonGetNumber(opJson, "N", 4096 * 4096));
        kernel.kind = KERNEL_VECTOR;
        kernel.npuIndex = npuIndex;
        kernel.vectorFlops = 8 * elemCount;
        kernel.loadBytes = elemCount * bytesPerElement;
        kernel.storeBytes = elemCount * bytesPerElement;
        snprintf(kernelName, sizeof(kernelName), "op%d-GELU", opIndex);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);

    } else {
        /* 自定义算子: unit=AIV 时按向量处理; ckind 显式指定 vec/dma/cube;
         * 提供 FLOPs/字节 参数直接构造内核。
         * 注意: 此处【刻意不】写 kernel.npuIndex —— 与原实现一致
         * (原实现自定义分支漏写 npu, 恒为 0, 即固定落在 0 号卡)。
         * 如需修正放置语义, 请同步评估对既有结果的影响。 */
        const char *unitText = jsonGetString(opJson, "unit", NULL);
        const char *computeKind = jsonGetString(
            opJson, "ckind",
            (unitText && !strcmp(unitText, "AIV")) ? "vec" : "cube");
        if (!strcmp(computeKind, "vec"))
            kernel.kind = KERNEL_VECTOR;
        else if (!strcmp(computeKind, "dma"))
            kernel.kind = KERNEL_DMA;
        else
            kernel.kind = KERNEL_CUBE;

        if (kernel.kind == KERNEL_DMA) {
            double dmaBytes = jsonGetNumber(opJson, "bytes", 1e6);
            kernel.loadBytes = dmaBytes / 2;
            kernel.storeBytes = dmaBytes / 2;
        } else {
            double flops = jsonGetNumber(opJson, "flops", 1e9);
            kernel.loadBytes = jsonGetNumber(opJson, "loadB",
                                             jsonGetNumber(opJson, "inBytes", 1e6));
            kernel.storeBytes = jsonGetNumber(opJson, "storeB",
                                              jsonGetNumber(opJson, "outBytes", 1e6));
            if (kernel.kind == KERNEL_VECTOR)
                kernel.vectorFlops = flops;
            else
                kernel.cubeFlops = flops;
            if (kernel.cubeFlops <= 0 && kernel.vectorFlops <= 0)
                kernel.cubeFlops = 1e9;
        }
        snprintf(kernelName, sizeof(kernelName), "op%d-Custom", opIndex);
        assignKernelName(&kernel, kernelName);
        builderAppendKernel(builder, kernel);
    }
}

/* ============================================================================
 * 算子编排激励展开(顶层): 处理 repeat 展开、放置算卡、链式/DAG 依赖
 * ==========================================================================*/
static int buildOperatorGraphStimulus(EngineJob *job, JsonValue *srcJson,
                                      KernelBuilder *builder) {
    JsonValue *opsArray = jsonGetMember(srcJson, "ops");
    int baseOpCount = jsonArrayLength(opsArray);
    int graphRepeat = job->stimulus.repeat;
    int isDagLayout;
    int totalExpanded, expandedIndex;
    OpKernelRange *opRanges;
    int result = 0;

    logxTraceEnter(LOG_LEVEL_DEBUG, "buildOperatorGraphStimulus", NULL);

    if (baseOpCount < 1) {
        snprintf(job->errorMessage, sizeof(job->errorMessage), "算子列表为空");
        logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
        result = 1;
        goto done;
    }

    /* DAG 编排下禁止重复整图(否则边序号无法解析) */
    isDagLayout = !strcmp(jsonGetString(srcJson, "layout", "chain"), "dag");
    if (isDagLayout && graphRepeat > 1) {
        snprintf(job->errorMessage, sizeof(job->errorMessage), "DAG 编排下 repeat 须为 1");
        logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
        result = 1;
        goto done;
    }

    {
        /* ================================================================
         * **【GEMM_1】展开与放置参数(单算子情形)
         * autoAssignNpu = src.autoNpu = 0(GEMM_1 显式给 0 ⇒ 自动轮询关闭);
         * bytesPerElement = src.prec=2 ≥1.5 ⇒ 2.0B/元素(FP16, bpp=2);
         * totalExpanded = baseOpCount(1) × graphRepeat(1) = 1 ⇒ 下面第一遍循环
         *   只跑一次: opRanges[0].firstKernelIndex=0, appendOperatorKernel 之后
         *   lastKernelIndex=0 —— 单算子展开出单内核、只占 builder 的一个槽位。
         * ================================================================ */
        int autoAssignNpu = jsonGetInt(srcJson, "autoNpu", 0);
        double bytesPerElement = (jsonGetNumber(srcJson, "prec", 2) >= 1.5) ? 2.0 : 1.0;

        totalExpanded = baseOpCount * graphRepeat;
        opRanges = (OpKernelRange *)malloc(sizeof(OpKernelRange) *
                                           (size_t)(totalExpanded > 0 ? totalExpanded : 1));

        /* 第一遍: 展开每个算子并分配算卡 */
        for (expandedIndex = 0; expandedIndex < totalExpanded; expandedIndex++) {
            int opIndexInList = expandedIndex % baseOpCount;
            JsonValue *opJson = opsArray->u.array.items[opIndexInList];
            int npuIndex = jsonGetInt(opJson, "npu", -1);
            /* **【GEMM_1】放置解析: op 无 "npu" 键 ⇒ npuIndex=-1(<0);
             * autoAssignNpu=0 ⇒ 三元取假分支 npuIndex=0 —— 这就是"autoNpu=0 →
             * 唯一内核固定落 0 号卡"的实现点; 若 autoNpu=1 才会按
             * opIndexInList % numNpu 轮询。GEMM_1: npuIndex=0, 单卡无越界。 */
            if (npuIndex < 0)
                npuIndex = autoAssignNpu ? opIndexInList % job->hardware.numNpu : 0;
            if (npuIndex >= job->hardware.numNpu) {
                snprintf(job->errorMessage, sizeof(job->errorMessage),
                         "算子 #%d 放置 NPU=%d 越界(算卡共 %d 张, 编号 0..%d)",
                         expandedIndex + 1, npuIndex,
                         job->hardware.numNpu, job->hardware.numNpu - 1);
                logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
                free(opRanges);
                result = 1;
                goto done;
            }
            opRanges[expandedIndex].firstKernelIndex = builder->count;
            appendOperatorKernel(builder, opJson, expandedIndex, npuIndex, bytesPerElement);
            opRanges[expandedIndex].lastKernelIndex = builder->count - 1;
        }

        /* 链式布局: 自动追加 前一个算子末内核 -> 后一个算子首内核 的依赖 */
        if (!isDagLayout) {
            for (expandedIndex = 1; expandedIndex < totalExpanded; expandedIndex++) {
                builderAddDependency(builder,
                                     opRanges[expandedIndex - 1].lastKernelIndex,
                                     opRanges[expandedIndex].firstKernelIndex);
            }
        }

        /* DAG 布局: 显式依赖边(边两端的序号都指“原始算子表”下标) */
        {
            JsonValue *edgesArray = jsonGetMember(srcJson, "edges");
            int edgeCount = jsonArrayLength(edgesArray);
            /* **【GEMM_1】依赖边解析: GEMM_1 的 edges=[] ⇒ edgeCount=0,
             * 下面的循环体一次也不执行 ⇒ 内核 0 的 predecessorCount 保持 0
             * (无前驱)。加上 layout="dag" 且 repeat=1, 链式布局的自动串行边
             * 也不追加 ⇒ 单内核"孤立就绪" —— runOperatorGraph 第一轮即选中它,
             * 这也是"唯一内核、无依赖"在依赖数组层面的完整处理。 */
            int edgeIndex;
            for (edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++) {
                JsonValue *edgeJson = edgesArray->u.array.items[edgeIndex];
                int fromOp = -1, toOp = -1;

                if (edgeJson->type == JSON_ARRAY && jsonArrayLength(edgeJson) >= 2) {
                    fromOp = (int)edgeJson->u.array.items[0]->u.numberValue;
                    toOp = (int)edgeJson->u.array.items[1]->u.numberValue;
                } else {
                    fromOp = jsonGetInt(edgeJson, "a", -1);
                    toOp = jsonGetInt(edgeJson, "b", -1);
                }

                if (fromOp >= 0 && toOp >= 0 && fromOp < baseOpCount && toOp < baseOpCount) {
                    /* 找到该原始算子第一次展开(第 0 份)的位置, 连依赖 */
                    int fromExpanded = -1, toExpanded = -1;
                    int expandedScan;
                    for (expandedScan = 0; expandedScan < totalExpanded; expandedScan++) {
                        if (expandedScan % baseOpCount == fromOp && fromExpanded < 0)
                            fromExpanded = expandedScan;
                        if (expandedScan % baseOpCount == toOp && toExpanded < 0)
                            toExpanded = expandedScan;
                    }
                    if (fromExpanded >= 0 && toExpanded >= 0) {
                        builderAddDependency(builder,
                                             opRanges[fromExpanded].lastKernelIndex,
                                             opRanges[toExpanded].firstKernelIndex);
                    }
                } else {
                    snprintf(job->errorMessage, sizeof(job->errorMessage),
                             "DAG 边 [%d,%d] 越界", fromOp, toOp);
                    logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
                    free(opRanges);
                    result = 1;
                    goto done;
                }
            }
        }
        free(opRanges);
    }

    /* 内核级调试打印(保留原 COSIM_DBG 语义) */
    if (g_cosim_dbg) {
        fprintf(stderr, "DBG ops_build end n=%d cubeF0=%g inB0=%g outB0=%g\n",
                builder->count,
                builder->count > 0 ? builder->items[0].cubeFlops : 0,
                builder->count > 0 ? builder->items[0].loadBytes : 0,
                builder->count > 0 ? builder->items[0].storeBytes : 0);
    }
    logxTraceLeave(LOG_LEVEL_DEBUG, "buildOperatorGraphStimulus", "内核数=%d", builder->count);

done:
    if (result)
        logxTraceLeave(LOG_LEVEL_DEBUG, "buildOperatorGraphStimulus", "失败: %s",
                       job->errorMessage);
    return result;
}

/* ============================================================================
 * 工作量统计(按算卡聚合 flops/搬运量): 用于计算“理论下界”
 * ==========================================================================*/
typedef struct {
    double cubeFlops;   /* 该卡的 Cube 计算总量 */
    double vectorFlops; /* 该卡的 Vector 计算总量 */
    double dmaBytes;    /* 该卡的搬运字节总量(输入+输出) */
    int    kernelCount; /* 该卡上的内核个数 */
} NpuWorkload;

static void summarizeNpuWorkload(EngineJob *job, NpuWorkload *stats) {
    int kernelIndex;
    int npuCount = job->hardware.numNpu;

    memset(stats, 0, sizeof(NpuWorkload) * (size_t)npuCount);
    for (kernelIndex = 0; kernelIndex < job->kernelCount; kernelIndex++) {
        WorkKernel *kernel = &job->kernels[kernelIndex];
        int npuIndex = kernel->npuIndex;
        if (npuIndex < 0)
            npuIndex = 0;
        if (npuIndex >= npuCount)
            npuIndex = npuCount - 1;
        stats[npuIndex].cubeFlops += kernel->cubeFlops;
        stats[npuIndex].vectorFlops += kernel->vectorFlops;
        stats[npuIndex].dmaBytes += kernel->loadBytes + kernel->storeBytes;
        stats[npuIndex].kernelCount++;
    }
}

/* ============================================================================
 * 建模主入口(engine.h 导出)
 * 流程: 选择激励模式 -> 生成内核 -> 统计理论下界 -> 输出 HBM 容量告警
 * ==========================================================================*/
int buildWorkloadKernels(EngineJob *job, JsonValue *requestJson) {
    /* ================================================================
     * **【GEMM_1】建模主流程(GEMM_1 走法)
     * builder: 动态内核数组(扩容翻倍, 初始 4096 槽) —— GEMM_1 只用 1 个槽。
     * 分支: job->stimulus.mode==STIMULUS_MODEL 不成立 ⇒ else 支: 先置
     *   pipelineActive=0、microBatchWaves=1(单卡无 PP 波形), 再调
     *   buildOperatorGraphStimulus() 产出 1 个内核。
     * 校验: kernelCount=1(≥1 且 ≤2,000,000, 通过)。
     * 之后进入"理论下界"块与 HBM 告警块(见下), 最后 logx 打印
     *   内核数=1 波形数=1 PP流水=关 理论下界=429,497 cycles。
     * ================================================================ */
    KernelBuilder builder;
    JsonValue *srcJson;
    int ok = 0;

    builder.count = builder.capacity = 0;
    builder.items = NULL;

    logxTraceEnter(LOG_LEVEL_INFO, "buildWorkloadKernels", NULL);

    srcJson = jsonGetMember(requestJson, "src");
    if (!srcJson) {
        snprintf(job->errorMessage, sizeof(job->errorMessage), "缺少 src");
        logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
        goto done;
    }

    if (job->stimulus.mode == STIMULUS_MODEL) {
        if (buildModelStimulus(job, &builder)) {
            free(builder.items);
            goto done;
        }
    } else {
        job->pipelineActive = 0;
        job->microBatchWaves = 1;
        if (buildOperatorGraphStimulus(job, srcJson, &builder)) {
            free(builder.items);
            goto done;
        }
    }

    job->kernels = builder.items;
    job->kernelCount = builder.count;

    if (job->kernelCount < 1) {
        snprintf(job->errorMessage, sizeof(job->errorMessage),
                 "未生成任何内核(请检查配置)");
        logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
        free(builder.items);
        job->kernels = NULL;
        goto done;
    }
    if (job->kernelCount > 2000000) {
        snprintf(job->errorMessage, sizeof(job->errorMessage),
                 "内核规模过大 %d", job->kernelCount);
        logxLog(LOG_LEVEL_ERROR, "%s", job->errorMessage);
        free(builder.items);
        job->kernels = NULL;
        goto done;
    }

    /* ---- 理论下界(cycles): 按“每卡并行上限”估算, 用于效率评分 ---- */
    {
        NpuWorkload npuWorkload[256];
        const HardwareConfig *hardware = &job->hardware;
        double clockGhz = hardware->coreClockGhz;
        double maxLowerBound = 0, sumLowerBound = 0;
        int parallelCards, npuIndex;

        summarizeNpuWorkload(job, npuWorkload);

        /* 模型模式按 PP 阶段数并行; 算子模式全卡并行 */
        parallelCards = job->pipelineActive
                            ? (job->model.pipelineParallel > hardware->numNpu
                                   ? hardware->numNpu : job->model.pipelineParallel)
                            : hardware->numNpu;

        /* ================================================================
         * **【GEMM_1】理论下界逐行(本块只算"每卡"上限, 不做流水/排队展开)
         * npuWorkload[256]: 按卡聚合的负载(见 summarizeNpuWorkload)——
         *   GEMM_1 只有 npu0: cubeFlops=1.3743895e11, vectorFlops=0,
         *   dmaBytes=100,663,296, kernelCount=1。
         * parallelCards: ops 模式非流水 ⇒ = hardware->numNpu=1(循环只转一圈)。
         * 对每张卡分别算三本"时间账"(单位 cycle, 口径=FLOPs/B÷每秒能力×频率):
         *   cubeLowerBound   = cubeFlops×clockGhz/(cubeTf×1e3)
         *      = 1.3743895e11×1/(320×1e3) ≈ 429,497 —— 为什么分母是 ×1e3:
         *      TFLOPS×1e12(FLOP/s)÷freq×1e9(cycle/s)=TF×1e3(FLOP/cycle@1GHz);
         *   vectorLowerBound = vectorFlops×clockGhz/(vecTf×1e3)=0/16e3=0;
         *   dmaLowerBound    = dmaBytes×clockGhz/(hbmBw×1e3)
         *      = 100,663,296/(1.2×1e3)=83,886 cyc(搬运账, 计算账的 ~1/5)。
         * perNpuBound = max(cube,vector,dma) = max(429,497, 0, 83,886)=429,497
         *   —— 计算远大于搬运 ⇒ 理论上是 Compute-bound(roofline 口径)。
         * maxLowerBound/sumLowerBound 合并: pipelineActive=0 ⇒
         *   lowerBoundCycles=microBatchWaves×sumLowerBound=1×429,497;
         *   最后的"多卡并行取 max"分支要求 parallelCards>1 —— GEMM_1=1 不进入。
         * ================================================================ */
        for (npuIndex = 0; npuIndex < parallelCards; npuIndex++) {
            double cubeLowerBound = npuWorkload[npuIndex].cubeFlops * clockGhz /
                                    (hardware->cubePeakTflops * 1e3);
            double vectorLowerBound = npuWorkload[npuIndex].vectorFlops * clockGhz /
                                      (hardware->vectorPeakTflops * 1e3);
            double dmaLowerBound = npuWorkload[npuIndex].dmaBytes * clockGhz /
                                   (hardware->hbmBandwidthTBs * 1e3);
            double perNpuBound = cubeLowerBound > vectorLowerBound
                                     ? cubeLowerBound : vectorLowerBound;
            if (dmaLowerBound > perNpuBound)
                perNpuBound = dmaLowerBound;
            if (perNpuBound > maxLowerBound)
                maxLowerBound = perNpuBound;
            sumLowerBound += perNpuBound;
        }

        if (job->pipelineActive) {
            /* PP 流水线: 波形数相关, T = (waves-1)*max + Σ */
            job->lowerBoundCycles = (job->microBatchWaves - 1) * maxLowerBound + sumLowerBound;
        } else {
            /* 单卡多波形 / 并行多卡 */
            job->lowerBoundCycles = job->microBatchWaves * sumLowerBound;
        }
        if (!job->pipelineActive && parallelCards > 1)
            job->lowerBoundCycles = maxLowerBound;   /* 多卡并行: 取下界最大者 */

        /* ---- HBM 容量提示(非阻断告警) ---- */
        /* **【GEMM_1】告警判定: maxPerCardBytes=npu0 的 dmaBytes=100,663,296B
         * (≈0.1GB) vs hbmCapacityBytes=hbmSizeGb×1e9=64e9 —— 远未超容 ⇒
         * 不写 job->warningMessage, GEMM_1 结果里 warn 字段为空串。 */
        {
            double maxPerCardBytes = 0;
            double hbmCapacityBytes = hardware->hbmSizeGb * 1e9;
            int cardIndex;
            for (cardIndex = 0; cardIndex < hardware->numNpu; cardIndex++) {
                if (npuWorkload[cardIndex].dmaBytes > maxPerCardBytes)
                    maxPerCardBytes = npuWorkload[cardIndex].dmaBytes;
            }
            if (maxPerCardBytes > hbmCapacityBytes) {
                snprintf(job->warningMessage, sizeof(job->warningMessage),
                    "%s%s每卡单步流式数据量约 %.1f GB 超过 HBM 容量 %.1f GB(容量提示: 权重/激活无法全驻留, 需重复从外部载入; 仿真仍按流式加载近似)",
                    job->warningMessage[0] ? job->warningMessage : "",
                    job->warningMessage[0] ? "; " : "",
                    maxPerCardBytes / 1e9, hbmCapacityBytes / 1e9);
            }
        }
    }

    logxLog(LOG_LEVEL_INFO, "建模完成: 内核数=%d 波形数=%d PP流水=%s 理论下界=%.0f cycles",
            job->kernelCount, job->microBatchWaves,
            job->pipelineActive ? "开" : "关", job->lowerBoundCycles);
    ok = 1;

done:
    logxTraceLeave(LOG_LEVEL_INFO, "buildWorkloadKernels",
                   ok ? "成功" : "失败: %s", job->errorMessage);
    return ok ? 0 : 1;
}

/* ============================================================================
 * 释放建模阶段分配的资源
 * ==========================================================================*/
void freeJobResources(EngineJob *job) {
    if (job->kernels) {
        free(job->kernels);
        job->kernels = NULL;
    }
    job->kernelCount = 0;
}
