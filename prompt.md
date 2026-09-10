# 软硬协同调度仿真平台 — 完整需求 Prompt（原始规格 + 后续补充）

> 本文件是该项目**完整的原始需求**（含二次迭代补充），与最终交付实现 `index.html` 对应。

---

## Role, Background & System Goal

帮我构建一个仿真平台。

### 平台背景 (Background)

本仿真平台服务于**软硬件协同设计 (Hardware-Software Co-Design)** 闭环体系。项目团队分为：

1. **调度算法团队**：专门开发编译期 / 运行期的调度算法（包括 Task DAG 拓扑排序、Tile 级切分策略、模调度 / 流水线展开、多核负载均衡策略等）；
2. **仿真工具团队**：构建硬件级精准仿真平台，为调度算法提供客观、高精度的评估与归因环境。

### 核心目标 (Core Objective)

本仿真器的核心目标是作为一个**“调度算法的硬核裁判与硬件瓶颈诊断引擎”**。
在**固定且相同的硬件基础配置**（如算子 / 任务激励不变、算卡规格不变、Tile 架构与规格不变、内部互联带宽不变、AI Core/AI CPU 主频不变）下，模拟运行由不同调度算法生成的任务执行序列（Task Graph/DAG），**准确评估并对比不同调度算法的调度效率与优度**，归因流水线气泡（Bubble）与片上资源瓶颈，指导算法迭代与硬件匹配。

---

## 1. 输入内容与规范 (Input Specification)，需要在 web 可视化界面上提供相应的配置能力

仿真器接受以下三部分输入，并进行工程与物理合理性校验：

### 1.1 硬件基础配置 (Hardware Base Configuration)

允许用户完全自定义硬件环境（若未指定则自动填充 3.1 节默认代际规格）：

- **算卡整体架构**：包含计算 NPU 数量（`NUM_NPUS`）、卡间拓扑（`NET_TOPOLOGY`）。
- **主频与算力规格**：AI Core 主频（`FREQ_GHZ`）、AI CPU 主频（`AICPU_FREQ_GHZ`）、Cube 峰值算力（`CUBE_PEAK`）、Vector 峰值算力（`VECTOR_PEAK`）。
- **存储与 Tile 架构规格**：L0 Buffer 尺寸（`L0_SIZE`）、L1/UB SRAM 容量（`UB_SIZE`）、Shared L2 Cache 容量（`L2_SIZE`）、HBM 容量与带宽（`HBM_BW`）。
- **内部总线与互联带宽**：MTE1 (L1->L0) 带宽与时延、MTE2 (L2/HBM->L1/UB) 带宽与时延、MTE3 (L1/UB->HBM) 带宽与时延、Scalar 控制总线带宽、AI Core <-> AI CPU 交互带宽、HCCS/RoCE 卡间网络带宽与延迟。

### 1.2 激励与 Tile 规格配置 (Workload Incentive & Tile Specifications)

- **算子 / 任务激励 (Workload Incentive)**：输入待执行的目标算子或复杂计算图（如 Conv, GEMM, FlashAttention, MoE Transformer Block 等）。
- **Tile 规格与切分架构**：
  * **Tile Shape 映射**：指定计算块粒度（如 \(M_k \times N_k \times K_k\)）。
  * **SRAM / L0 映射**：定义每个 Tile 在片上缓冲区的驻留空间与 double/triple buffering 策略。

### 1.3 待评测的调度算法序列 (Scheduled Task DAG Inputs)

支持同时导入一个或多个调度算法生成的任务调度方案（如 `算法 A: 深度优先列表调度` vs `算法 B: 关键路径模调度`），每个方案包含：

- **Task 节点**：执行类型（`MTE1/2/3` 搬运、`AIC` 矩阵计算、`AIV` 向量计算、`AI CPU` 下发控制 / Fallback、`HCCS/RoCE` 通信）、数据量（Bytes）、计算量（FLOPs）。
- **资源与拓扑绑定**：指定 Task 运行的具体 Core ID、分配的 SRAM 物理内存偏移（Memory Offset）、Task 间的硬依赖边（Edges/Flags）。

---

## 2. 硬件架构与连接拓扑模型 (Hardware Architecture Topology)

- **控制通路 (Control Path)**：Host CPU <--(PCIe)--> AI CPU <--(Control Bus)--> TS (Task Scheduler) <--> AI Core (Scalar/Vector/Cube)。
- **数据通路 (Data Path)**：
  * 片外：`HBM` <--(MTE2)--> `L2 Cache` <--(MTE2)--> `L1 / UB SRAM` <--(MTE1)--> `L0A/B/C` <--(Cube/Vector)--> `输出` <--(MTE3)--> `L1/UB/HBM`。
  * 卡间：`Local HBM/SRAM` <--(MTE3/DMA)--> `HCCS/RoCE Controller` <--(Interconnect)--> `Remote NPU`。

---

## 3. 硬件规格参考参数表 (Hardware Specs Reference Table)

### 3.1 基础硬件与主频规格

| 参数名称             | 默认规格       | 配置标识符            |
| ---------------- | ---------- | ---------------- |
| AI Core 主频       | 1.0 GHz    | `FREQ_GHZ`       |
| AI CPU 主频        | 2.0 GHz    | `AICPU_FREQ_GHZ` |
| AI Core 数量       | 32 核 / NPU | `NUM_AI_CORES`   |
| Cube 峰值 (FP16)   | 320 TFLOPS | `CUBE_PEAK`      |
| Vector 峰值 (FP16) | 16 TFLOPS  | `VECTOR_PEAK`    |

### 3.2 存储、Tile 架构与内部互联带宽

| 组件 / 路径             | 默认容量 / 带宽       | 配置标识符          | 默认时延                | 配置标识符 (时延)      |
| ------------------- | --------------- | -------------- | ------------------- | --------------- |
| L0 Buffer (L0A/B/C) | 64 KB / 核       | `L0_SIZE`      | 2.0 TB/s (Per Core) | `L0_BW`         |
| L1 / UB SRAM        | 256 KB / 核      | `UB_SIZE`      | 15 Cycles (MTE1)    | `MTE1_LAT`      |
| Shared L2 Cache     | 192 MB (Total)  | `L2_SIZE`      | 3.0 TB/s (Total)    | `L2_BW`         |
| HBM 显存              | 64 GB, 1.2 TB/s | `HBM_BW`       | 40 Cycles (MTE2/3)  | `MTE2_LAT`      |
| AI Core <-> AI CPU  | 64 GB/s         | `AIC_AICPU_BW` | 0.8 us              | `AIC_AICPU_LAT` |
| HCCS 卡间互联           | 392 GB/s        | `HCCS_BW`      | 1.0 us              | `HCCS_LAT`      |
| RoCE 机间网络           | 400 Gbps        | `ROCE_BW`      | 2.5 us              | `ROCE_LAT`      |

---

## 4. 模型输入参数定义 (Input Schema Specification)

仿真工具必须支持以下三组核心参数的交互配置（提供预设如 Llama-3-70B、DeepSeek-V3 及自定义输入）：

### 4.1 模型结构与训练参数 (Model Architecture Parameters)

- **隐层维度 (h)**：Hidden Dimension / \(d_{\text{model}}\)（例如：8192）
- **层数 (L)**：Transformer Block 总层数（例如：80）
- **注意力头数 (H) & KV 头数 (\(H_{\text{kv}}\))**：用于区分 MHA / GQA / MLA 结构
- **词表大小 (V)**：Vocabulary Size（例如：128,000）
- **序列长度 (s)**：Context Length（例如：4096）
- **Micro-Batch Size (b)**：单卡单步处理批次大小
- **数值精度 (Precision)**：FP16/BF16 (2 Bytes) 或 FP8 (1 Byte)

### 4.2 并行划分策略 (Parallel Domain Strategies)

- **张量并行度 (\(P_{\text{TP}}\))**：切分 MLP 与 Attention 权重矩阵
- **流水线并行度 (\(P_{\text{PP}}\))**：按层切分 Stage，配合 Micro-batch 管道
- **数据并行度 (\(P_{\text{DP}}\))**：配合 ZeRO Stage (Off / ZeRO-1 / ZeRO-2 / ZeRO-3)
- **上下文并行度 (\(P_{\text{CP}}\))**：沿序列维度 s 切分 Attention
- **专家并行度 (\(P_{\text{EP}}\))**：MoE 架构下路由专家节点切分
- **激活重算 (Activation Checkpointing)**：None / Selective / Full
- **CPU Offload**：None / Optimizer / Parameters

---

## 5. 仿真计算与评估规则 (Simulation Engine Logic)

1. **时延计算公式**：
   * 搬运 / 通信延迟 \(T_{\text{comm}} = \text{Latency} + \frac{\text{DataSize}}{\text{Bandwidth}}\)
   * 计算执行延迟 \(T_{\text{compute}} = \frac{\text{TileFLOPs}}{\text{Peak\_Performance}}\)
2. **气泡 (Bubble) 与依赖冲突归因**：
   * 准确捕捉由于数据未就绪导致的 Wait Flag 停顿、AI CPU 指令下发延迟造成的流水线空闲、以及多核间 Load Imbalance 带来的长尾等待。
3. **算法优度指标计算**：
   * \(\text{Bubble 占比} = \frac{\text{Stall Cycles}}{\text{Total Cycles}}\)
   * \(\text{访存掩盖率} = \frac{\text{被计算完全掩盖的 DMA 搬运量}}{\text{总 DMA 搬运数据量}}\)

---

## 5. 仿真输出与算法评估报告 (Output & Benchmarking Report)，需要做到可视化呈现，在 web 页面里面

仿真完成后，按以下 5 个部分输出结果：

### 5.0 拓扑

首先呈现最终的物理和逻辑拓扑情况，包括 npu 间拓扑，npu 内部不同处理单元（ai core，ai cpu，cube，vector，mte，cache 等）之间的逻辑拓扑，含带宽时延等信息；说执行算子经过调度后，在 npu/tile 上的依赖情况，不同的卡如何有序的调度。

### 5.1 端到端执行耗时 (Total Cycles & Execution Latency)

- 提供总 Cycle 数及微秒（us）转换。
- 拆分 “纯计算时间”、“DMA 搬运时间”、“控制下发等待时间” 及 “掩盖后实际端到端耗时”。

### 5.2 硬件执行泳道图与气泡归因 (Swimlane Diagram & Bubble Diagnosis)

- 输出各硬件单元（各 NPU、TILE 区分呈现：AI CPU、MTE1/2/3、Cube、Vector、HCCS）在时间轴上的 ASCII 执行泳道图。
- **气泡归因分析**：精准标注图中 Bubble 的产生原因（例如：`Tile 2 搬运未被 Tile 1 计算掩盖`，`Core 0 与 Core 31 负载不均衡导致 Sync 阻塞`）。

### 5.3 硬件利用率与架构瓶颈诊断 (Hardware Utilization)

- 给出 Cube/Vector 利用率、MTE 带宽饱和度、SRAM 内存占用峰值。
- 判定系统在当前调度算法下属于 Compute-bound、Memory-bound 还是 Overhead-bound。

### 5.4 调度算法优度量化对比 (Scheduling Algorithm Evaluation Scoreboard)

在**完全相同的硬件规格与激励**前提下，生成多算法对比表：

| 调度算法名称 | 端到端总 Cycles | 总耗时 (us) | 流水线 Bubble 占比 (%) | 访存掩盖率 (%) | SRAM 内存峰值 (KB) | 多核负载均衡度 (StdDev) | 综合优度评分 (1-100) |
| ------ | ----------- | -------- | ----------------- | --------- | -------------- | ---------------- | -------------- |

- **裁判结论**：明确声明在当前硬件环境与激励下哪个调度算法更优，并从 “Tile 粒度”、“指令重排顺序”、“访存 - 计算重叠度” 三个维度阐明性能差异的根本原因。

---

## 二次迭代补充需求 (Incremental)

### 算子激励输入页面

还需要提供输入算子的页面，可以是单个业界的算子，也可以是一个任务有多个算子组合，还可以是一个自定义的多算子计算图编排能力，可以放到一起，在一个页面，灵活可配。

> 实现要点：单算子（GEMM / Conv2D / FlashAttention / MoE / TransformerBlock / LayerNorm / Softmax / GELU / Custom）、多算子组合（链式）、自定义多算子计算图（依赖编排，支持分支 / 合并），与「模型生成」激励二选一。

---

## 技术决策（已确认）

| 决策项   | 选择                                                                                                                                                  |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 交付形态  | 单文件 HTML 应用（原生 JS + SVG，浏览器直接打开，无需构建）                                                                                                               |
| 首版范围  | 一次性实现完整规范（三类配置 + 引擎 + 5.0~5.4 全部输出）                                                                                                                 |
| 引擎保真度 | Cycle 级近似模型（\(T_{\text{comm}} = \text{Latency} + \text{DataSize}/\text{BW}\)，\(T_{\text{compute}} = \text{FLOPs}/\text{Peak}\)，含依赖等待 / 资源竞争 / 多核负载） |
