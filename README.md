# 软硬协同调度仿真平台（C 引擎 + Web 前端 单文件 exe）

> 需求来源: `prompt.md`（软硬协同设计调度仿真）。交付形态：**单文件 `dist\CoSimPlatform.exe`**。
> 双击 exe → 自动找空闲端口并打开默认浏览器 → 页面上完成 5 个页面的配置 → C 引擎在本地完成
> tile 切分、软件流水调度与 **Cycle 级近似仿真** → 返回拓扑/泳道图/利用率/评分板等可视化结果。

## 一、五大页面

| # | 页面 | 内容 |
|---|------|------|
| 1 | 硬件配置 | 分组可配：算卡整体架构（NPU 数、卡间拓扑 all-to-all/ring/mesh/torus/linear）、主频与算力（AI Core/AI CPU 主频、每卡核数、Cube/Vector 峰值）、存储与 Tile（L0/UB/L2/HBM 容量带宽）、内部总线与互联（MTE1/2/3 时延、AIC↔AICPU、HCCS、RoCE 带宽时延）；提供默认/大卡(64核512T)/小卡(16核128T) 一键预设 |
| 2 | 模型配置 | 预设 Llama-3-70B / DeepSeek-V3(MoE) / 自定义；模型结构（h/L/头数/KV头/V/s/MicroBatch/FFN/精度/MoE开关/专家数/TopK/专家FFN维）、并行域 TP·PP·DP·CP·EP、训练策略（ZeRO/激活重算/CPU Offload/每算子 Tile 切分数） |
| 3 | 算子激励 | 激励来源二选一（模型生成 / 算子编排）；算子卡片：类型（GEMM/Conv2D/FlashAttention/LayerNorm/Softmax/GELU/MoE/Transformer Block/自定义FLOPs字节）、逐类型参数、放置 NPU、**依赖（序号，逗号分隔，支持分支/合并计算图）**、全局精度；内置 GEMM→Norm→GELU 链与分支合并图示例 |
| 4 | 调度算法 | 算法卡片：名称/描述、**指令重排顺序（就绪顺序/关键路径优先）**、核绑定（单核/静态分区8核/全核均衡）、软件流水（单/双/三级缓冲）、Tile 上限、UB 占比；内置 3 个对照算法（朴素列表/关键路径模调度/深度优先流水）；泳道视图参数（算法/NPU/核心范围/分箱数） |
| 5 | 仿真结果 | 参考桌面设计：5.0 卡间拓扑(网格+全互联Fabric)与调度后逻辑 DAG(最长路径分层采样)；5.1 统计块(端到端Cycles/纯计算/DMA搬运量/控制下发/气泡/掩盖率)；5.2 泳道 SVG(单元图例+悬停任务) + ASCII 泳道(可复制) + 气泡归因表；5.3 利用率条+受限判定徽标；5.4 评分板(🏆胜者高亮) + 裁判结论三维度(Tile粒度/指令重排/访存-计算重叠) |

## 二、运行与构建

### 跨页一致性校验（前端即时拦截 + C 引擎权威校验）

| 规则 | 处置 |
|------|------|
| 并行域乘积 TP·PP·DP·CP·EP ≤ 算卡数（如 7 卡配 TP=8 即被拦截） | 报错，定位到「模型与并行」页 |
| TP ≤ 每卡 AI Core 核数 | 报错 |
| PP ≤ 模型层数 L（每个流水级至少 1 层） | 报错 |
| 算子「放置 NPU」编号须在 0..算卡数-1 | 报错，定位到「算子激励」页 |
| 算子依赖序号须小于自身序号（防 DAG 成环） | 报错 |
| 调度算法使用核数 ≤ 每卡核数；缓冲深度 1-4；Tile上限/UB占比范围 | 报错，定位到「调度算法」页 |
| EP>1 但未启用 MoE | 告警(不阻断)，显示于结果页顶部 |
| 每卡单步流式数据量 > HBM 容量 | 告警(不阻断)，结果页顶部黄色提示 |

- **运行（最终用户）**：双击 `dist\CoSimPlatform.exe`。默认绑定 `0.0.0.0`（全部网卡），控制台给出**局域网访问地址**（如 `http://192.168.x.x:8098/`）并自动打开浏览器；本机回退地址 `http://127.0.0.1:8098/`。关闭控制台即退出。
  - 端口范围 8098-8128；**3080 为 DeepSeek Harness 保留端口，本应用拒绝占用**（即使 `--port 3080` 也会自动改用默认范围）。
  - 可选参数：`--port 9000`、`--host 127.0.0.1`（仅回环）/`--host 0.0.0.0`（全部网卡）、`--no-browser`、
    `--run xxx.json`（自测：读配置 JSON 直接输出结果，`COSIM_DBG=1` 可开启内核级调试打印）。
- **运维日志**：进程启动即把日志写入独立日志文件（默认当前目录 `cosim_engine.log`），记录**函数调用链、请求处理过程、错误与耗时**，便于排障；不会写入业务 JSON。
  - `COSIM_LOG_FILE` 指定日志文件路径（默认 `cosim_engine.log`）
  - `COSIM_LOG_LEVEL` 日志级别：`1`=仅错误、`2`=+警告、`3`=+主流程调用链（默认，推荐）、`4`=+内部细节
- **重新构建（开发）**：需要 MSVC（本机已验证：`C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools`）。
  ```
  powershell -ExecutionPolicy Bypass -File .\build.ps1
  ```
  脚本自动执行两步：① `web/index.html` → `webdata.h`（字节数组嵌入）；② `vcvars64 + cl /std:c11 /utf-8` 编译。
  亦可用 GCC/MinGW 手工编译（链接 `-lws2_32 -lshell32`）。

## 三、架构

```
CoSimPlatform.exe
 ├─ server.c/main   Win32+winsock 内嵌 HTTP 服务, 自动开浏览器 (默认绑定 0.0.0.0, 端口 8098 起试绑, 3080 保留给 Harness)
 ├─ webdata.h       前端 index.html 字节数组 (GET /)
 ├─ jsonx.c         无依赖 JSON 解析/写出
 ├─ cfg.c           请求配置解码 + 校验 + 单位换算
 ├─ work.c          建模: 模型激励逐层 9 类算子(PP分卡) / 算子激励(链/DAG) → 逻辑内核
 ├─ sim.c           Cycle 级近似仿真引擎
 │                  单位: 每卡 AI CPU/MTE2/MTE3/HCCS + 每核 MTE1/Cube/Vector
 │                  微任务: CPU下发→LD(MTE2)→M1(L1→L0)→Cube/Vector→ST(MTE3)
 │                  双/多级缓冲(依赖 chunk c-dbuf), PP 波形合并 T=(waves-1)·Dmax+ΣD
 ├─ run.c           多算法运行 → 评分板/裁判结论/泳道 JSON
 ├─ logx.c          运维日志(函数调用链/请求过程/耗时), 默认 cosim_engine.log
 │                  环境变量: COSIM_LOG_FILE / COSIM_LOG_LEVEL(1-4)
 └─ web/index.html  五页面 SPA (原生 JS + Canvas/SVG, 无外部依赖, 离线可用)
```

## 四、仿真方法学（近似模型说明）

- 时延: `T_comm = Latency + DataSize/BW`；`T_compute = FLOPs/峰值`（FP16 默认 2B/元素）。
- 每个逻辑内核按“UB 可驻留字节”切成 K 型 chunk（每核 P 块），块间按缓冲深度做软件流水；
  MTE2/MTE3 为卡级共享带宽资源，Cube/Vector/MTE1 为核级资源；等待数据/带宽/下发即产生气泡并归因。
- 激励建模：模型模式生成每层内核（QKV/Attn(Mix)/OutProj/LN1/MLP/GELU/LN2；MoE 模式为 Router+专家FFN，按 TopK 稀疏计算、全部专家权重载入），
  PP 阶段按层切分到算卡，卡间边界激活走 HCCS；权重视作每步经 HBM 流式加载；算子模式支持逐类型参数、放置 NPU 与**依赖序号**（自动转内核级 DAG 边，跨卡自动插 HCCS）。
- 端到端合并：PP 波形数=mb，`T = (mb-1)·Dc_max + ΣDc`（mb=1 退化为逐卡串行，mb 大时逼近流水稳态）。
- 指标：Bubble%=100−Cube&Vector 占用率（含气泡/填充/失衡）；访存掩盖率=被计算覆盖的 DMA 字节占比
  （1024 时间箱判定）；SRAM 峰值≈每核在飞缓冲近似；受限类型由占用/瓶颈占比判定；评分=效率/气泡/掩盖/均衡的加权。
- 调度算法维度：指令重排顺序（就绪顺序 DFS / 关键路径优先 CP）影响内核发射次序；核绑定（单核/静态分区/全核）决定并行度；
  缓冲深度决定搬运-计算重叠窗口；每算子 Tile 切分数(tileSplit)与 Tile 上限(maxChunk)约束切块粒度。
- 近似与未建模：激活重算、ZeRO/Offload、DP 多副本、卡间消息并发冲突、L2 命中率、指令级细节等未计入，
  数值用于**横向算法对比与瓶颈归因**，非硬件级精度。DAG 分支若共卡，默认同卡核资源下串行化（建议分卡并行）。

## 五、目录

```
prompt.md          原始需求
build.ps1          一键构建(生成 webdata.h + MSVC 编译)
web/index.html     前端源码(5 页面)
*.c / *.h          C 引擎源码(含运维日志模块 logx.c / logx.h)
webdata.h          构建生成(勿手改)
dist\CoSimPlatform.exe   交付 exe
```

---

## 附：仓库内容说明

- 本仓库为 **C 引擎源码快照**：11 个 `.c/.h` 源文件 + `webdata.h`（构建生成的前端内嵌数据）+ `README.md` + 原始需求 `prompt.md`。
- 源码中带 **【GEMM_1】** 关键字的中文批注，用于说明“一次 GEMM 仿真从建模到出结果”的完整流程与每个变量/公式。
- 未收录：前端源码 `web/index.html`、构建脚本 `build.ps1`、可执行文件 `dist/CoSimPlatform.exe`；`send_mail.py` 因含邮箱授权码未收录，见 `tools/send_mail.example.py`（脱敏示例）。
- 构建：Windows + MSVC（`cl /std:c11 /utf-8`）编译 `cfg.c jsonx.c logx.c run.c server.c sim.c work.c`，链接 `ws2_32.lib shell32.lib`。
- 运行：`CoSimPlatform.exe --host 127.0.0.1 --port 8098` 打开浏览器操作；日志写入 `cosim_engine.log`。