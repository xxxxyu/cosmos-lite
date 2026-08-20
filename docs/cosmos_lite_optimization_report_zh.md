<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xiangyu Li.
SPDX-License-Identifier: OpenMDW-1.1
-->

# 从 Cosmos Framework 到 Cosmos Lite：机器人策略压缩与推理优化技术报告

[English version](cosmos_lite_optimization_report.md)

## 摘要

Cosmos Lite 的起点是一个很具体的工程目标：把 Cosmos3 Nano 16B 机器人
策略从需要数据中心 GPU 的原始运行方式，变成能在单张 RTX 4090 24GB 上
直接部署、复现和评测的系统。后来项目又加入了 Cosmos3 Edge 4B，并继续
优化单机器人、batch size 1 的请求延迟。

最终结果不是由某一个“万能优化”带来的，而是几类改动叠加：

1. 用 W4A16/W8A16 weight-only 量化解决显存问题，并把权重、配置、
   tokenizer 和 VAE 打包成可直接加载的自包含 bundle。
2. 把 UniPC denoise steps 从 4 减到 2，在保留 guidance 3 的情况下，把
   denoiser forward 次数减半。
3. 在 RTX 4090 上启用 generation branch 的 FP8 W8A8，降低占主导地位的
   大矩阵乘法成本。
4. 编译完整的 MoT language block，合并可合并的逐元素操作，并减少 Python
   dispatch。
5. 复用 FP8 Q/K/V 和 gated-MLP 的输入量化结果，避免对同一 activation
   重复量化。
6. 按真实 attention shape 选择 SageAttention 或 FlashAttention2；Nano 还
   复用同一请求内不变的 condition K/V。
7. 为 Edge 的主要 FP8 shape 实现 RTX 4090/SM89 专用 Triton kernel。
8. 删除请求预处理中的无效工作：只 resize 唯一一张真实观测帧，不再 resize
   后续 32 张已知为零的占位帧。

在 RoboLab-120 全任务 `default` instruction mode、guidance 3、2 denoise
steps、32x8 action chunk 协议下，当前保留的最快配置是：

| Model            | 当前配置                                                                              | RTX 4090 request p50 | Peak reserved |     RoboLab-120 SR |
| ---------------- | ------------------------------------------------------------------------------------- | -------------------: | ------------: | -----------------: |
| Cosmos3 Edge 4B  | GenW8A8 + compile + shared FP8 + Triton SM89 + Sage FP16-PV + sparse transform        |         **331.4 ms** |       8.79 GB | 231/1,200 = 19.25% |
| Cosmos3 Nano 16B | GenW8A8 + compile + shared FP8 + Sage FP8-PV + condition K/V cache + sparse transform |         **958.5 ms** |      15.51 GB | 380/1,200 = 31.67% |

每组全任务结果覆盖 120 个任务、每任务 10 个 episode。针对 Banana 的小规模
paired gate 分别为 Edge 40/50、Nano 49/50；它们适合受控消融，但不再作为主要
质量指标。

这两个结果属于当前优化分支，不应和只包含 W4A16/W8A16 的 `v0.2.0` 稳定
发布混为一谈。本文的目的，是完整说明哪些方法有效、哪些没有，以及为什么。

## 1. 范围与读数方式

### 1.1 模型不是普通 LLM

Cosmos3 Policy 使用 MoT（Mixture of Transformers）结构。每个 decoder
layer 中有两条主要计算路径：

- **Understanding branch**：处理文本、状态和视觉条件。
- **Generation branch**：处理扩散 latent，并生成 action chunk。

两条路径各有 attention 和 MLP 投影。它们的 token 数、矩阵 shape、调用次数
和精度敏感性都不同。因此，“把 language model 全部量化成 4 bit”不是足够
准确的描述，也不是可靠的策略设计方法。

量化计划覆盖 Nano 的 504 个、Edge 的 336 个 MoT language Linear module。
Action adapter、embedding、normalization、vision component 和 VAE 仍为 BF16。
因此本文的 `full_w8a8` 是“所有目标 Linear 都使用 W8A8”，不是整个模型的
每一个算子都变成 FP8。

### 1.2 W4A16、W8A16 和 W8A8 是什么

- **W4A16**：权重为 4 bit，输入 activation 保持 BF16/FP16。
- **W8A16**：权重为 8 bit，输入 activation 保持 BF16/FP16。
- **W8A8**：权重和矩阵乘法输入都使用 8 bit。本文的高性能实现使用 FP8
  E4M3，输出回到 BF16。

W4A16/W8A16 的主要价值首先是节省显存。只有 kernel 与真实 shape 匹配时，
它们才一定会加速。W8A8 会进一步减少 Tensor Core 的计算量，但需要额外做
activation quantization；如果这个步骤本身太慢，FP8 GEMM 再快也没有意义。

### 1.3 延迟和成功率不能混在一起看

本文使用以下边界：

- **Kernel latency**：一个独立算子的时间，只能判断算子潜力。
- **Generate latency**：模型 `generate_samples_from_batch` 的时间。
- **Request latency**：从 policy server 收到输入，到返回 action 的时间；包含
  预处理、generation 和少量后处理，不包含 IsaacSim。
- **Success rate (SR)**：仿真任务闭环成功率，是最终质量门控。

Open-loop replay 还报告 action error：

- **L1 mean**：所有 action 元素绝对误差的平均值。
- **Linf p95**：先取每个请求中最大的 action 元素误差，再报告 95 分位。

这些误差适合快速筛选，但不能代替 rollout。一个小的 action 差异可能在闭环
中被环境放大，也可能完全不影响任务完成。

### 1.4 术语速查

- **Shape**：算子实际处理的矩阵维度。维度改变后，同一个 kernel 的速度可能
  明显变化。
- **p50**：中位数；一半请求比它快，另一半比它慢。
- **Paired rollout**：让两个策略从相同初始状态和随机种子开始评测，比两组
  互不相关的实验更容易看出真实差异。
- **Peak reserved**：PyTorch 的 CUDA 内存池曾保留的最大显存。它通常比当前
  tensor 实际占用更保守，适合判断能否部署。
- Attention 中的 **Q/K/V** 分别是 query、key 和 value；**PV** 指第二次矩阵
  乘法，也就是用 attention probability 对 value 加权。

## 2. 三个阶段的系统演进

| 阶段                       | Nano 16B                                                      | Edge 4B                                   | 主要意义                                                  |
| -------------------------- | ------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| NVIDIA 原始框架            | CloseFridge BF16 加载 31.76GB、峰值 33.09GB，无 4090 baseline | BF16 g3/s2 为 582ms、9.20GB、251/1,200    | 能训练和推理，但不是 4090 Nano 部署方案                   |
| Cosmos Lite 首个稳定量化版 | Full W8 g3/s2 为 2403ms、21.42GB、378/1,200                   | Full W8 g3/s2 为 576ms、8.71GB、239/1,200 | 自包含 W4/W8 bundle，一键 replay/rollout                  |
| 当前最快优化版             | 958.5ms、15.51GB、380/1,200                                   | 331.4ms、8.79GB、231/1,200                | FP8、compile、attention/cache、SM89 kernel 和数据路径优化 |

Nano 原始 BF16 没有可靠的单卡 4090 request latency，因为它首先就不满足
24GB 显存约束。以第一个可部署的 Full W8 g3/s2 为基准，当前 Nano request
p50 提升约 **2.51x**。Edge 原始 BF16 本来就能运行；当前最快版本相对其
582ms 提升约 **1.76x**。

Nano 原始 BF16 显存来自 RoboCasa CloseFridge，而当前最快 Nano 延迟和 SR
来自 RoboLab-120 上的 DROID policy。跨环境表格展示部署能力的演进；2.51x
来自同为 RoboLab 协议的 Full W8 baseline。

主要 SR 来自 120 个任务、共 1,200 个 episode。置信区间互相重叠，因此几个
百分点的差异在其他机器人或输入协议上仍有不确定性。下文的 50-episode Banana
paired 结果用于局部消融。

### 2.1 技术决策总表

| 方向                 | 技术                                        | 证据范围                                   | 最终状态                 |
| -------------------- | ------------------------------------------- | ------------------------------------------ | ------------------------ |
| Artifact             | 流式量化和自包含 bundle                     | 全新部署、hash validation、replay、rollout | 保留                     |
| Weight precision     | Marlin W8A16                                | Nano/Edge/RoboCasa 完整 replay 与 rollout  | 稳定默认                 |
| Weight precision     | Marlin W4A16 和固定 W4/W8 mixed plan        | 完整 replay 与 rollout                     | 作为可选取舍保留         |
| Calibration          | Training-set AWQ 风格 input-channel scaling | 128-frame calibration、held-out replay     | 所有含 W4 的 bundle 使用 |
| Sampling             | Guidance 3、2 个 UniPC steps                | RoboLab paired 50-episode gate             | RoboLab 保留             |
| Sampling             | Guidance 1                                  | Replay 和 paired rollout                   | 不作为通用默认           |
| Activation precision | Generation-branch FP8 W8A8                  | Operator、replay32、50-episode rollout     | 当前最快分支保留         |
| Activation precision | Full FP8 W8A8                               | Operator、replay32、50-episode rollout     | 支持，但不是首选         |
| Activation precision | W4A8                                        | API 和 kernel 可行性研究                   | 延后，没有接入模型       |
| Graph                | Dynamic complete-language-block compile     | Replay32 和 50-episode rollout             | 保留                     |
| Graph                | Static compile、扩大到 VFM、CUDA Graphs     | 端到端 replay                              | 放弃                     |
| Projection           | 共享 FP8 activation quantization            | Bit-exact replay32                         | 保留                     |
| Projection           | 拼接权重的 packed projection GEMM           | Operator 和 replay                         | 放弃                     |
| Attention            | Shape-aware Sage + FA2 fallback             | Operator、replay32、rollout                | 保留                     |
| Attention            | FlashInfer                                  | Operator benchmark                         | 放弃                     |
| Cache                | Request-local condition K/V                 | Replay 和 rollout                          | 只在 Nano 保留           |
| Cache                | PAB/TeaCache/SmoothCache/BAC 风格近似复用   | 论文调研和收益上限分析                     | 延后，未实现             |
| GEMM                 | Shape-tuned Triton SM89 FP8                 | NCU、operator、replay、rollout             | 只在 Edge 保留           |
| Data path            | 稀疏 future-frame transform                 | Bit-exact replay 和端到端 latency          | 默认保留                 |
| Input/control        | 缩短 action chunk、减少视角、提前 resize    | 端到端测试                                 | 放弃                     |
| CFG execution        | 合并 conditional/unconditional batch        | 端到端 replay                              | 放弃                     |
| Alternative engine   | AllSpark                                    | Real-shape operator benchmark              | 未集成                   |
| Alternative engine   | ExLlama、TensorRT-LLM、Machete              | 架构和 API 调研                            | 未实现，详见第 5 节      |

## 3. 先解决可部署性：weight-only 量化与 bundle

### 3.1 为什么先选 Marlin W4A16/W8A16

早期候选包括 GPTQ/AWQ 风格的 W4A16、torchao INT8 weight-only、vLLM
Marlin、AllSpark、ExLlama 风格 kernel、TensorRT-LLM 和 Machete。最终先选
vLLM Marlin，原因很实际：

- 支持 RTX 4090 所在的 Ada/SM89。
- 已有成熟的 W4/W8 packed layout 和 CUDA kernel。
- 能嵌入现有 PyTorch module，不需要把整个 MoT 模型改造成 LLM engine。
- W4 与 W8 可同时出现在一个模型中，便于 mixed precision。

Machete 需要 SM90/Hopper，不适用于 4090。ExLlamaV2/V3 对消费级 GPU 很有
参考价值，但其 runtime 和 LLM 假设较强，没有完成 Cosmos module 级集成。
TensorRT-LLM 也有可信的 INT4/INT8 路径，但完整导出 MoT、vision、VAE 和
policy server 的成本远高于替换一个 Linear backend，因此保留为后续路线，
没有在本阶段实现。

### 3.2 四种可部署精度策略

Cosmos Lite 保留了四种 W4/W8 策略：

| Strategy        | Precision map                  | 作用                       |
| --------------- | ------------------------------ | -------------------------- |
| `full_w8`       | 所有目标 Linear 为 W8A16       | 校准无关、质量优先         |
| `full_w4`       | 所有目标 Linear 为 W4A16       | 最低模型显存               |
| `attention_w8`  | 两条 attention 路径 W8，MLP W4 | RoboCasa 的显存/质量平衡点 |
| `gen_branch_w8` | generation branch W8，其余 W4  | 保留生成路径精度           |

RoboCasa CloseFridge 的两次独立 H100 50-episode 结果为：Full W8 96%、
Full W4 92%、Attention W8 96%、Gen-branch W8 94%。对应 RTX 4090 peak
allocated 为 19.20、12.47、13.93、15.84GB。它证明四种方案都能满足 24GB，
但也说明更低 bit 不等于更高任务成功率。

Nano RoboLab Banana 的 W4 结果更尖锐：g3/s4 下 Full W4 只有 26/50，低于
Full W8 的 43/50；因此 Full W4 被保留为可选 artifact，而不是通用默认值。

### 3.3 calibration 做了什么

W4 和 mixed 策略使用 128 个 training-split episode，每个 episode 取一帧，
收集每个 Linear 输入通道的 activation 最大值。随后用 AWQ 风格的缩放：在
不改变浮点函数的前提下，缩放输入通道与对应权重通道，再量化权重。

关键结果：

- RoboCasa Full W4 的 eval32 L1 mean 从 0.0538 降到 0.0177，改善 67%。
- 多数 mixed candidate 的 L1 改善 20%-68%。
- calibration 没有稳定消除少数 action spike；Full W4 的 Linf max 仍超过 1。
- 某些 precision map 的平均误差改善，但 Linf 反而变差。

因此 calibration 被保留，但结论不是“校准后 W4 就安全”，而是“校准能明显
改善分布，最后仍必须做闭环评测”。Full W8 不需要 calibration。

### 3.4 为什么早期 sensitivity 最优方案后来被降级

早期 open-loop sensitivity 发现 attention 比 MLP 更敏感。一个更细的候选
把 generation branch 和 understanding attention 保持 W8，只把 understanding
MLP 放到 W4；它达到较低的 L1，因此曾被称为 `mixed_best`。

但在同协议 50-episode RoboCasa rollout 中，`mixed_best` 只有 78%，而
Attention W8 为 90%。这个结果说明按 replay error 排名不能直接预测闭环质量。
最终产品只保留结构简单、证据更完整的四种策略。

### 3.5 流式量化与自包含 bundle

量化过程不把完整 BF16 checkpoint 放入 GPU。导出器逐层读取 BF16 权重，
完成量化和 packing 后立即释放，再处理下一层。RoboCasa release smoke 的
stream-pack peak 只有约 0.7-1.0GB。

bundle 包含 packed weights、剩余 BF16 tensors、runtime config、tokenizer、
VAE、precision map、来源 revision、文件大小和 SHA256。部署时不再读取原始
DCP/BF16 checkpoint 或 calibration data。这项工作本身不加速 kernel，但它
解决了早期“量化模型仍暗中依赖源 checkpoint”的严重部署风险。

## 4. 最大的第一步加速：减少 denoiser forward

在当前实现中，一次请求的 denoiser forward 数近似为：

```text
num_steps * (2 if guidance > 1 else 1)
```

Guidance 3 使用 conditional 和 unconditional 两个分支，也叫 CFG
（classifier-free guidance）。Guidance 1 去掉第二个分支。把 denoise steps
从 4 减到 2，则保留 CFG，但求解器只走两个时间点。

### 4.1 Nano

Nano Full W8：

| Guidance / steps | Request p50 |       Banana SR |
| ---------------- | ----------: | --------------: |
| g3/s4            |      4110ms |     43/50 = 86% |
| g3/s2            |  **2403ms** | **45/50 = 90%** |
| g1/s4            |      2431ms |     32/50 = 64% |
| g1/s2            |      1565ms |     40/50 = 80% |

g3/s2 降低 41.5% request latency，并没有观察到质量下降。g1 虽然更快，但
失败更多，失败 episode 往往跑满 horizon，最终整轮评测反而更慢，所以放弃。

### 4.2 Edge

Edge BF16 本身从 s4 的 21/50 提升到 s2 的 34/50，请求从 1042ms 降到
582ms。这是 checkpoint/sampler interaction，不是量化带来的“精度提升”。
UniPC 是数值求解器，更多 step 只表示更细的积分；如果模型学到的 vector
field 与 schedule 并不完全匹配，更多求解点不保证 action 更好。

所以 RoboLab 的当前默认是 g3/s2。RoboCasa 的结果波动更大，仍把 g3/s4
保留为保守默认值。

## 5. 为什么 W4/W8 省显存，却不一定更快

### 5.1 真实 shape 比通用 LLM benchmark 更重要

RTX 4090 的 real-shape GEMM benchmark 显示：

- Nano 的 call-weighted W4/W8 比 BF16 约快 1.06x/1.04x。
- Edge 的 call-weighted W4/W8 反而比 BF16 慢 4%/6%。
- 大 generation MLP shape 能摊薄解包和 kernel launch 成本。
- token 很少的 condition projection 经常是 BF16 更快。

Marlin 的 Nsight Compute 结果也不是“Tensor Core 已经满载”。代表性 kernel
使用 255 registers/thread、约 101KB shared memory/block，occupancy 只有
约 8%-16%，大部分 scheduler cycle 没有 eligible warp。主要问题是 tile、
register/shared-memory 压力和 latency hiding，而不是单纯 DRAM 或算力不够。

### 5.2 尝试过的其他 W4/W8 backend

| Backend                  | 结果                                                                         | 决策             |
| ------------------------ | ---------------------------------------------------------------------------- | ---------------- |
| torchao INT8 weight-only | 数值接近 W8，但 H100 policy request 可到数十秒；代表 shape 也比 BF16 慢 3-4x | 只作精度参考     |
| vLLM AllSpark W8A16      | 4090 top-shape weighted 结果接近 BF16，未稳定胜过 Marlin                     | 不集成           |
| Machete                  | 需要 SM90                                                                    | 不适用于 4090    |
| ExLlamaV2/V3             | 调研过，面向消费卡，但没有合适的独立 `nn.Linear` 接口和直接兼容 layout       | 作为参考，未集成 |
| TensorRT-LLM/ModelOpt    | 潜力可信，但需要较大的整图导出工程                                           | 延后             |
| PyTorch INT8 `_int_mm`   | 不支持 token=10 shape，且动态 W8A8 较慢                                      | 放弃             |

这里的重要取舍是：不为了“backend 更多”增加维护成本。只有能在 Cosmos
真实 shape 上提供清楚收益的路径，才进入端到端模型。

## 6. FP8 W8A8：第一次明显加速大 GEMM

RTX 4090 支持 FP8 Tensor Core，但不支持 Blackwell 的原生 FP4。FP8 因此是
4090 上最值得尝试的 activation quantization 路线。

### 6.1 先测 kernel，再接模型

早期 RoboCasa top-8 shape benchmark 发现：

- FP8 GEMM kernel-only 比 Marlin W4 快约 39%。
- 加上 eager dynamic activation quantization 后，整体反而比 Marlin 慢。
- 对大 MLP 使用 static activation scale，operator 可省 30.7%，但推算端到端
  上限只有约 14.5%。

因此没有把早期 PyTorch `_scaled_mm` static-scale 路线产品化。它需要额外
校准、处理 saturation，还没有足够大的端到端收益。

### 6.2 最终实现

后续实现采用成熟的 vLLM CUTLASS FP8 op：

- 权重使用静态 per-output-channel E4M3 scale。
- activation 使用 dynamic per-token E4M3 scale。
- 输出为 BF16。
- `gen_branch_w8a8` 只把 generation branch 改成 FP8；其余保留 calibrated W4。
- `full_w8a8` 把所有目标 Linear 改成 FP8。

Nano GenW8A8 相对 GenW8A16 request p50 从 1674.5ms 降到 1340.0ms，提升
20.0%。Full W8A8 为 1329.9ms，只再快 10.1ms，却多占 3.57GB reserved
memory，因此 mixed GenW8A8 是更好的 Nano 起点。

Edge GenW8A8 相对 GenW8A16 从 573.5ms 降到 502.8ms，提升 12.3%；Full
W8A8 为 510.6ms，速度和 replay error 都更差，因此同样选择 GenW8A8。

50-episode rollout：Nano GenW8A8 为 47/50，Edge 为 39/50。两者都通过
首轮质量门控。

### 6.3 equalization 和 W4A8

使用 DROID train128 input-amax 做 `alpha=0.5` equalization，并没有同时改善
平均误差、tail error 和延迟，因此放弃，保留更简单的 dynamic FP8。

W4A8 没有进入模型。4090 没有原生 FP4 Tensor Core；把 INT4 权重反量化后
再走 FP8 GEMM，现有架构上缺少成熟、明显获益的融合 kernel。kernel-level
study 没显示足够强的理由，因此没有手写 W4A8 实现。

## 7. 编译计算图：从失败到有效

### 7.1 第一版为什么失败

早期直接对 Marlin 模型使用 `torch.compile` 时，Marlin custom op 没有 fake
tensor/meta 实现，fullgraph 直接失败。允许 graph break 后虽然能跑，steady
generate 从约 1031ms 变成 1053ms，首次请求约 20秒。这个版本被放弃。

### 7.2 后来为什么能成功

FP8 路线把 quantized Linear 组织成可追踪的 custom-op 边界，并改为编译完整
MoT language block。Inductor 不能进入 CUTLASS FP8 GEMM 内部重写 kernel，
但可以处理其前后的 norm、residual 和 pointwise 操作，并减少 Python launch。

动态 language-block compile 的结果：

| Model | Eager p50 | Compiled p50 |  改善 |
| ----- | --------: | -----------: | ----: |
| Edge  |   502.8ms |      441.4ms | 12.2% |
| Nano  |  1322.7ms |     1130.9ms | 14.5% |

首次请求需要约 12.5秒（Edge）或 14.9秒（Nano）编译，因此部署必须 warm up。

### 7.3 被否决的 compile/graph 变体

| Candidate                         | 结果                                                   | 决策     |
| --------------------------------- | ------------------------------------------------------ | -------- |
| Static shape compile              | 新 prompt 长度触发 6-7秒重编译                         | 放弃     |
| Compile language + 全部 VFM heads | steady latency 无收益，冷启动约翻倍，action error 更大 | 放弃     |
| Inductor CUDA Graphs              | shape capture 开销，无稳定收益                         | 默认关闭 |
| 重复 prompt CUDA Graph            | Edge 433.6ms，未胜过 compile-only                      | 放弃     |
| MLP-only compile                  | 只有 1%-2% 收益，parity 更差                           | 放弃     |
| Attention-only compile            | 约 6% 收益，parity 更差                                | 放弃     |
| `force_same_precision`            | 没改善 parity                                          | 放弃     |

CUDA Graph 不是框架“不支持”，而是 policy prompt 长度变化、CPU preprocessing、
tokenization、sampler state 和 action 回传让 whole-request capture 不划算。

## 8. 减少重复 FP8 activation quantization

Generation Q/K/V projection 读取同一个 activation；Nano gated MLP 的 gate/up
也读取同一个 activation。原路径每个 Linear 都单独做一次 FP8 quantization。

`FP8_PROJECTION_FUSION=shared` 只量化一次，再把同一结果送给多个 GEMM。
它不拼接权重，不改变 GEMM tile，因此 replay32 action bit-identical。

- Edge request p50：436.8 -> 428.6ms，改善 1.9%。
- Nano request p50：1144.3 -> 1119.9ms，改善 2.1%。

另一个候选把多个 projection 权重沿输出维拼起来，单算子上更快，但改变
CUTLASS accumulation tiling，Nano reserved memory 增到 19GB，端到端也没有
稳定胜过 shared quantization，因此没有暴露到部署接口。

## 9. Attention 与 condition K/V cache

### 9.1 先看占比，再决定是否优化

早期 RoboCasa Nsight 中 attention 只占 GPU kernel time 的约 4.4%，因此单独
优化 attention 不可能带来成倍加速。进入 FP8 + compile 后，GEMM 已经变快，
长序列 generation attention 才成为值得处理的次级瓶颈。

真实 shape 显示：generation attention 的 Q 长度约 3093、KV 长度约 3175；
understanding causal attention 只有约 82 tokens。一个 backend 不适合所有 shape。

### 9.2 Shape-aware SageAttention

| Shape          | FlashAttention2 | Sage FP8-PV | 结论      |
| -------------- | --------------: | ----------: | --------- |
| Edge long Gen  |         0.706ms |     0.324ms | Sage 更快 |
| Nano long Gen  |         1.220ms |     0.551ms | Sage 更快 |
| Edge short Und |         0.084ms |     0.245ms | FA2 更快  |
| Nano short Und |         0.083ms |     0.231ms | FA2 更快  |

最终 policy 只在 SM89、dense non-causal、Q length >= 512 时用 Sage，其余
情况继续使用 FlashAttention2。

Edge 首轮 FP8-PV Sage rollout 为 34/50，而当时 FA2 baseline 为 37/50，点估计
不利，因此没有直接推广。随后测试更高精度的 INT8-QK + FP16-PV/FP32：

- 单算子 0.463ms，仍快于 FA2 的 0.706ms。
- replay action L1 比 FP8-PV 降低约 13%，Linf p95 降低约 14%。
- Edge request p50 从 349.7ms 降到 331.4ms。
- paired rollout 为 40/50，对应 FA2 baseline 39/50，wins/losses 6/5。

因此 Edge 最终选择 FP16-PV Sage。更快的 FP16+FP32 variant 在完整图中遇到
不支持的 WARPQ=16 shape，被放弃。FlashInfer BF16 为 0.725ms，没有胜过 FA2，
也没有集成。

### 9.3 Condition K/V cache

在一次 denoising request 内，understanding tokens 不变。Nano cache 在第一步
分别保存 conditional/unconditional 的 understanding K/V，第二步只重新计算
generation 路径。cache 只活一个请求，不跨机器人 observation。

Nano Sage+cache request p50 达到 987.2ms（加入 sparse transform 后为 958.5ms），
50-episode SR 为 49/50。Edge 上 cache 数值正确，但没有可重复 latency 收益，
所以只在 Nano 默认开启。

### 9.4 没有实现通用 DiT block cache

PAB、TeaCache、SmoothCache 和机器人 diffusion policy 的 BAC 都被调研过。
它们会跨 timestep 复用 attention 或 block 输出，可能带来更大收益，但当前
只有 2 denoise steps，可复用空间有限，而且会改变 action。现阶段只保留精确
的 condition K/V cache，没有实现近似 block cache。

## 10. RTX 4090 专用 Triton FP8 GEMM

vLLM CUTLASS FP8 kernel 在 Edge 主要 shape 上使用较大的 tile，Nsight Compute
显示 register/shared-memory 压力高、eligible warp 少。为此实现了只覆盖四个
已验证 Edge generation shape 的 Triton SM89 kernel，其他 shape 自动回退
CUTLASS。

| Edge shape M x K x N | CUTLASS |  Triton |  改善 |
| -------------------- | ------: | ------: | ----: |
| 3093 x 2048 x 9216   | 0.569ms | 0.407ms | 28.6% |
| 3093 x 2048 x 2048   | 0.165ms | 0.111ms | 32.8% |
| 3093 x 2048 x 1024   | 0.245ms | 0.075ms | 69.6% |
| 3093 x 9216 x 2048   | 0.629ms | 0.532ms | 15.4% |

Edge request p50 改善约 8.4%，rollout 从 matched FA2/CUTLASS control 的 37/50
变为 39/50，通过门控。

Nano shape 的 Triton kernel 也有 11%-23% 单算子收益，request p50 改善 8.7%，
但 rollout 从 49/50 降到 47/50。对已有 49/50 的基线来说，这组结果没有足够的
质量余量，因此 Nano allowlist 排除这些 shape，继续使用 CUTLASS。

## 11. 请求数据路径：最后 30ms 来自 CPU

细粒度 profile 把请求拆成 sample construction、batch construction、CUDA
generation、action transfer 和 postprocess。Edge 旧路径有 36.64ms 花在 sample
construction，而输入只有第一张 frame 是真实图像，后续 32 张都是零占位。

新路径先 resize 第一张 frame，再按目标分辨率直接分配零 tensor：

| Edge data path   | Build sample | CUDA generate | WebSocket request |
| ---------------- | -----------: | ------------: | ----------------: |
| Resize 33 frames |      36.64ms |      338.35ms |          382.84ms |
| Resize 1 frame   |   **2.31ms** |      339.87ms |      **349.70ms** |

replay32 的 action 每个元素完全一致。Nano 同样从 987.2ms 降到 958.5ms。
该优化默认开启，可用 `SPARSE_VIDEO_TRANSFORM=0` 回退；自定义 transform 类型
仍走旧路径。

## 12. 其他明确放弃或降级的尝试

| 方法                           | 主要结果                                                 | 处理               |
| ------------------------------ | -------------------------------------------------------- | ------------------ |
| Guidance 1                     | request 更快，但 Nano/Edge 闭环失败明显增加              | 不推荐             |
| Action chunk 16/8              | 理论工作量更小，实测延迟反而更差，并改变控制协议         | 放弃               |
| Camera pre-resize 到 192       | 模型最终仍进入 256 bucket，无明显收益                    | 放弃               |
| Wrist-only / 减少视角          | 改变输入分布，收益不够                                   | 放弃               |
| CFG batching                   | Edge 517.4 -> 555.5ms；FLOPs 没减少，attention path 更差 | 放弃               |
| Edge condition cache           | 正确但无可重复收益                                       | 关闭               |
| Full W8A8                      | 已支持；Nano 只比 GenW8A8 快 10.1ms，却多 3.57GB         | 保留选项，不作默认 |
| Nano Triton FP8                | kernel 和 replay 更快，但 SR 点估计 98% -> 94%           | 不启用             |
| Edge Sage FP8-PV               | 最快，但早期 SR 点估计不利                               | 改用 FP16-PV       |
| FlashInfer                     | long-attention operator 未胜过 FA2                       | 不集成             |
| 通用 shape-aware W4/W8 routing | 小 shape 有收益空间，但最终请求由大 Gen shape 主导       | 没有增加复杂路由   |
| Multi-env batching             | 可能提高仿真吞吐，但真机一次只有一个 rollout             | 低优先级，未做     |

## 13. 最终取舍与推荐配置

正式发布把模型 artifact 与运行时采样分开。Nano 推荐 GenW8A8 作为综合最优
方案，W8A16 作为免校准、可在 24GB 部署的 fallback。Edge 推荐 GenW8A8
作为低延迟方案，NVIDIA BF16 作为官方 quality-first 路径；Edge W8A16 仅保留为
可选压缩 artifact。Guidance、denoise steps 和 shift 都可以在不重新构建 checkpoint
的情况下修改；每个 benchmark 表格会单独注明使用的 sampler。

### 13.1 当前最快模型与运行时配置

Edge：

```bash
TORCH_COMPILE=1 \
COMPILED_REGION=language \
COMPILE_DYNAMIC=1 \
FP8_PROJECTION_FUSION=shared \
FP8_GEMM_BACKEND=triton_sm89 \
SAGE_ATTENTION=1 \
SAGE_PV=fp16_fp32 \
examples/robolab_quant/pipeline.sh serve
```

Nano：

```bash
TORCH_COMPILE=1 \
COMPILED_REGION=language \
COMPILE_DYNAMIC=1 \
FP8_PROJECTION_FUSION=shared \
SAGE_ATTENTION=1 \
CONDITION_KV_CACHE=1 \
examples/robolab_quant/pipeline.sh serve
```

两者都需要一个不计入 steady latency 的 warmup request。Sparse transform
默认开启。所有优化均有独立回退边界，不需要重新打包权重。

### 13.2 为什么现在适合收敛一版

Edge 的 CUDA generation 约 330ms，Nano 约 948ms；sample build、action transfer
和 postprocess 已只占几毫秒。剩余大部分时间是真实模型计算。继续做普通 Python
清理或小算子替换，不太可能再带来大幅改善。

下一阶段如果还要明显降低延迟，需要投入更高的方案：更深的 FP8 GEMM 与
norm/residual 融合、token pruning、early exit、模型蒸馏，或带质量门控的
block cache。这些不再是“低风险免费加速”，应作为单独的 kernel 工程或研究
项目评估。

## 14. 可复现资料与参考

仓库内详细数据：

- [RoboLab 主 benchmark](benchmarks/robolab.md)
- [RoboLab 消融实验](benchmarks/robolab_ablations.md)
- [RoboCasa365 benchmark](../examples/robocasa365_quant/BENCHMARKS.md)
- [FP8 W8A8 experiment](experiments/fp8_w8a8.md)
- [Graph optimization experiment](experiments/graph_optimization.md)
- [RTX 4090 SM89 optimization](experiments/rtx4090_sm89.md)

外部技术参考：

- NVIDIA Cosmos Framework: <https://github.com/NVIDIA/Cosmos-Framework>
- NVIDIA Cosmos3 Nano Policy DROID: <https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID>
- NVIDIA Cosmos3 Edge Policy DROID: <https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID>
- AWQ: <https://arxiv.org/abs/2306.00978>
- MARLIN: <https://arxiv.org/abs/2408.11743>
- UniPC: <https://arxiv.org/abs/2302.04867>
- Classifier-free guidance: <https://arxiv.org/abs/2207.12598>
- vLLM FP8 quantization: <https://docs.vllm.ai/en/latest/features/quantization/fp8/>
- PyTorch compile for diffusion models: <https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/>
- Triton: <https://github.com/triton-lang/triton>
- SageAttention: <https://github.com/thu-ml/SageAttention>
- FlashInfer: <https://github.com/flashinfer-ai/flashinfer>
- PAB: <https://arxiv.org/abs/2408.12588>
- TeaCache: <https://liewfeng.github.io/TeaCache/>
- SmoothCache: <https://openaccess.thecvf.com/content/CVPR2025W/eLVM/papers/Liu_SmoothCache_A_Universal_Inference_Acceleration_Technique_for_Diffusion_Transformers_CVPRW_2025_paper.pdf>
- Block-wise Adaptive Caching for diffusion policies: <https://block-wise-adaptive-caching.github.io/>
