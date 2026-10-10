# RoboLab 并行异步评测

这套入口将多个 RoboLab 向量环境放在一个仿真进程内，再按配置把多个仿真进程映射到不同 GPU。策略服务独立启动；不在评测代码中加载权重，也不抢占已有 GPU 任务。

```bash
python examples/robolab_quant/evaluation/run_topology.py topology.json /absolute/new/output
```

配置包含 `sim_python`、`robolab_repo`、`policy_gpus`、`horizon`、`delay_s`、`jitter_s`、`seed_offset`，以及 `lanes`。每个 lane 指定独立 `name`、`sim_gpu`、`task`、`mode`、`num_envs`、`num_runs`、`host`、`port`。总回合数为所有 lane 的 `num_envs × num_runs` 之和。输出目录必须不存在。

每个环境独享一个 websocket 连接、动作块、推理线程和抖动流。观测复制仍在仿真线程进行；回合结束先等待未完成请求，避免把上一回合响应带入下一回合。trace 中同时记录 run 与 env_id，结果按 `(run, env_id)` 验收，不将环境编号丢掉。

## 两种用途必须分开

- **吞吐与任务质量评测**：可共享策略服务、多环境同步推进，报告整批端到端回合/小时、成功数、峰值显存和拓扑。建议从小规模验证后扩到文档中的2张策略卡＋6张仿真卡、每进程10环境。
- **部署延迟机制评测**：共享服务引入排队，同步向量环境中一个环境等待会拖住其他环境，改变异步控制的墙钟语义。必须固定服务负载及环境数，重新建立并行同步基线；不得将其延迟与旧单环境数据直接合并。

并行环境采用 batch_seed；配对身份是 `(batch_seed, env_id)`。不宣称第 i 个向量环境等于单环境 seed+i。RoboLab一次 reset 消耗批量随机数，环境数变化也可能改变初始状态。跨方法比较必须固定向量宽度、任务和 reset 批次。后续若要与旧串行 seed 完全对应，应增加初始状态导入与哈希验收。

`summary.json` 的吞吐包含进程加载、编译及正式执行。RoboLab 的 run 级耗时可能被复制到多个环境结果中，不能将这些墙钟相加当成设备时间。请求 trace 的 transport 包含共享服务排队，不是纯 GPU 推理时间。

## 当前实现与限制

客户端按环境隔离异步状态，但 RoboLab 主循环仍逐环境调用 infer，首块与 sync 请求仍可能串行等待；这不是模型端 batching。先测量收益，再决定是否需要 batched transport。服务自动部署、跨主机调度和集群资源管理不在此入口内。

GPU选择使用完整可见设备列表和明确的物理 `cuda:N` / renderer activeGpu，以避免 CUDA_VISIBLE_DEVICES 重编号与 Vulkan 不一致。驱动库路径可用 `library_path` 配置；`sim_env` 可设置 `VK_ICD_FILENAMES` 限定单份Vulkan驱动，及 `ROBOLAB_WARP_PATH` 指向独立兼容Warp目录。本机验证采用Warp 1.8.1。异构机器须验证设备映射后再投入全量评估。

## 验证

```bash
uv run --no-project --with numpy python -m unittest discover -s examples/robolab_quant/evaluation -p 'test_*.py'
```

单元测试覆盖环境状态隔离、局部reset、单个在途请求、过期动作处理、gripper融合和GPU拓扑冲突。真实仿真结果与吞吐验收另存实验输出，不以单元测试替代。
