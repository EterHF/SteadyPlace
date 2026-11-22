# SteadyPlace

机器人动力学与控制课程项目。在 MuJoCo 中用 Franka Panda 搬运 50 mm 方块，在未知台高（±40 mm）及持续推力或多次随机脉冲下，将物块完整放稳于 64 mm 承托台。任务要求 **7.2 s 内完成**，放置全程的台面竖向力峰值 **≤10 N**。

广义动量观测器用关节状态和指令力矩估计载荷与接触。上层 SAC 调节水平与法向刚度、近台速度、减速起点和夹爪指令，基线与 SAC 共用承托控制和松爪判据。

[选题 PPT](slides/机器人期末项目_选题说明_北大模板精简版.pptx) · [PDF](slides/机器人期末项目_选题说明_北大模板精简版.pdf) · [持续扰动配对演示](results/coupled_pilot/paired_demo.mp4)

## 复现实验

在项目根目录建立环境并运行基线评估。

```bash
conda create -y -p "$PWD/.conda/envs/carryplace" -c conda-forge python=3.11 pip
conda activate "$PWD/.conda/envs/carryplace"
python -s -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -s -m sim.evaluate_coupled --workers 4 --output runs/coupled
```

复跑单种子 30 万步 SAC 及配对评估（Linux，CPU）。

```bash
python -s -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -s -m pip install -r requirements-rl.txt
mkdir -p runs/sac
cp results/sac_pilot/{config,protocol,split}.json results/sac_pilot/worker.sh runs/sac/
(
  export PYTHONPATH="$PWD"
  cd runs/sac
  bash worker.sh && python -s -m sim.evaluate_sac
)
```

## 已完成结果

基线共测了 300 个主回合。固定刚度下，近台速度从 16 提到 50 mm/s，放置峰值平均增加 **9.25 N**（双方均触桌的 50 对），完成用时中位差为 **−0.55 s**（双方均完成的 33 对）。[基线复测](docs/方案2_跨阶段耦合复测.md)

| 同一批 90 个独立测试工况 | 成功 | 超时 | 峰值均值（N） | 完成用时中位数（s） |
| --- | ---: | ---: | ---: | ---: |
| 固定刚度，22 mm/s | 30/90 | 20 | 11.08 | 6.25 |
| 调参后的常量控制 | 21/90 | 46 | 16.28 | 5.81 |
| SAC，单种子 30 万步 | 14/90 | 67 | 11.41 | 6.73 |

峰值只计已触桌回合，用时只计完成放置的回合（含冲击超限）。SAC 有 **39/90** 个回合到时限仍未触桌。[测试汇总](results/sac_pilot/test_300000.json) · [逐回合数据](results/sac_pilot/test_300000_runs.json)

## TODO

- [ ] 比较固定速度与基于名义间隙、剩余时间的分段调速，测量低台未触桌率和高台放置峰值。
- [ ] 固定刚度与夹爪，只学习近台速度和减速起点。对照原奖励与加入全程峰值、超时惩罚的奖励。
- [ ] 用 3 个训练种子和新抽样测试工况复测，按台高及无扰、持续推力、多次随机脉冲分组报告成功率、冲击和用时。

## 引用

Panda 模型来自 [MuJoCo Menagerie 的固定版本](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda)，许可见 [Apache-2.0](assets/franka_emika_panda/LICENSE)。算法、仿真器与训练框架的出处见[文献调研](docs/文献调研.md)。
