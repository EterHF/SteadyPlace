# SteadyPlace

机器人动力学与控制课程项目。Franka Panda 从固定料台逐个抓取四个 50 mm 方块，在单轴往复托盘上叠成四层。机械臂撤开后，整塔需在一个完整的托盘周期内保持承托和层序。

## 任务演示

[完整叠塔演示](results/stack_demo/demo.mp4) · [加快托盘后的倾倒演示](results/stack_demo/fast/demo.mp4)

两个工况共用脚本式笛卡尔阻抗控制器。末端跟随托盘的位置与速度，按固定时序下降、松爪和撤离。方块质量为 85 g，方块和托盘的滑动摩擦参数分别为 0.46 和 0.48。托盘行程约 21 cm，改变周期会同时改变速度、加速度和放置时的运动相位。

| 托盘周期 | 参考加速度峰值 | 撤手后一周期 |
| --- | ---: | --- |
| 2.00 s | 1.04 m/s² | 四层稳定，全程无机械臂接触，最大相邻层水平偏移 6.3 mm |
| 1.30 s | 2.45 m/s² | 四块均完成放置，撤手后整塔倾覆 |

逐层首触后 100 ms 的峰值力与法向冲量、末次撤手用时、层间错位及底块相对托盘的漂移均有记录。撤手后按法向接触力检查四个承托面，要求至少 99% 的仿真步四个承托面同时有效、积木维持层序且全程无机械臂接触。 [默认工况数据](results/stack_demo/summary.json) · [加快工况数据](results/stack_demo/fast/summary.json)

## 复现

首次使用时在项目内配置 Conda 环境。

```bash
conda create -y -p "$PWD/.conda/envs/carryplace" -c conda-forge python=3.11 pip
conda activate "$PWD/.conda/envs/carryplace"
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
python -m sim.demo --output runs/demo
```

录制视频需要 PATH 中带 `libx264` 的 FFmpeg。在 Linux 服务器上用 EGL 离屏渲染。

```bash
conda install -y -c conda-forge ffmpeg
MUJOCO_GL=egl python -m sim.demo --render --output runs/demo
MUJOCO_GL=egl python -m sim.demo --render --period 1.30 --output runs/demo-fast
```

托盘参考周期可设为 0.5 s 及以上。第二条录制命令在倒塔时仍会保存视频和数据，并以状态码 1 退出。

## 基线试跑

六种托盘周期各取四个起始相位。定时、瞬时承托裕度、落点校正加承托裕度三种策略共用底层阻抗。规则试跑使用无噪声仿真位姿和当前参考加速度。可行性对照用中心对齐、各层交替偏航 ±0.3° 的预堆塔，在定时撤手对应的相位检查两个往复周期。 [逐工况结果](results/stack_baselines/summary.json)

预堆塔通过的 20 组中，三种策略分别成功 9、9 和 7 组。1.42 秒周期的四种相位均通过预堆检查，定时策略完成其中一组。

```bash
OPENBLAS_NUM_THREADS=1 python -m sim.evaluate_baselines --workers 8 --output runs/baselines
```

## 学习基线

残差 SAC 与残差 TD3 共用原来的阻抗随动。它们在第四块开始下降前读入托盘状态和无噪声仿真位姿、速度，给出 XY 落点各 ±10 mm 的修正和至多一周期的下降前等待。训练按最终成塔、撤手后的承托与倾斜、第四块首触冲量及等待时间给分。

训练从完整脚本生成的 60 个第四块下降前快照采样。测试另用 24 个托盘周期和起始相位均与训练分开的工况。两种算法各用种子 41 采样 768 次第四块放置，与定时和均匀随机残差配对比较。

24 组中，定时与随机分别成塔 11、10 组，残差 SAC 和残差 TD3 分别成塔 15、12 组。1.42 秒周期的六组里，SAC 与定时均完成两组。 [逐工况动作与结果](results/stack_learning/summary.json)

从项目根目录在已配置的环境中运行。评估结果写入 `runs/stack-learning/evaluate/summary.json`。

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-rl.txt
mkdir -p runs/stack-learning
cp -r configs/stack_learning/. runs/stack-learning/
(
  export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  cd runs/stack-learning || exit
  for stage in . sac td3 evaluate; do
    (cd "$stage" && python -m sim.learn_baselines) || exit
  done
)
```

## TODO

- [x] 在同一组周期与起始相位下比较定时松爪和瞬时承托裕度规则。
- [x] 在独立周期与相位上评估残差 SAC 和残差 TD3。
- [ ] 预测各层的滑移与倾覆风险，联合选择落点和松爪时机。
- [ ] 在不同起始相位、摩擦和加速度下配对比较成塔率、冲击与用时。

## 引用

仿真使用 MuJoCo，Panda 模型取自 MuJoCo Menagerie 的固定版本，采用 Apache-2.0 许可。 [MuJoCo](https://mujoco.org/) · [Panda 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda) · [许可证](assets/franka_emika_panda/LICENSE)

选题参考运动托盘运输和基于接触预测的控制。 [Keep it Upright](https://arxiv.org/abs/2305.17484) · [Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity](https://arxiv.org/abs/2505.13350)

学习基线沿用接触装配中的残差学习思路，SAC 与 TD3 使用 Stable-Baselines3 实现。 [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201) · [Panda 装配中的残差 SAC](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JoaoCarvalho/Adapting_Object_Centric_Probabilistic_Movement_Primitives_with_Residual_Reinforcement_Learning.pdf) · [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html) · [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
