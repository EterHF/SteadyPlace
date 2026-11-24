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

## TODO

- [ ] 增加依据当前姿态、托盘运动和接触信号放行的规则基线。
- [ ] 预测各层的滑移与倾覆风险，联合选择落点和松爪时机。
- [ ] 在不同起始相位、摩擦和加速度下配对比较成塔率、冲击与用时。

## 引用

仿真使用 MuJoCo，Panda 模型取自 MuJoCo Menagerie 的固定版本，采用 Apache-2.0 许可。 [MuJoCo](https://mujoco.org/) · [Panda 模型](https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda) · [许可证](assets/franka_emika_panda/LICENSE)

选题参考运动托盘运输和基于接触预测的控制。 [Keep it Upright](https://arxiv.org/abs/2305.17484) · [Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity](https://arxiv.org/abs/2505.13350)
