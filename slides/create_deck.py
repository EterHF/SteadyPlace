"""Build the editable course-proposal deck and the figure from recorded pilot traces."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/smoke"
OUT = ROOT / "slides"
FONT = "Noto Sans SC"
PAPER, WHITE, INK = "F3F7F7", "FFFFFF", "142C36"
MUTED, LINE, DARK = "476069", "CFDEDF", "102531"
TEAL, TEAL_LIGHT, AMBER, AMBER_LIGHT = "067971", "E3F3EF", "A64B2C", "F9ECE3"
PALE, MINT = "C0D3DB", "74D6C5"


def box(slide, x, y, w, h, fill, outline=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, *(Inches(v) for v in (x, y, w, h)))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    if outline is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = RGBColor.from_string(outline)
        shape.line.width = Pt(0.7)
    return shape


def label(slide, x, y, w, h, value, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, url=None):
    shape = slide.shapes.add_textbox(*(Inches(v) for v in (x, y, w, h)))
    frame = shape.text_frame
    frame.clear()
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.word_wrap = False
    frame.vertical_anchor = MSO_ANCHOR.TOP
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(value.split("\n")):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = Pt(size * 1.34)
        paragraph.space_before = paragraph.space_after = Pt(0)
        run = paragraph.add_run()
        run.text = line
        if url:
            run.hyperlink.address = url
        run.font.name = FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = RGBColor.from_string(color)
        props = run._r.get_or_add_rPr()
        east_asia = OxmlElement("a:ea")
        east_asia.set("typeface", FONT)
        props.append(east_asia)
    return shape


def pic(slide, name, x, y, w, h):
    slide.shapes.add_picture(str(name), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def page(prs, section, title, source=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    num = len(prs.slides)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(PAPER)
    box(slide, 0.64, 0.46, 0.25, 0.045, TEAL)
    label(slide, 1.00, 0.31, 8.8, 0.29, section, 12, TEAL, True)
    label(slide, 0.64, 0.77, 12.05, 0.68, title, 31, INK, True)
    box(slide, 0.64, 6.94, 12.05, 0.012, LINE)
    label(slide, 0.64, 7.02, 11.35, 0.24, source or "机器人动力学与控制课程 · 期末项目选题", 10, MUTED)
    label(slide, 12.11, 7.00, 0.58, 0.26, f"{num:02d} / 09", 10, MUTED, align=PP_ALIGN.RIGHT)
    return slide


def notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


def pilot_plot(records, result=PILOT / "pilot_plot.png", fixed_color="355E78", phase_color=AMBER):
    # Register user-installed CJK fonts when Matplotlib's existing cache predates them.
    for filename in font_manager.findSystemFonts():
        if Path(filename).name.startswith("NotoSansSC-"):
            font_manager.fontManager.addfont(filename)
    font_manager.findfont(FONT, fallback_to_default=False)
    plt.rcParams.update({"font.family": FONT, "font.size": 11, "axes.unicode_minus": False,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.labelcolor": f"#{INK}", "xtick.color": f"#{MUTED}", "ytick.color": f"#{MUTED}"})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.65), dpi=210, facecolor=f"#{WHITE}")
    figure.subplots_adjust(left=0.078, right=0.982, bottom=0.17, top=0.82, wspace=0.38)
    names = {"fixed": ("固定分轴", f"#{fixed_color}"), "phase": ("规则分阶段", f"#{phase_color}")}
    for method, (name, color) in names.items():
        with (PILOT / f"{method}_trace.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        data = np.array([[float(row[key]) for key in
                          ("time_s", "x_m", "y_m", "ref_x_m", "ref_y_m", "target_fz_n")] for row in rows])
        time = data[:, 0]
        carry = time <= 2.0
        error = 1000.0 * np.linalg.norm(data[:, 1:3] - data[:, 3:5], axis=1)
        axes[0].plot(time[carry], error[carry], color=color, linewidth=1.75, label=name)
        touch_time = records[method]["first_touch_s"]
        elapsed_ms = 1000.0 * (time - touch_time)
        window = (elapsed_ms >= -0.01) & (elapsed_ms <= 100.01)
        axes[1].plot(elapsed_ms[window], data[window, -1], color=color, linewidth=1.75)
    start = records["fixed"]["scenario"]["onset"]
    duration = records["fixed"]["scenario"]["duration"]
    axes[0].axvspan(start, start + duration, color=f"#{phase_color}", alpha=0.14, linewidth=0)
    axes[0].text(start - 0.05, 18.5, "脉冲", color=f"#{phase_color}", fontsize=10, ha="right")
    axes[0].set(xlim=(0, 2.0), ylim=(0, 20), xlabel="搬运时间（s）", ylabel="水平误差（mm）")
    axes[1].set(xlim=(0, 100), ylim=(0, 24), xlabel="距首次触桌（ms）", ylabel="竖直承托力（N）")
    for ax in axes:
        ax.grid(axis="y", alpha=0.24, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(f"#{LINE}")
        ax.tick_params(length=3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    figure.savefig(result, facecolor=f"#{WHITE}")
    plt.close(figure)
    return result


def create_deck():
    summary = json.loads((PILOT / "summary.json").read_text(encoding="utf-8"))
    records = {r["method"]: r for r in summary["episodes"] if r["case"] == "standard_pulse"}
    graph = pilot_plot(records)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13 + 1 / 3), Inches(7.5)
    prs.core_properties.title = "面向抗扰搬运与柔顺放置的机械臂控制策略学习"
    prs.core_properties.subject = "机器人动力学与控制课程期末项目选题"

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(DARK)
    box(slide, 0.66, 0.72, 0.44, 0.047, MINT)
    label(slide, 0.66, 0.88, 6.0, 0.33, "机器人动力学与控制课程 · 期末项目选题", 14, MINT, True)
    label(slide, 0.66, 1.66, 6.05, 1.94, "抗扰搬运\n与柔顺放置", 47, WHITE, True)
    label(slide, 0.69, 3.87, 5.93, 0.92, "基于任务空间变阻抗的\n机械臂控制策略学习", 25, WHITE)
    box(slide, 0.69, 5.36, 5.77, 0.012, "486471")
    label(slide, 0.69, 5.62, 5.77, 0.79, "搬运中抵抗外力，落台时控制冲击。\n怎样在同一回合里协调两者？", 17, PALE)
    pic(slide, PILOT / "grasp.png", 6.70, 1.49, 5.99, 3.744)
    label(slide, 6.70, 5.44, 6.00, 0.35, "MuJoCo 自建场景 · Franka Panda 双指夹持", 13, PALE)
    notes(slide, "以搬运中的跟踪与落台瞬间的接触力为两个目标。用一个物理仿真回合把它们连起来。")

    slide = page(prs, "选题依据", "同一次搬放，需要兼顾两种性能", "依据  [1][2] 阻抗与操作空间 · [4] 抗扰搬运 · [13][15] 接触冲击")
    box(slide, 0.64, 1.70, 5.88, 3.63, WHITE, LINE)
    box(slide, 6.80, 1.70, 5.89, 3.63, WHITE, LINE)
    for x, fill, accent, title, premise, body, metric in (
        (0.64, TEAL_LIGHT, TEAL, "搬运要稳", "悬空横移时，物块受到水平外力",
         "较高的水平刚度可减小低频偏差。\n短脉冲下还取决于阻尼、惯量与摩擦。", "量测  轨迹偏差与掉落"),
        (6.80, AMBER_LIGHT, AMBER, "放下要轻", "实际台面高度变化，触桌时刻随之改变",
         "接触峰值受法向刚度和触前速度影响。\n还要确认承托，再开夹并抬手。", "量测  峰值力、冲量与放稳"),
    ):
        box(slide, x, 1.70, 5.88, 0.69, fill)
        label(slide, x + 0.26, 1.81, 5.30, 0.48, title, 23, accent, True)
        label(slide, x + 0.26, 2.62, 5.30, 0.45, premise, 17, INK, True)
        label(slide, x + 0.26, 3.24, 5.30, 0.85, body, 16, MUTED)
        box(slide, x + 0.26, 4.42, 5.30, 0.012, LINE)
        label(slide, x + 0.26, 4.59, 5.30, 0.38, metric, 16, accent, True)
    box(slide, 0.64, 5.65, 12.05, 0.88, DARK)
    label(slide, 0.94, 5.89, 11.37, 0.43, "统一下降轨迹后，反馈调刚度能否改善搬运偏差与触桌冲击的权衡？", 20, WHITE, True)
    notes(slide, "小扰动且准静态时，位置偏差大致与水平刚度成反比。接触峰值还有接近速度、夹持载荷、台面和求解器等因素。固定分轴可以同时设水平强和法向软，需要把它作为强基线。")

    slide = page(prs, "相关工作", "阻抗学习有依据，搬放权衡仍需实测", "引用编号见第 9 页与《文献调研》全文")
    rows = (
        ("阻抗与抗扰", "Hogan / Khatib [1][2] · Carey & Werfel [4]", "用任务空间阻抗调力–运动关系，外扰下调柔顺度。"),
        ("学习变阻抗", "VICES [6] · Bogdanovic 等 [7]", "在擦桌、开门等任务中比较增益或动作表示的影响。"),
        ("接触与放置", "Kazemi 等 [14] · Arita 等 [15] · Li 等 [17]", "用承托反馈释放，用接触前减速处理力峰值。"),
        ("相近系统评测", "Surendran & Peri [18] · CoRL 2026 Workshop", "MuJoCo 放瓶等任务中，调优阻抗与位姿残差各有适用工况。"),
    )
    for i, (topic, origin, finding) in enumerate(rows):
        y = 1.70 + i * 0.99
        box(slide, 0.64, y, 12.05, 0.89, WHITE, LINE)
        label(slide, 0.86, y + 0.20, 2.03, 0.42, topic, 18, TEAL, True)
        label(slide, 3.01, y + 0.075, 9.37, 0.37, origin, 15, INK, True)
        label(slide, 3.01, y + 0.46, 9.37, 0.35, finding, 15, MUTED)
    box(slide, 0.64, 5.85, 12.05, 0.72, TEAL_LIGHT)
    label(slide, 0.86, 6.02, 11.65, 0.4, "本课任务  物块外力脉冲 + 隐藏的实际桌高，在同一条搬放流程中检验二维调刚度。", 16, TEAL, True)
    notes(slide, "选题采用自建任务，底层从 Hogan 和 Khatib 的阻抗及任务空间思想出发，上层参考 VICES。Kazemi 等已有抓取、搬运和柔顺释放流程。Surendran 与 Peri 的研讨会工作用的是接触方向参考偏差，本课用物块外力与实际台高偏移，学习的对象是刚度。")

    slide = page(prs, "自建任务", "抬起 50 mm 方块，越过间隙再落台", "MuJoCo 3.3.7 [22] · MuJoCo Menagerie 的 Panda 模型 [23] · 图为本工程实际渲染")
    frames = (
        ("disturbance.png", "受扰横移", "横移 0.42 m，物块承受外力脉冲"),
        ("contact.png", "减速接近与触桌", "实际台高可变，接触力逐物理步记录"),
        ("settled.png", "承托后开夹", "检测持续支撑，释放并抬手 0.10 m"),
    )
    for i, (name, title, body) in enumerate(frames):
        x = 0.64 + i * 4.08
        box(slide, x, 1.70, 3.89, 3.60, WHITE, LINE)
        pic(slide, PILOT / name, x + 0.045, 1.745, 3.80, 2.375)
        label(slide, x + 0.17, 4.27, 3.58, 0.42, title, 20, INK, True)
        label(slide, x + 0.17, 4.82, 3.58, 0.29, body, 12, MUTED)
    box(slide, 0.64, 5.60, 12.05, 0.99, TEAL_LIGHT)
    label(slide, 0.87, 5.79, 11.57, 0.38, "从已夹稳并抬升 0.14 m 开始计时，夹指与物块由摩擦接触传力。", 18, TEAL, True)
    label(slide, 0.87, 6.21, 11.57, 0.26, "标准工况  0.25 kg · 水平脉冲 8 N × 80 ms · 目标台面标称高度 0.12 m", 13, MUTED)
    notes(slide, "抓取初始姿态由 IK 设置，闭夹和抬升由物理仿真完成。方块是独立的自由关节刚体，扰动力施加于其质心。实验聚焦已经抓住之后的运输、接触、释放三个阶段。橙色箭头显示脉冲的方向。")

    slide = page(prs, "初步方法", "上层学习两组刚度，下层执行力矩控制", "方法依据  [1][2] 笛卡尔阻抗 · [6] 末端变阻抗接口 · [8] 残差学习 · [21] SAC")
    modules = (
        (0.64, 3.70, "观测", "关节与末端运动状态\n名义间隙、阶段、低通承托力"),
        (4.80, 3.53, "SAC · 50 Hz", "对规则增益加有界残差\n分别调水平 k_xy 和法向 k_z"),
        (8.80, 3.89, "笛卡尔阻抗 · 500 Hz", "位置/速度误差生成任务空间力\n用雅可比转置映射为关节力矩"),
    )
    for x, w, title, body in modules:
        box(slide, x, 1.84, w, 2.05, WHITE, LINE)
        box(slide, x, 1.84, w, 0.06, TEAL)
        label(slide, x + 0.18, 2.04, w - 0.36, 0.46, title, 20, TEAL, True)
        label(slide, x + 0.18, 2.74, w - 0.36, 0.80, body, 14.5, INK)
    label(slide, 4.42, 2.54, 0.34, 0.46, "→", 20, MUTED, True)
    label(slide, 8.43, 2.54, 0.34, 0.46, "→", 20, MUTED, True)
    box(slide, 0.64, 4.18, 12.05, 1.14, DARK)
    label(slide, 0.88, 4.36, 11.60, 0.46,
          "F = K(x_ref − x) + D(v_ref − v)          K = diag(k_xy, k_xy, k_z)", 21, WHITE, True)
    label(slide, 0.89, 4.96, 11.58, 0.32, "增益限幅、平滑和限速，力矩按关节饱和。各方法共用轨迹、姿态与释放规则。", 14, PALE)
    label(slide, 0.66, 5.58, 11.90, 0.40, "奖励兼顾跟踪误差、首触峰值与冲量、控制平滑和最终成功。", 18, INK, True)
    label(slide, 0.66, 6.16, 11.90, 0.33, "承托力取自仿真接触信息。策略使用名义间隙，实际台高与外力参数留给评测。", 14, MUTED)
    notes(slide, "SAC 读取运动状态、物块相对位置和速度、名义间隙与低通承托力，输出两个归一化动作。乘法残差在标称增益上限幅，等效质量近似给出阻尼并叠加固定姿态反馈。变刚度限幅和限速用于实证试验，严格无源性另行检验。")

    slide = page(prs, "任务目标与评价", "搬运、触桌和放稳分别计量", "首触窗口从目标台对物块出现正支撑力时开始，每 2 ms 采样一次")
    cards = (
        (0.64, "搬运误差", "mm", "记录水平偏差峰值、RMS 和掉落，\n另统计脉冲至结束后 350 ms 的偏差。", TEAL),
        (4.72, "接触冲击", "N / N·s", "首触后 100 ms 取竖直力峰值与总冲量，\n同时记录物块触前垂向速度。", AMBER),
        (8.80, "整任务成功", "成功数 / 尝试数", "6.8 s 内开夹并抬手。松夹后，\n物块在目标区持续稳定 0.30 s。", TEAL),
    )
    for x, title, unit, body, accent in cards:
        box(slide, x, 1.76, 3.89, 3.22, WHITE, LINE)
        label(slide, x + 0.20, 2.00, 3.46, 0.44, title, 22, accent, True)
        label(slide, x + 0.20, 2.66, 3.46, 0.55, unit, 22, INK, True)
        label(slide, x + 0.20, 3.51, 3.65, 0.96, body, 13.3, MUTED)
    box(slide, 0.64, 5.24, 12.05, 0.73, TEAL_LIGHT)
    label(slide, 0.88, 5.43, 11.57, 0.37,
          "放稳阈值  水平误差 ≤25 mm · 倾角 ≤7° · 线速 ≤0.015 m/s · 角速 ≤0.15 rad/s", 15.5, TEAL, True)
    label(slide, 0.67, 6.23, 11.98, 0.33, "未触桌记为缺失的力数据，仍纳入失败率。并报触桌数和完整测量窗口数。", 15, MUTED)
    notes(slide, "搬运还计算从脉冲起点至结束后 350 ms 的窗口指标，无扰对照匹配这个时间窗。Fz 只汇总目标台对物块的接触反力的世界竖直分量。成功要求夹指脱离、目标面接触、位置和姿态到位以及低速驻留。两种控制器的力比较还包括共有有效窗口的配对差。")

    slide = page(prs, "实验规划", "先选强规则基线，再看学习的增量", "物理条件、参考轨迹与释放守卫配对共享。正式评测在冻结的留出工况中进行。")
    box(slide, 0.64, 1.75, 7.12, 0.52, DARK)
    label(slide, 0.85, 1.82, 2.22, 0.39, "对照方法", 15, WHITE, True)
    label(slide, 3.25, 1.82, 4.24, 0.39, "本组检验的问题", 15, WHITE, True)
    comparisons = (
        ("固定分轴阻抗", "扫描两轴刚度，保留验证集 Pareto 前沿"),
        ("规则分阶段阻抗", "用共同的间隙和力信息调规则"),
        ("规则 + SAC", "给定相同下降轨迹，测二维残差收益"),
        ("力反馈消融", "训练时屏蔽调参反馈，释放守卫保留"),
    )
    for i, (name, test) in enumerate(comparisons):
        y = 2.27 + i * 0.63
        box(slide, 0.64, y, 7.12, 0.63, WHITE if i % 2 == 0 else "E9F1F1", LINE)
        label(slide, 0.85, y + 0.13, 2.35, 0.32, name, 14.5, TEAL, True)
        label(slide, 3.25, y + 0.13, 4.45, 0.32, test, 13.3, INK)
    box(slide, 8.00, 1.75, 4.69, 3.04, TEAL_LIGHT)
    label(slide, 8.22, 1.96, 4.20, 0.42, "留出 96 个工况", 21, TEAL, True)
    label(slide, 8.22, 2.61, 4.16, 1.05, "域内、域外各配无扰与加扰\n变化量  质量、摩擦、实际桌高、脉冲\n每种学习设置使用 3 个训练种子", 14, INK)
    label(slide, 8.22, 3.97, 4.16, 0.39, "同一工况重放并比较配对差", 14, TEAL, True)
    box(slide, 0.64, 5.09, 12.05, 1.46, WHITE, LINE)
    label(slide, 0.88, 5.34, 11.59, 0.42, "验证集调参 → SAC 与消融训练 → 留出复测", 20, INK, True)
    label(slide, 0.88, 5.98, 11.59, 0.36, "CPU 求解物理。本机 8 张 A100 用于并行独立运行，每个学习进程使用 1 张卡。", 14, MUTED)
    notes(slide, "基线在统一验证集调参，固定分轴网格为 4×5。测试按域内和域外各 24 对无扰/加扰配置设计。先校验物理步长对峰值的敏感性，再按每种子 30 万步预算训练。报告留出集成功率、误差和峰值力前沿，记录实际触前垂速解释速度机制。")

    slide = page(prs, "原型试跑", "搬放流程跑通，触桌峰值仍需调参", "以下为标准脉冲的两个联调回合。完整试跑记录见 results/smoke/summary.json 与逐步 CSV。")
    box(slide, 0.64, 1.79, 8.47, 3.64, WHITE, LINE)
    pic(slide, graph, 0.72, 1.89, 8.31, 3.447)
    box(slide, 9.32, 1.79, 3.37, 3.64, TEAL_LIGHT)
    label(slide, 9.53, 2.02, 3.02, 0.68, "8 / 8", 35, TEAL, True)
    label(slide, 9.54, 2.78, 2.98, 0.71, "两种基线 × 四种工况\n均完成放稳", 15.5, TEAL, True)
    box(slide, 9.54, 3.65, 2.91, 0.012, LINE)
    label(slide, 9.54, 3.87, 2.94, 0.95, "另注入强扰动，\n掉落判定按预期触发。", 15, INK)
    fixed, phase = records["fixed"], records["phase"]
    desc = (f"规则调度初值触桌 {phase['peak_force_n']:.1f} N，固定初值 {fixed['peak_force_n']:.1f} N。"
            f"实际触前垂速分别为 {abs(phase['contact_speed_z_m_s'])*1000:.1f} 和 {abs(fixed['contact_speed_z_m_s'])*1000:.1f} mm/s。")
    label(slide, 0.66, 5.75, 12.01, 0.42, desc, 15.5, INK, True)
    label(slide, 0.66, 6.22, 12.01, 0.36, "正式比较关注给定相同参考时，刚度、实际触前速度和峰值力如何一起变化。", 15, MUTED)
    notes(slide, "初值的多项指标方向并不一致，单次曲线只验证能量测轨迹、触桌和释放，不能用于控制器排序。下一步在独立验证集调好两种规则基线，再用学习方法、消融与留出工况比较。")

    slide = page(prs, "参考文献与开源引用", "动力学、接触与学习的主要来源", "完整 20 篇研究与算法、开源项目条目，见配套《文献调研》。下列书目可点击。")
    sources = (
        ("[1] Hogan · 阻抗控制理论", "ASME JDSMC 1985", "https://doi.org/10.1115/1.3140702"),
        ("[2] Khatib · 操作空间力/运动表述", "IEEE JRA 1987", "https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf"),
        ("[4] Carey & Werfel · 自适应柔顺搬运", "ICRA 2021", "https://ntrs.nasa.gov/citations/20240003652"),
        ("[6] Martín-Martín 等 · VICES", "IROS 2019", "https://arxiv.org/abs/1906.08880"),
        ("[7] Bogdanovic 等 · 接触任务变阻抗", "IEEE RA-L 2020", "https://arxiv.org/abs/1907.07500"),
        ("[8] Johannink 等 · 残差强化学习", "ICRA 2019", "https://arxiv.org/abs/1812.03201"),
        ("[14] Kazemi 等 · 柔顺抓取与释放", "RSS 2012", "https://www.roboticsproceedings.org/rss08/p23.html"),
        ("[15] Arita 等 · 触前减冲与接触过渡", "IEEE T-RO 2023", "https://arxiv.org/abs/2212.03545"),
        ("[17] Li 等 · 几何不确定下的稳定放置", "IEEE RA-L 2026", "https://arxiv.org/abs/2409.17725"),
        ("[18] Surendran & Peri · 放瓶控制器对照", "CoRL Workshop 2026", "https://rai-inst.com/resources/workshop/an-empirical-analysis-of-controller-selection-for-safe-execution-of-umi-style-policies/"),
        ("[21] Haarnoja 等 · Soft Actor-Critic", "ICML 2018", "https://proceedings.mlr.press/v80/haarnoja18b.html"),
        ("[22] Todorov 等 · MuJoCo", "IROS 2012", "https://doi.org/10.1109/IROS.2012.6386109"),
    )
    for i, (name, venue, url) in enumerate(sources):
        x = 0.64 + 6.14 * (i // 6)
        y = 1.68 + 0.67 * (i % 6)
        box(slide, x, y + 0.61, 5.91, 0.009, LINE)
        label(slide, x, y, 5.89, 0.32, name, 14.3, TEAL, True, url=url)
        label(slide, x, y + 0.32, 5.89, 0.27, venue, 10.5, MUTED)
    box(slide, 0.64, 5.87, 12.05, 0.70, DARK)
    label(slide, 0.85, 6.07, 4.42, 0.33, "[23] Menagerie / Panda · Apache-2.0", 13, WHITE, True,
          url="https://github.com/google-deepmind/mujoco_menagerie/tree/fb36886b3023db08c198264e1e9956ab202c3e3b/franka_emika_panda")
    label(slide, 5.57, 6.07, 2.74, 0.33, "[24] Gymnasium", 13, WHITE, True,
          url="https://arxiv.org/abs/2407.17032")
    label(slide, 8.46, 6.07, 3.96, 0.33, "[25] Stable-Baselines3 / PyTorch", 13, WHITE, True,
          url="https://jmlr.org/papers/v22/20-1364.html")
    notes(slide, "项目使用原始 Menagerie Panda 模型资产，版本和许可随工程保留。场景、控制器、奖励和指标在本项目实现。图片是原型仿真渲染，曲线由原始 CSV 生成。")

    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > prs.slide_width or shape.top + shape.height > prs.slide_height:
                raise ValueError(f"Shape outside slide: {shape.name}")
    prs.save(OUT / "机器人期末项目_选题说明.pptx")


if __name__ == "__main__":
    create_deck()
