"""Migrate the proposal to native pages from the supplied PKU template library."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt

from create_deck import OUT, PILOT, box, label, notes, pic


COVER_PAGE, HEADER_PAGE = 21, 196
SLIDE_COUNT = 10
PPT_MASTER_SCRIPTS = Path.home() / ".codex/skills/ppt-master/scripts"
RED, GOLD = "94070A", "D7C8B5"
INK, GRAY = "282323", "5F5352"
WHITE, PINK, SAND = "FFFFFF", "F8EEEE", "F5EFE8"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def clone_shape(source, slide):
    element = deepcopy(source._element)
    for node in list(element.iter()):
        if node.tag in {qn("a:hlinkClick"), qn("a:hlinkMouseOver")}:
            node.getparent().remove(node)
            continue
        for key, old_rid in list(node.attrib.items()):
            if not key.startswith(REL_NS):
                continue
            rel = source.part.rels[old_rid]
            if source.part.package is not slide.part.package or rel.is_external:
                raise ValueError(f"Unsupported relationship on {source.name}: {rel.reltype}")
            node.set(key, slide.part.relate_to(rel.target_part, rel.reltype))
    for index, node in enumerate(element.iter(qn("p:cNvPr")), start=slide.shapes._next_shape_id):
        node.set("id", str(index))
    slide.shapes._spTree.insert_element_before(element, "p:extLst")
    return slide.shapes[-1]


def fill_text(shape, lines, size=None):
    paragraphs = shape.text_frame.paragraphs
    if len(lines) != len(paragraphs):
        raise ValueError(f"Text line count changed for template shape {shape.name}")
    for line, paragraph in zip(lines, paragraphs):
        first, *others = paragraph.runs
        first.text = line
        if size:
            first.font.size = Pt(size)
        for run in others:
            paragraph._p.remove(run._r)


def write(slide, x, y, w, h, value, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, url=None):
    shape = label(slide, x, y, w, h, value, size, color, bold, align, url)
    for tag, font in (("a:latin", "Arial"), ("a:ea", "微软雅黑")):
        for item in shape._element.iter(qn(tag)):
            item.set("typeface", font)
    return shape


def add_cover(presentation, layout, cover):
    slide = presentation.slides.add_slide(layout)
    # Keep the template's cropped photo, diagonal accent, heading styles and university mark.
    for index in (0, 1, 6, 7, 8, 9):
        shape = clone_shape(cover.shapes[index], slide)
        if index == 7:
            fill_text(shape, ("面向抗扰搬运与柔顺放置的", "机械臂控制策略学习"), size=38)
        if index == 8:
            fill_text(shape, ("机器人动力学与控制课程 · 期末项目选题",), size=17)
    return slide


def page(presentation, layout, header, section, headline=None, *, page_number=None):
    slide = presentation.slides.add_slide(layout)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(WHITE)
    for index, shape in enumerate(header.shapes):
        imported = clone_shape(shape, slide)
        if index == 3:
            fill_text(imported, (section,))
    if headline:
        write(slide, 0.64, 1.08, 12.05, 0.53, headline, 27, bold=True)
    number = page_number if page_number is not None else len(presentation.slides)
    write(slide, 11.94, 7.01, 0.75, 0.26, f"{number:02d} / {SLIDE_COUNT:02d}", 10, GRAY,
          align=PP_ALIGN.RIGHT)
    return slide


def research_question(slide, trial_count, challenge_count):
    for x, tint, title, cause, goal in (
        (0.64, PINK, "受扰搬运与对准", "外力可持续到下降与触桌", "让物块完整落在承托台内"),
        (6.80, SAND, "柔顺放置", "台高变化使触桌时刻不确定", "降低首触峰值力与冲量"),
    ):
        box(slide, x, 1.79, 5.89, 2.20, WHITE, GOLD)
        box(slide, x, 1.79, 5.89, 0.60, tint)
        write(slide, x + 0.24, 1.89, 5.37, 0.45, title, 21, RED, True)
        write(slide, x + 0.24, 2.60, 5.37, 0.37, cause, 17, GRAY)
        write(slide, x + 0.24, 3.20, 5.37, 0.44, goal, 20, bold=True)
    box(slide, 0.64, 4.23, 12.05, 0.68, RED)
    write(slide, 0.91, 4.37, 11.50, 0.43, "能否兼顾完整落台、触桌冲击和完成用时？", 19, WHITE, True)
    progress = (
        ("已完成", f"{trial_count} 回合扰动筛查\n{challenge_count} 回合挑战测试"),
        ("方法思路", "提前减速与承托后回中\nSAC 学习刚度残差"),
        ("待开展", "回中再开夹的规则对照\n学习策略正式训练与消融"),
    )
    for i, (status, content) in enumerate(progress):
        x = 0.64 + 4.08 * i
        box(slide, x, 5.14, 3.89, 1.43, WHITE, GOLD)
        write(slide, x + 0.19, 5.28, 3.50, 0.39, status, 17, RED, True)
        write(slide, x + 0.19, 5.78, 3.50, 0.64, content, 14.5)
    notes(slide, f"已完成仿真原型、{trial_count} 回合等冲量扰动筛查、{challenge_count} 回合窄台与跨触桌扰动对照，以及 SAC 训练接口联调。下一步针对高台冲击与窄台越边，先做提前减速和回中再释放的规则对照，再验证学习收益。")


def related_work(slide):
    rows = (
        ("阻抗控制", "Hogan [1] · Khatib [2]", "用刚度与阻尼描述末端力和运动的关系"),
        ("学习变阻抗", "Martín-Martín 等 [3] · Bogdanovic 等 [4] · Johannink 等 [5]", "学习阻抗增益或在基控制器上加学习残差"),
        ("接触与放置", "Kazemi 等 [6] · Arita 等 [7] · Surendran 与 Peri [8]", "通过柔顺释放、触前减速与调参降低接触风险"),
    )
    for index, (title, citation, finding) in enumerate(rows):
        y = 1.79 + 1.31 * index
        box(slide, 0.64, y, 12.05, 1.12, WHITE, GOLD)
        box(slide, 0.64, y, 0.045, 1.12, RED)
        write(slide, 0.90, y + 0.25, 3.00, 0.45, title, 19, RED, True)
        write(slide, 3.52, y + 0.13, 8.90, 0.32, citation, 15.5, bold=True)
        write(slide, 3.52, y + 0.62, 8.90, 0.32, finding, 15, GRAY)
    box(slide, 0.64, 5.87, 12.05, 0.61, PINK)
    write(slide, 0.89, 6.00, 11.57, 0.36, "本项目在同一搬放任务中改变施力时程与实际台高。", 17, RED, True)
    notes(slide, "底层采用操作空间与阻抗控制思想，上层参考末端变阻抗和残差强化学习。既有工作为接触前减速和承托后释放提供了依据。正式实验将同步量测受扰搬运与首次触桌冲击。")


def simulation_environment(slide):
    pic(slide, PILOT / "grasp.png", 0.64, 1.80, 6.29, 3.93125)
    items = (
        ("物理仿真", "MuJoCo 3.3.7 · 物理步长 2 ms\n记录台面对物块的竖直接触力"),
        ("机械臂模型", "Franka Panda 7 轴 · 双指夹爪\n模型来自 MuJoCo Menagerie"),
        ("训练接口", "Gymnasium · Stable-Baselines3 / PyTorch\n已完成 SAC 短步联调"),
    )
    for i, (title, body) in enumerate(items):
        y = 1.80 + 1.49 * i
        box(slide, 7.18, y, 5.51, 1.31, PINK if i == 0 else WHITE, GOLD)
        write(slide, 7.43, y + 0.11, 4.97, 0.40, title, 18, RED, True)
        write(slide, 7.43, y + 0.59, 4.97, 0.59, body, 15, INK)
    integration = json.loads((PILOT.parent / "integration/training.json").read_text(encoding="utf-8"))
    notes(slide, "机器人模型来自 MuJoCo Menagerie，提交 fb36886b3023db08c198264e1e9956ab202c3e3b，采用 Apache-2.0 许可。台面、物块与任务逻辑为自建，承托力由仿真接触数据得到。"
          f"SAC 接口联调执行了 {integration['policy_steps']} 个策略步和 {integration['updates']} 次梯度更新，验证了训练流程。策略效果将在正式实验中评估。")


def task_scene(slide):
    frames = (
        ("disturbance.png", "受扰横移", "横移 0.42 m，物块受横向脉冲"),
        ("contact.png", "减速触桌", "目标台实际高度可变"),
        ("settled.png", "释放并抬手", "确认持续承托后开夹"),
    )
    for i, (filename, title, caption) in enumerate(frames):
        x = 0.64 + i * 4.08
        box(slide, x, 1.78, 3.89, 3.61, WHITE, GOLD)
        box(slide, x, 1.78, 3.89, 0.047, RED)
        pic(slide, PILOT / filename, x + 0.045, 1.875, 3.80, 2.375)
        write(slide, x + 0.16, 4.43, 3.57, 0.39, title, 19, RED, True)
        write(slide, x + 0.16, 4.91, 3.57, 0.34, caption, 14, GRAY)
    box(slide, 0.64, 5.75, 12.05, 0.66, PINK)
    write(slide, 0.89, 5.90, 11.57, 0.37, "标准工况  50 mm 方块 · 0.25 kg · 横向脉冲 8 N × 80 ms", 16.5, RED, True)
    notes(slide, "三帧截图来自已跑通的单次脉冲搬放原型。脚本夹稳并将物块抬升 0.14 m 后计时。横向脉冲施加在物块质心。接近时按标称台面减速，承托力达到阈值并持续约 80 ms 后开夹，再抬手 0.10 m。")


def add_equation(slide, name, latex, x, y, width, height, points):
    if str(PPT_MASTER_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PPT_MASTER_SCRIPTS))
    from svg_to_pptx.drawingml.context import ConvertContext
    from svg_to_pptx.native_objects.formula import build_native_formula

    bounds = {f"data-pptx-{key}": str(96 * value) for key, value in
              zip(("x", "y", "width", "height"), (x, y, width, height))}
    marker = ElementTree.Element("g", bounds)
    context = ConvertContext(id_counter=slide.shapes._next_shape_id)
    equation = build_native_formula(marker, context, {"name": name, "latex": latex,
                                    "font_size": points * 96 / 72, "color": f"#{INK}"})
    element = parse_xml(f'<shape {nsdecls("a", "p")}>{equation.xml}</shape>')[0]
    slide.shapes._spTree.insert_element_before(element, "p:extLst")


def control_method(slide):
    modules = (
        (0.64, "观测", "跟踪误差、名义间隙\n低通承托力"),
        (4.86, "SAC · 50 Hz [9]", "在规则增益上加有界残差\n调水平与法向刚度"),
        (9.08, "笛卡尔阻抗 · 500 Hz", "误差生成末端力\n用雅可比转置映射为关节力矩"),
    )
    for x, title, body in modules:
        box(slide, x, 1.91, 3.61, 2.06, WHITE, GOLD)
        box(slide, x, 1.91, 3.61, 0.06, RED)
        write(slide, x + 0.20, 2.14, 3.21, 0.42, title, 19.5, RED, True)
        write(slide, x + 0.20, 2.85, 3.21, 0.79, body, 15.5)
    write(slide, 4.39, 2.59, 0.39, 0.41, "→", 21, GRAY)
    write(slide, 8.61, 2.59, 0.39, 0.41, "→", 21, GRAY)
    box(slide, 0.64, 4.28, 7.24, 1.63, PINK, GOLD)
    add_equation(slide, "末端平移力", r"\mathbf{F}=\mathbf{K}(\mathbf{x}_{\mathrm{ref}}-\mathbf{x})+\mathbf{D}(\mathbf{v}_{\mathrm{ref}}-\mathbf{v})",
                 0.84, 4.82, 6.84, 0.59, 26)
    box(slide, 8.12, 4.28, 4.57, 1.63, SAND, GOLD)
    add_equation(slide, "分轴刚度矩阵", r"\mathbf{K}=\mathrm{diag}(k_{xy},\,k_{xy},\,k_z)",
                 8.32, 4.82, 4.17, 0.59, 24)
    write(slide, 0.86, 4.41, 6.79, 0.31, "末端平移力", 16, RED, True)
    write(slide, 8.34, 4.41, 4.12, 0.31, "分轴刚度矩阵", 16, RED, True)
    write(slide, 0.86, 5.52, 6.79, 0.29, "刚度 × 位置误差 + 阻尼 × 速度误差", 13.4, GRAY)
    write(slide, 8.34, 5.52, 4.12, 0.29, "水平 x、y 共用刚度，竖直 z 单独设定", 13.4, GRAY)
    write(slide, 0.66, 6.20, 12.00, 0.42, "底层阻抗已用于基线评估，SAC 残差效果待正式验证。", 18, bold=True)
    notes(slide, "方法思路是用 SAC 给规则刚度增加两个有界残差。策略还读取关节状态、物块相对位置与速度及任务阶段。阻尼由等效质量近似确定。增益平滑限速、关节力矩饱和。奖励兼顾跟踪精度、触桌冲击和放置成功。桌高、质量和扰动参数不直接提供给策略。")


def experiment_design(slide):
    for x, width, title in ((0.64, 6.48, "控制对照"), (7.38, 5.31, "工况与指标")):
        box(slide, x, 1.80, width, 0.50, RED)
        write(slide, x + 0.19, 1.87, width - 0.38, 0.34, title, 16.5, WHITE, True)
    controls = (
        ("固定分轴阻抗", "以 520 / 300 N/m 作对照"),
        ("回中再开夹", "确认承托，回到台内后开夹"),
        ("提前减速", "检验高台首触冲击与用时"),
        ("规则 + SAC", "与调参规则共用轨迹和释放条件"),
    )
    for i, (name, change) in enumerate(controls):
        y = 2.30 + i * 0.74
        box(slide, 0.64, y, 6.48, 0.74, WHITE if i % 2 == 0 else SAND, GOLD)
        write(slide, 0.84, y + 0.17, 2.55, 0.35, name, 15.5, RED, True)
        write(slide, 3.45, y + 0.18, 3.40, 0.35, change, 14.5)
    metrics = (
        ("窄台", "4 / 8 N 恒力下的完整放稳率与边距"),
        ("触桌", "触前垂速、首触峰值与冲量"),
        ("效率", "等待开夹与完整放稳的用时"),
    )
    for i, (name, value) in enumerate(metrics):
        y = 2.30 + i * (2.96 / 3)
        box(slide, 7.38, y, 5.31, 2.96 / 3, WHITE if i % 2 == 0 else SAND, GOLD)
        write(slide, 7.57, y + 0.31, 0.81, 0.37, name, 15.5, RED, True)
        write(slide, 8.45, y + 0.33, 4.02, 0.35, value, 13.7)
    box(slide, 0.64, 5.57, 12.05, 0.88, PINK)
    write(slide, 0.88, 5.77, 11.55, 0.45, "先测回中能减少多少越边、减速能降低多少冲击，再比较学习收益。", 17.5, RED, True)
    notes(slide, "这一页是待开展的实验。窄台实验在承托后控制法向下压并横向回中，以物块投影、边距和稳定状态作为开夹条件。对照现有守卫，比较开夹等待、完整放稳和承托段力。高台实验固定水平刚度 520 N/m，交叉比较两档法向刚度和是否提前减速。先调优规则，再评估三个种子的 SAC。力反馈和阶段消融在相同参考轨迹、释放条件及独立工况上进行。")


def challenge_results(slide, summary):
    narrow = {(row["condition"], row["method"]): row for row in summary
              if row["surface"] == "pad_64mm" and row["height_mm"] == "all"}
    methods = ("fixed_360_300", "phase_rule", "fixed_520_300")
    conditions = (
        ("none", "无扰"),
        ("carry_constant_8n", "搬运 · 8 N 恒力"),
        ("descent_single_8n", "下降 · 8 N 单次"),
        ("descent_triple_8n", "下降 · 8 N 三次"),
        ("descent_constant_4n", "下降 · 4 N 恒力"),
        ("descent_constant_8n", "下降 · 8 N 恒力"),
    )
    rows = [("窄台完整放稳数", "固定 360/300", "规则分阶段", "固定 520/300")]
    for condition, title in conditions:
        results = [narrow[condition, method] for method in methods]
        rows.append((title, *(f"{r['full_support']}/{r['attempts']}" for r in results)))
    write(slide, 0.64, 1.81, 12.05, 0.35, "台高 0 / +20 mm · 下降时距台面 20 mm 触发施扰 · 开夹时停止", 14.5, GRAY)
    columns = (2.31, 1.86, 1.86, 1.86)
    for i, row in enumerate(rows):
        x, y = 0.64, 2.39 + 0.48 * i
        for width, value in zip(columns, row):
            fill = RED if i == 0 else PINK if i >= 5 else WHITE
            color = WHITE if i == 0 else RED if i >= 5 else INK
            box(slide, x, y, width, 0.48, fill, GOLD if i else None)
            write(slide, x + 0.045, y + 0.09, width - 0.09, 0.33, value, 14.5, color,
                  i == 0 or i >= 5, align=PP_ALIGN.CENTER)
            x += width
    box(slide, 8.80, 2.39, 3.89, 3.36, SAND, GOLD)
    write(slide, 9.01, 2.57, 3.47, 0.35, "窄台俯视示意", 16, RED, True, align=PP_ALIGN.CENTER)
    pad_size = 1.70
    cube_size = pad_size * 50 / 64
    shift = (pad_size - cube_size) / 2
    box(slide, 9.895, 3.14, pad_size, pad_size, "E9DCD1", GOLD)
    box(slide, 9.895 + shift, 3.14 + shift, cube_size, cube_size, PINK, RED)
    write(slide, 9.99, 3.81, 1.51, 0.36, "50 mm", 16, RED, True, align=PP_ALIGN.CENTER)
    write(slide, 8.99, 5.10, 3.51, 0.35, "台宽 64 mm · 对齐后每侧余 7 mm", 13.7, GRAY, align=PP_ALIGN.CENTER)
    matched = [[narrow[condition, method] for method in methods]
               for condition in ("carry_constant_8n", "descent_constant_8n")]
    counts = [f"{sum(r['full_support'] for r in group)}/{sum(r['attempts'] for r in group)}" for group in matched]
    box(slide, 0.64, 6.03, 12.05, 0.65, PINK)
    write(slide, 0.88, 6.18, 11.55, 0.35,
          f"8 N 恒力  ·  搬运期完整放稳 {counts[0]}  ·  持续至触桌 {counts[1]}", 17, RED, True)
    notes(slide, "宽台和窄台各 288 回合，原先的放稳条件全部满足。宽台完整放稳 288/288，窄台 196/288，失败的 92 回合均有物块水平投影越边。物块相对样本在三种基线、两档台高和六种施扰配置中配对。单次与三次脉冲各次持续 80 ms，三次脉冲的起点相隔 320 ms。持续恒力最长 720 ms，接近期施力在开夹时停止。两种下降恒力组首触时均仍受扰。所有时程按实际施力积分，宽台和窄台使用同一开夹条件。")


def disturbance_results(slide, episodes, summary):
    totals = {row["method"]: row for row in summary if row["timing"] == "all"}
    columns = (2.20, 1.25, 1.75, 1.70, 1.64, 1.71, 1.80)
    rows = [("基线", "放稳数", "搬运 RMS\nmm", "触桌峰值\nN", "触桌冲量\nN·s", "触前垂速\nmm/s", "用时\ns")]
    metrics = ("carry_rms_mm", "touch_peak_n", "touch_impulse_n_s", "touch_speed_mm_s", "settling_time_s")
    for method, name in (("fixed_360_300", "固定 360/300"), ("phase_rule", "规则分阶段"),
                         ("fixed_520_300", "固定 520/300")):
        data = totals[method]
        values = [f"{data[key]['mean']:.{digits}f}" for key, digits in zip(metrics, (2, 2, 3, 1, 2))]
        rows.append((name, f"{data['successes']}/{data['attempts']}", *values))
    write(slide, 0.64, 1.79, 12.05, 0.34, "无扰、单次、三次、持续恒力 · 搬运或下降期间 · 台高 −20 / 0 / +20 mm", 14.5, GRAY)
    for i, row in enumerate(rows):
        x = 0.64
        y = 2.24 if i == 0 else 2.98 + 0.54 * (i - 1)
        for width, value in zip(columns, row):
            box(slide, x, y, width, 0.74 if i == 0 else 0.54, RED if i == 0 else WHITE if i % 2 else PINK,
                GOLD if i else None)
            write(slide, x + 0.06, y + (0.12 if i or "\n" in value else 0.24), width - 0.12, 0.54 if i == 0 else 0.34,
                  value, 14 if i == 0 else 16, WHITE if i == 0 else INK, i == 0, align=PP_ALIGN.CENTER)
            x += width
    write(slide, 0.65, 4.72, 12.04, 0.34, "误差、力与用时取均值。三种有扰组等冲量 0.64 N·s，下降期测 0 / +20 mm 台高。", 13.5, GRAY)

    high_table = [row for row in episodes if row["height_mm"] == 20]
    phase_p90, fixed_p90 = (np.quantile([row["peak_force_n"] for row in high_table if row["method"] == method], 0.9)
                            for method in ("phase_rule", "fixed_520_300"))
    early, late = (np.mean([row["rms_descent_error_m"] for row in episodes if row["method"] == "phase_rule" and
                            row["timing"] == timing and row["profile"] == "single" and row["height_mm"] in (0, 20)])
                   * 1000 for timing in ("carry", "descent"))
    findings = (
        ("规则基线在高台冲击偏大", f"+20 mm 时，规则峰值 P90 为 {phase_p90:.1f} N\n固定 520/300 为 {fixed_p90:.1f} N"),
        ("下降期施扰增大轨迹偏差", f"单次脉冲延后，规则下降 RMS 上升\n{early:.2f} → {late:.2f} mm（相同台高）"),
    )
    for i, (title, detail) in enumerate(findings):
        x = 0.64 + 6.16 * i
        box(slide, x, 5.19, 5.89, 1.42, PINK if i == 0 else SAND, GOLD)
        write(slide, x + 0.20, 5.31, 5.49, 0.36, title, 17, RED, True)
        write(slide, x + 0.20, 5.79, 5.49, 0.63, detail, 14.5)
    notes(slide, "三个基线按相同工况测试。搬运期四种时程与三档台高交叉，下降期三个有扰时程平移 1.50 s，与名义和 +20 mm 台高交叉，每个组合复用八组质量、台面摩擦和力方向。每种基线 144 回合全放稳，触桌峰值力和冲量均取首触后 100 ms。用时从抬升后算到放稳。固定 520/300 N/m 的两项增益分别沿水平和竖直。下降期扰动结束与首触至少相隔 164 ms。下降 RMS 跨名义与 +20 mm 两档台高配对。")


def references(slide):
    papers = (
        ("Hogan · ASME J. Dyn. Syst. Meas. Control, 1985",
         "Impedance Control: An Approach to Manipulation:\nPart I—Theory", "https://doi.org/10.1115/1.3140702"),
        ("Khatib · IEEE J. Robot. Autom., 1987",
         "A Unified Approach for Motion and Force Control of\nRobot Manipulators: The Operational Space Formulation", "https://doi.org/10.1109/JRA.1987.1087068"),
        ("Martín-Martín et al. · IROS, 2019",
         "Variable Impedance Control in End-Effector Space: An Action\nSpace for Reinforcement Learning in Contact-Rich Tasks", "https://arxiv.org/abs/1906.08880"),
        ("Bogdanovic, Khadiv & Righetti · IEEE RA-L, 2020",
         "Learning Variable Impedance Control for Contact\nSensitive Tasks", "https://doi.org/10.1109/LRA.2020.3011379"),
        ("Johannink et al. · ICRA, 2019",
         "Residual Reinforcement Learning for Robot Control", "https://doi.org/10.1109/ICRA.2019.8794127"),
        ("Kazemi et al. · RSS, 2012",
         "Robust Object Grasping using Force Compliant\nMotion Primitives", "https://www.roboticsproceedings.org/rss08/p23.html"),
        ("Arita et al. · IEEE T-RO, 2023",
         "Smoothly Connected Preemptive Impact Reduction\nand Contact Impedance Control", "https://doi.org/10.1109/TRO.2023.3286045"),
        ("Surendran & Peri · CoRL Workshop, 2026",
         "An Empirical Analysis of Controller Selection\nfor Safe Execution of UMI-Style Policies",
         "https://rai-inst.com/resources/workshop/an-empirical-analysis-of-controller-selection-for-safe-execution-of-umi-style-policies/"),
        ("Haarnoja et al. · ICML, 2018",
         "Soft Actor-Critic: Off-Policy Maximum Entropy\nDeep Reinforcement Learning with a Stochastic Actor", "https://proceedings.mlr.press/v80/haarnoja18b.html"),
    )
    for i, (authors, title, url) in enumerate(papers):
        x = 0.64 + 6.16 * (i // 5)
        y = 1.35 + 1.11 * (i % 5)
        write(slide, x, y, 5.88, 0.28, f"[{i + 1}] {authors}", 12.7, RED, True, url=url)
        write(slide, x, y + 0.31, 5.88, 0.75, title, 14, INK)
        box(slide, x, y + 1.035, 5.88, 0.009, GOLD)
    notes(slide, "书目按正文引用顺序排列，点击作者行可打开论文出处。")


def migrate(template, original, output):
    presentation = Presentation(template)
    source = Presentation(original)
    if len(source.slides) != 9 or abs(presentation.slide_width - source.slide_width) > Inches(0.01):
        raise ValueError("This migration expects the original nine-page, widescreen proposal")
    cover, header = (presentation.slides[i - 1] for i in (COVER_PAGE, HEADER_PAGE))
    layout = header.slide_layout
    if cover.slide_layout != layout or len(cover.shapes) != 10 or len(header.shapes) != 5:
        raise ValueError("The PKU template page roster does not match the supplied edition")

    # Retain the actual template master and layout. Unused sample pages and their media leave the package.
    slide_ids = presentation.slides._sldIdLst
    for slide_id in list(slide_ids):
        presentation.part.drop_rel(slide_id.rId)
        slide_ids.remove(slide_id)
    for extension in presentation.part._element.xpath("./p:extLst/*"):
        if any(node.tag.endswith("}sectionLst") for node in extension.iter()):
            extension.getparent().remove(extension)

    benchmark = PILOT.parent / "disturbance_benchmark"
    episodes = json.loads((benchmark / "runs.json").read_text(encoding="utf-8"))["episodes"]
    summary = json.loads((benchmark / "summary.json").read_text(encoding="utf-8"))
    total = len(episodes)
    success = sum(row["is_success"] for row in episodes)
    challenge = json.loads((PILOT.parent / "challenge_benchmark/summary.json").read_text(encoding="utf-8"))
    challenge_count = sum(row["attempts"] for row in challenge if row["height_mm"] == "all")
    first = add_cover(presentation, layout, cover)
    notes(first, source.slides[0].notes_slide.notes_text_frame.text)
    research_question(page(presentation, layout, header, "研究问题与进展", "搬运要稳，落台要轻", page_number=2),
                      total, challenge_count)
    sections = (
        (3, "相关工作", "变阻抗与接触放置", related_work),
        (4, "已完成 · 仿真环境", "MuJoCo 中的 Franka Panda", simulation_environment),
        (5, "已完成 · 搬放原型", "从受扰横移到稳定放置", task_scene),
        (8, "方法思路 · 控制设计", "上层调刚度，下层控力矩", control_method),
        (9, "待开展 · 正式实验", "回中再开夹能减少多少越边？", experiment_design),
    )
    for number, section, headline, draw in sections:
        draw(page(presentation, layout, header, section, headline, page_number=number))
    measured = page(presentation, layout, header, "已完成 · 扰动评估", f"三种基线共 {success}/{total} 回合放稳", page_number=6)
    measured_id = slide_ids[-1]
    disturbance_results(measured, episodes, summary)
    references(page(presentation, layout, header, "参考文献", page_number=10))
    tested = page(presentation, layout, header, "已完成 · 挑战任务", "50 mm 方块要完整落在 64 mm 承托台内", page_number=7)
    challenge_results(tested, challenge)
    # Group completed work ahead of the proposal and keep existing slide IDs.
    slide_ids.insert(5, slide_ids[-1])
    slide_ids.insert(5, measured_id)
    props = presentation.core_properties
    prior = source.core_properties
    props.title, props.subject, props.author = prior.title, prior.subject, prior.author
    props.keywords = prior.keywords
    props.comments = "版式来自用户提供的《北京大学学术答辩PPT速查手册 3.0》。"
    props.last_modified_by = ""
    presentation.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--source", type=Path, default=OUT / "机器人期末项目_选题说明.pptx")
    parser.add_argument("--output", type=Path, default=OUT / "机器人期末项目_选题说明_北大模板精简版.pptx")
    args = parser.parse_args()
    migrate(args.template, args.source, args.output)


if __name__ == "__main__":
    main()
