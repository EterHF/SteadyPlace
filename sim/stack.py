"""Panda and four free blocks on a physically driven slide."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np


TORQUE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12])
BLOCK_COUNT = 4
BLOCK_WIDTH = 0.050
TABLE_Z = 0.12
TRAVEL = 0.105
HOME = np.array([0.0, -0.35, 0.0, -2.15, 0.0, 1.80, np.pi / 4])
DOWN = np.diag([1.0, -1.0, -1.0])
OPEN = 255.0
HOLD = 118.0
PARK = np.array([0.43, -0.16, 0.37])
IMPACT_WINDOW = 0.10


@dataclass
class Placement:
    layer: int
    landing: np.ndarray
    above_tray: np.ndarray
    payload_kg: float
    grasped: bool
    lift_height_mm: float


def cycle_is_stable(cycle):
    return (
        cycle["ordered"]
        and cycle["arm_contact_steps"] == 0
        and cycle["fraction_all_layers_supported"] >= 0.99
    )


def build_model():
    panda_dir = Path(__file__).resolve().parents[1] / "assets/franka_emika_panda"
    robot = ET.parse(panda_dir / "panda.xml").getroot()
    scene = ET.parse(Path(__file__).with_name("scene.xml")).getroot()
    robot.set("model", scene.get("model"))
    robot.find("compiler").set("meshdir", str(panda_dir / "assets"))
    robot.find("option").attrib.update(scene.find("option").attrib)
    robot.remove(robot.find("keyframe"))
    robot.append(scene.find("visual"))
    for section in ("asset", "worldbody"):
        robot.find(section).extend(scene.find(section))
    ET.SubElement(robot.find(".//body[@name='hand']"), "site",
                  name="tcp", pos="0 0 0.1029", size="0.004", rgba="0 0 0 0")
    actuators = robot.find("actuator")
    gripper = actuators[-1]
    actuators.clear()
    for joint, limit in enumerate(TORQUE_LIMIT, start=1):
        ET.SubElement(actuators, "motor", name=f"torque{joint}", joint=f"joint{joint}",
                      ctrlrange=f"{-limit} {limit}")
    gripper.set("gainprm", str(0.04 * 2000 / 255))
    gripper.set("biasprm", "0 -2000 -30")
    actuators.append(gripper)
    actuators.extend(scene.find("actuator"))
    return mj.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))


class StackRun:
    """Scripted pick and place using TCP impedance and encoder-based tray tracking."""

    def __init__(self, period, on_step=None, *, tray_phase=0.0):
        self.model = build_model()
        self.data = mj.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.omega = 2 * np.pi / period
        self.period = period
        self.tray_phase = tray_phase
        self.tcp = self.model.site("tcp").id
        self.slide = self.model.joint("tray_slide").id
        self.slide_q = self.model.jnt_qposadr[self.slide]
        self.slide_v = self.model.jnt_dofadr[self.slide]
        self.tray_drive = self.model.actuator("tray_drive").id
        self.gripper = self.model.actuator("actuator8").id
        self.tray = self.model.body("tray").id
        self.arm_root = self.model.body("link0").id
        self.blocks = [self.model.body(f"block{i}").id for i in range(1, BLOCK_COUNT + 1)]
        self.cubes = [self.model.geom(f"cube{i}").id for i in range(1, BLOCK_COUNT + 1)]
        self.supports = [self.model.geom("tray_top").id, *self.cubes[:-1]]
        self.fingers = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros_like(self.jacp)
        self.force = np.zeros(6)
        self.reference = PARK.copy()
        self.follow = False
        self.on_step = on_step

        self.data.qpos[self.slide_q] = TRAVEL * np.cos(tray_phase)
        self.data.qvel[self.slide_v] = -TRAVEL * self.omega * np.sin(tray_phase)
        self.data.qpos[:7] = HOME
        self.data.qpos[7:9] = 0.04
        self._initialize_arm()

    @property
    def tray_velocity(self):
        return float(self.data.qvel[self.slide_v])

    def _orientation_error(self):
        rotation = self.data.site_xmat[self.tcp].reshape(3, 3)
        return 0.5 * np.cross(rotation.T, DOWN.T).sum(axis=0)

    def _jacobians(self):
        mj.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.tcp)
        return self.jacp[:, :7], self.jacr[:, :7]

    def _initialize_arm(self):
        """Find the initial parked pose. All later arm and tray motion uses actuators."""
        limits = self.model.jnt_range[:7]
        for _ in range(120):
            mj.mj_forward(self.model, self.data)
            error = np.r_[PARK - self.data.site_xpos[self.tcp], self._orientation_error()]
            if np.linalg.norm(error) < 1e-5:
                return
            jacobian = np.vstack(self._jacobians())
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.002 * np.eye(6), error)
            self.data.qpos[:7] += np.clip(delta, -0.07, 0.07)
            np.clip(self.data.qpos[:7], limits[:, 0] + 0.025, limits[:, 1] - 0.025, out=self.data.qpos[:7])
        raise RuntimeError("Park pose is unreachable")

    def tray_reference(self):
        angle = self.omega * self.data.time + self.tray_phase
        return TRAVEL * np.cos(angle), -TRAVEL * self.omega * np.sin(angle)

    def _step(self, position, velocity, grip, payload_kg, vertical_stiffness):
        displacement = self.data.qpos[self.slide_q]
        tray_ref, speed_ref = self.tray_reference()
        # The reciprocating motion is a force-controlled, finite-mass slide.
        self.data.ctrl[self.tray_drive] = (
            -self.model.body_mass[self.tray] * self.omega**2 * tray_ref
            + self.model.dof_damping[self.slide_v] * speed_ref
            + 1100.0 * (tray_ref - displacement) + 62.0 * (speed_ref - self.tray_velocity)
        )

        jacp, jacr = self._jacobians()
        qvel = self.data.qvel[:7]
        stiffness = np.array([1850.0, 1850.0, vertical_stiffness])
        damping = 2.0 * np.sqrt(1.4 * stiffness)
        force = stiffness * (position - self.data.site_xpos[self.tcp]) + damping * (velocity - jacp @ qvel)
        force[2] += payload_kg * 9.81
        moment = 54.0 * self._orientation_error() - 6.0 * (jacr @ qvel)
        torque = jacp.T @ force + jacr.T @ moment + self.data.qfrc_bias[:7] - 0.9 * qvel
        self.data.ctrl[:7] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.data.ctrl[self.gripper] = grip
        mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def move(self, label, target, duration, *, follow=False, grip=OPEN, payload_kg=0.0,
             vertical_stiffness=1000.0, measure=None):
        steps = ceil(duration / self.dt - 1e-9)
        for step in range(steps):
            u = (step + 1) / steps
            progress = u**3 * (10 - 15*u + 6*u*u)
            rate = 30 * u*u * (1-u)**2 / (steps * self.dt)
            position = self.reference + progress * (target - self.reference)
            velocity = rate * (target - self.reference)
            # Blend both frames and their velocities when entering or leaving the tray.
            weight = self.follow * (1 - progress) + follow * progress
            displacement = self.data.qpos[self.slide_q]
            position[1] += weight * displacement
            weight_rate = rate * (int(follow) - int(self.follow))
            velocity[1] += weight * self.tray_velocity + weight_rate * displacement
            self._step(position, velocity, grip, payload_kg, vertical_stiffness)
            if measure is not None:
                measure()
            if self.on_step is not None:
                self.on_step(self, label)
        self.reference = target
        self.follow = follow

    def support_force(self, layer):
        """Normal force for the expected interface, used for evaluation only."""
        pair = {self.cubes[layer], self.supports[layer]}
        normal = 0.0
        for i, contact in enumerate(self.data.contact):
            if pair == {contact.geom1, contact.geom2}:
                mj.mj_contactForce(self.model, self.data, i, self.force)
                normal += self.force[0]
        return float(normal)

    def gripping(self, layer):
        touching_fingers = set()
        for contact in self.data.contact:
            if contact.geom1 == self.cubes[layer]:
                touching_fingers.add(self.model.geom_bodyid[contact.geom2])
            if contact.geom2 == self.cubes[layer]:
                touching_fingers.add(self.model.geom_bodyid[contact.geom1])
        return self.fingers <= touching_fingers

    def _landing_target(self, layer, landing):
        return landing

    def _release_ready(self, layer, landing, payload_kg, measure):
        return True

    def prepare_placement(self, layer):
        """Grasp a block and move it to the tray-following approach pose."""
        source = self.model.body_pos[self.blocks[layer]].copy()
        landing = self.model.body_pos[self.tray].copy()
        landing[2] = TABLE_Z + (layer + 0.5) * BLOCK_WIDTH - 0.0018
        above_source = source + [0.0, 0.0, 0.083]
        above_tray = landing + [0.0, 0.0, 0.084]
        lift = source.copy()
        lift[2] = max(0.27, above_tray[2])
        payload = float(self.model.body_mass[self.blocks[layer]])
        prefix = f"{layer + 1}/{BLOCK_COUNT}"
        self.move(f"{prefix}  Approach block", above_source, 0.78)
        self.move(f"{prefix}  Approach block", source, 0.56)
        self.move(f"{prefix}  Grip", source, 0.32, grip=HOLD)
        grasped = self.gripping(layer)
        self.move(f"{prefix}  Lift", lift, 0.60, grip=HOLD, payload_kg=payload)
        lift_height = float(self.data.xpos[self.blocks[layer], 2] - source[2])
        self.move(f"{prefix}  Match tray motion", above_tray, 1.08, follow=True, grip=HOLD, payload_kg=payload)
        return Placement(layer, landing, above_tray, payload, grasped, 1000 * lift_height)

    def complete_placement(self, placement, *, xy_offset=None, pre_descent_wait=0.0):
        layer = placement.layer
        landing = placement.landing
        above_tray = placement.above_tray
        payload = placement.payload_kg
        prefix = f"{layer + 1}/{BLOCK_COUNT}"
        first_touch = None
        peak_force = 0.0
        impulse = 0.0

        def record_landing():
            nonlocal first_touch, peak_force, impulse
            normal = self.support_force(layer)
            if first_touch is None and normal > 0.05:
                first_touch = self.data.time
            if first_touch is not None and self.data.time - first_touch < IMPACT_WINDOW - 1e-9:
                peak_force = max(peak_force, normal)
                impulse += normal * self.dt

        if pre_descent_wait > 0.0:
            self.move(f"{prefix}  Wait above tower", above_tray, pre_descent_wait,
                      follow=True, grip=HOLD, payload_kg=payload, measure=record_landing)
        self.move(f"{prefix}  Descend with tray", landing + [0.0, 0.0, 0.023], 0.47,
                  follow=True, grip=HOLD, payload_kg=payload, measure=record_landing)
        landing = self._landing_target(layer, landing)
        if xy_offset is not None:
            landing = landing + [*xy_offset, 0.0]
        self.move(f"{prefix}  Touch down", landing, 0.49, follow=True, grip=HOLD,
                  payload_kg=payload, vertical_stiffness=620.0, measure=record_landing)
        self.move(f"{prefix}  Settle", landing, 0.14, follow=True, grip=HOLD,
                  payload_kg=payload, vertical_stiffness=620.0, measure=record_landing)
        released = self._release_ready(layer, landing, payload, record_landing)
        if released:
            self.move(f"{prefix}  Open gripper", landing, 0.24, follow=True,
                      vertical_stiffness=620.0, measure=record_landing)
            self.move(f"{prefix}  Retract", above_tray, 0.55, follow=True, measure=record_landing)
        result = {
            "layer": layer + 1,
            "grasped": placement.grasped,
            "lift_height_mm": placement.lift_height_mm,
            "first_touch_s": first_touch,
            "touch_peak_n_100ms": peak_force if first_touch is not None else None,
            "touch_impulse_ns_100ms": impulse if first_touch is not None else None,
            "supported_after_retract": released and self.support_force(layer) > 0.05,
            "placed_s": self.data.time,
        }
        if not released:
            result["failure"] = "release_gate_timeout"
        return result

    def place(self, layer):
        return self.complete_placement(self.prepare_placement(layer))

    def verify_cycle(self):
        displacement = []
        offsets = []
        tilts = []
        all_supported = 0
        arm_contact = 0
        ordered = True
        acceleration_peak = 0.0
        base_drift = 0.0
        cycle_start = self.data.time
        arm_geoms = set(np.flatnonzero(self.model.body_rootid[self.model.geom_bodyid] == self.arm_root))
        cube_geoms = set(self.cubes)
        base_offset = self.data.xpos[self.blocks[0], :2] - self.data.xpos[self.tray, :2]

        def measure():
            nonlocal all_supported, arm_contact, ordered, acceleration_peak, base_drift
            displacement.append(float(self.data.qpos[self.slide_q]))
            acceleration_peak = max(acceleration_peak, abs(float(self.data.qacc[self.slide_v])))
            block_positions = self.data.xpos[self.blocks]
            current_offset = block_positions[0, :2] - self.data.xpos[self.tray, :2]
            base_drift = max(base_drift, float(np.linalg.norm(current_offset - base_offset)))
            layer_spacing = np.diff(block_positions, axis=0)
            ordered &= bool(np.all(layer_spacing[:, 2] > BLOCK_WIDTH / 2))
            offsets.append(np.linalg.norm(layer_spacing[:, :2], axis=1))
            tilts.append(np.degrees(np.arccos(np.clip(self.data.xmat[self.blocks, 8], -1, 1))))
            all_supported += all(self.support_force(layer) > 0.0 for layer in range(BLOCK_COUNT))
            arm_contact += any(
                (contact.geom1 in cube_geoms and contact.geom2 in arm_geoms)
                or (contact.geom2 in cube_geoms and contact.geom1 in arm_geoms)
                for contact in self.data.contact
            )

        self.move("Post-release motion cycle", PARK, self.period, measure=measure)
        return {
            "duration_s": self.data.time - cycle_start,
            "tray_stroke_mm": 1000 * (max(displacement) - min(displacement)),
            "tray_acceleration_peak_m_s2": acceleration_peak,
            "max_base_drift_mm": 1000 * base_drift,
            "max_interlayer_offset_mm": 1000 * float(np.max(offsets)),
            "max_block_tilt_deg": float(np.max(tilts)),
            "fraction_all_layers_supported": all_supported / len(displacement),
            "arm_contact_steps": arm_contact,
            "ordered": ordered,
        }

    def run(self):
        layers = []
        for layer in range(BLOCK_COUNT):
            layers.append(self.place(layer))
            if "failure" in layers[-1]:
                break
        return self.evaluate_stack(layers)

    def evaluate_stack(self, layers):
        cycle = None if "failure" in layers[-1] else self.verify_cycle()
        success = (
            cycle is not None
            and all(layer["grasped"] and layer["lift_height_mm"] > 65.0
                and layer["supported_after_retract"] for layer in layers)
            and cycle_is_stable(cycle)
        )
        return {
            "success": success,
            "control": "scripted_cartesian_impedance_with_tray_tracking",
            "reference_period_s": self.period,
            "reference_amplitude_m": TRAVEL,
            "reference_acceleration_peak_m_s2": TRAVEL * self.omega**2,
            "task_finished_s": layers[-1]["placed_s"] if cycle is not None else None,
            "layers": layers,
            "post_release_cycle": cycle,
        }
