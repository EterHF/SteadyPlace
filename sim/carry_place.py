"""Gymnasium task for disturbed transport and supported placement."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from sim.momentum_observer import MomentumObserver, NOMINAL_LOAD_KG, nominal_bias


ROOT = Path(__file__).resolve().parents[1]
DT = 0.002
SUBSTEPS = 10
CARRY_TIME = 2.0
DEADLINE = 6.8
OBSERVER_DEADLINE = 8.0
SUPPORT_TARGET_N = 4.0
PLACEMENT_PEAK_LIMIT_N = 20.0
TABLE_Z = 0.12
HALF = 0.025
SOURCE = np.array([0.52, -0.21, TABLE_Z + HALF])
TARGET = np.array([0.52, 0.21, TABLE_Z + HALF])
LIFT = 0.14
HOME = np.array([0, -0.35, 0, -2.15, 0, 1.80, np.pi / 4])
DOWN = np.diag([1.0, -1.0, -1.0])
TORQUE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12])


@dataclass(frozen=True)
class Scenario:
    mass: float = 0.25
    friction: float = 0.65
    height: float = 0.0
    pulse: float = 8.0
    angle_deg: float = 25.0
    onset: float = 0.88
    duration: float = 0.08
    # Explicit (start, end) pairs in seconds override the single-pulse timing.
    force_intervals: tuple[tuple[float, float], ...] = ()
    # Start the force clock at this actual clearance during descent, up to opening.
    approach_force_gap_m: float | None = None
    # Replace the common force vector for each non-overlapping interval.
    force_vectors_xy_n: tuple[tuple[float, float], ...] = ()
    # A smooth increase from zero at the start of each interval.
    force_ramp_s: float = 0.0
    force_z_n: float = 0.0
    arm_scale: float = 1.0
    payload_com_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


def build_model(target_width_m=None, *, nominal_payload=False, robot_only=False):
    panda_dir = ROOT / "assets/franka_emika_panda"
    robot = ET.parse(panda_dir / "panda.xml").getroot()
    scene = ET.parse(Path(__file__).with_name("scene.xml")).getroot()
    robot.set("model", scene.get("model"))
    robot.find("compiler").set("meshdir", str(panda_dir / "assets"))
    robot.find("option").attrib.update(scene.find("option").attrib)
    robot.remove(robot.find("keyframe"))
    robot.append(scene.find("visual"))
    if target_width_m is not None:
        if not np.isfinite(target_width_m) or target_width_m <= 0:
            raise ValueError("Target width must be a positive length in meters")
        target = scene.find(".//body[@name='target_table_body']")
        geom = target.find("geom[@name='target_table']")
        half_width = target_width_m / 2.0
        half_height = geom.get("size").split()[2]
        geom.set("size", f"{half_width} {half_width} {half_height}")
        zone = target.find("site[@name='target_zone']")
        zone.set("type", "box")
        zone.set("size", f"{half_width} {half_width} 0.0004")
    if not robot_only:
        for section in ("asset", "worldbody", "contact"):
            robot.find(section).extend(scene.find(section))
    hand = robot.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tcp", pos="0 0 0.1029", size="0.004", rgba="0 0 0 0")
    if nominal_payload:
        load = ET.SubElement(hand, "body", name="nominal_load", pos="0 0 0.1029")
        inertia = NOMINAL_LOAD_KG * (2 * HALF)**2 / 6
        ET.SubElement(load, "inertial", pos="0 0 0", mass=str(NOMINAL_LOAD_KG),
                      diaginertia=f"{inertia} {inertia} {inertia}")
    actuators = robot.find("actuator")
    gripper = actuators[-1]
    actuators.clear()
    for i, limit in enumerate(TORQUE_LIMIT, start=1):
        ET.SubElement(actuators, "motor", name=f"torque{i}", joint=f"joint{i}", ctrlrange=f"{-limit} {limit}")
    gripper.set("gainprm", str(0.04 * 2000 / 255))
    gripper.set("biasprm", "0 -2000 -30")
    actuators.append(gripper)
    return mj.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))


def ramp(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def min_jerk(progress):
    u = np.clip(progress, 0.0, 1.0)
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


class CarryPlaceEnv(gym.Env):
    """Two actions scale horizontal and vertical stiffness of the shared controller."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, baseline="phase", fixed_gains=(360.0, 300.0), force_feedback=True,
                 randomize=False, render_mode=None, target_width_m=None, observer_feedback=False):
        if baseline not in {"fixed", "phase"}:
            raise ValueError(f"Unknown baseline {baseline!r}")
        self.model = build_model(target_width_m)
        self.data = mj.MjData(self.model)
        self.baseline = baseline
        self.observer = None
        if observer_feedback:
            self.observer = MomentumObserver(build_model(nominal_payload=True, robot_only=True), DT)
            self.control_model = build_model(robot_only=True)
            self.control_state = mj.MjData(self.control_model)
            self.control_bias = np.empty(self.control_model.nv)
        self.deadline = OBSERVER_DEADLINE if observer_feedback else DEADLINE
        self.peak_limit = PLACEMENT_PEAK_LIMIT_N
        self.open_xy_limit = 0.038 if target_width_m is None else target_width_m / 2 - HALF
        self.fixed_gains = np.asarray(fixed_gains, dtype=float)
        self.force_feedback = force_feedback
        self.randomize = randomize
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (39,), dtype=np.float32)
        self.payload = self.model.body("payload").id
        self.payload_geom = self.model.geom("payload_box").id
        self.target_geom = self.model.geom("target_table").id
        self.source_geom = self.model.geom("source_table").id
        self.floor_geom = self.model.geom("floor").id
        self.plate = self.model.body("target_table_body").id
        self.landing_pair = self.model.pair("landing_contact").id
        self.tcp = self.model.site("tcp").id
        self.fingers = {self.model.body(name).id for name in ("left_finger", "right_finger")}
        self.nominal_inertia = self.model.body_inertia[self.payload].copy()
        self.nominal_plate_z = float(self.model.body_pos[self.plate, 2])
        self.geom_body = self.model.geom_bodyid
        self.raw_wrench = np.zeros(6)
        self.j_pos = np.zeros((3, self.model.nv))
        self.j_rot = np.zeros_like(self.j_pos)
        self.renderer = None
        self.done = True

    @property
    def block(self):
        return self.data.xpos[self.payload]

    @property
    def tcp_pos(self):
        return self.data.site_xpos[self.tcp]

    def _orientation_error(self):
        rotation = self.data.site_xmat[self.tcp].reshape(3, 3)
        return 0.5 * np.cross(rotation.T, DOWN.T).sum(axis=0)

    def _jacobians(self):
        mj.mj_jacSite(self.model, self.data, self.j_pos, self.j_rot, self.tcp)
        return self.j_pos[:, :7], self.j_rot[:, :7]

    def _place_open_hand(self):
        # Only the reset setup uses IK. During the task the arm is torque controlled.
        self.data.qpos[:7] = HOME
        self.data.qpos[7:9] = 0.04
        for _ in range(120):
            mj.mj_forward(self.model, self.data)
            error = np.r_[SOURCE - self.tcp_pos, self._orientation_error()]
            if np.linalg.norm(error) < 1e-5:
                return
            jac = np.vstack(self._jacobians())
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002 * np.eye(6), error)
            self.data.qpos[:7] += np.clip(delta, -0.07, 0.07)
            limits = self.model.jnt_range[:7]
            np.clip(self.data.qpos[:7], limits[:, 0] + 0.025, limits[:, 1] - 0.025, out=self.data.qpos[:7])
        raise RuntimeError(f"Reset IK did not converge (error {error})")

    def _apply_control(self, reference, ref_velocity, gains, grip):
        jac, rot = self._jacobians()
        qvel = self.data.qvel[:7]
        stiffness = np.array([gains[0], gains[0], gains[1]])
        # Critical-damping heuristic using a fixed 2 kg equivalent mass.
        damping = 2.0 * np.sqrt(2.0 * stiffness)
        force = stiffness * (reference - self.tcp_pos) + damping * (ref_velocity - jac @ qvel)
        force[2] += self._vertical_feedforward()
        moment = 42.0 * self._orientation_error() - 5.0 * (rot @ qvel)
        if self.observer is None:
            bias = self.data.qfrc_bias[:7]
        else:
            self.control_state.qpos[:] = self.data.qpos[:self.control_model.nq]
            self.control_state.qvel[:] = self.data.qvel[:self.control_model.nv]
            nominal_bias(self.control_model, self.control_state, self.control_bias)
            bias = self.control_bias[:7]
        torque = jac.T @ force + rot.T @ moment + bias - 0.8 * qvel
        self.data.ctrl[:7] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.data.ctrl[7] = grip

    def _vertical_feedforward(self):
        return 0.0

    def _grip_command(self):
        return 28.0 if self.release_time is None else 255.0

    def _contact_state(self):
        force_z = 0.0
        touch = False
        held_by = set()
        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.efc_address < 0:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            if self.payload_geom in (a, b):
                other = b if a == self.payload_geom else a
                if other == self.target_geom:
                    touch = True
                    mj.mj_contactForce(self.model, self.data, i, self.raw_wrench)
                    # The frame rows are contact axes in world coordinates. The force acts on geom2.
                    z = self.raw_wrench[:3] @ contact.frame.reshape(3, 3)[:, 2]
                    force_z += z if b == self.payload_geom else -z
                if self.geom_body[other] in self.fingers:
                    held_by.add(int(self.geom_body[other]))
            elif a in (self.target_geom, self.source_geom) or b in (self.target_geom, self.source_geom):
                # Arm or gripper colliding with a table is a task failure.
                collision |= bool(self.geom_body[a] not in (0, self.plate) or
                                  self.geom_body[b] not in (0, self.plate))
        return max(0.0, force_z), touch, held_by, collision

    def _sample(self):
        rng = self.np_random
        if self.observer is not None:
            start = rng.uniform(0.50, 1.07)
            return Scenario(mass=rng.uniform(0.18, 0.34), friction=rng.uniform(0.45, 0.8),
                            height=rng.uniform(-0.04, 0.04), pulse=rng.uniform(0.0, 4.0),
                            angle_deg=rng.uniform(-180.0, 180.0), force_z_n=rng.uniform(-3.0, 3.0),
                            force_intervals=((start, start + rng.uniform(0.10, 0.65)),), force_ramp_s=0.07)
        return Scenario(mass=rng.uniform(0.18, 0.34), friction=rng.uniform(0.45, 0.8),
                        height=rng.uniform(-0.012, 0.012), pulse=rng.uniform(0.0, 10.0),
                        angle_deg=rng.uniform(-180.0, 180.0), onset=rng.uniform(0.57, 1.26),
                        duration=rng.uniform(0.06, 0.10))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.case = options.get("scenario") if options else None
        if self.case is None:
            self.case = self._sample() if self.randomize else Scenario()
        self.force_intervals = self.case.force_intervals or ((self.case.onset, self.case.onset + self.case.duration),)
        self.force_origin = 0.0 if self.case.approach_force_gap_m is None else None
        self.force_window_start = min(start for start, _ in self.force_intervals)
        self.force_window_end = max(end for _, end in self.force_intervals) + 0.35
        angle = np.deg2rad(self.case.angle_deg)
        self.force_xy = self.case.pulse * np.array([np.cos(angle), np.sin(angle)])
        self.force_vector = np.r_[self.force_xy, self.case.force_z_n]
        self.force_vectors = np.asarray(self.case.force_vectors_xy_n, dtype=float)
        if len(self.case.force_vectors_xy_n):
            if self.force_vectors.shape != (len(self.force_intervals), 2) or not np.all(np.isfinite(self.force_vectors)):
                raise ValueError("Provide one finite horizontal force vector per interval")
            self.force_vectors = np.column_stack((self.force_vectors, np.full(len(self.force_intervals), self.case.force_z_n)))
        if not np.isfinite(self.case.force_ramp_s) or self.case.force_ramp_s < 0:
            raise ValueError("Force rise time must be finite and nonnegative")
        self._configure_plant()
        mj.mj_setConst(self.model, self.data)
        mj.mj_resetData(self.model, self.data)
        self._place_open_hand()
        for i in range(700):
            fraction = min_jerk((i - 200) / 420.0)
            reference = SOURCE + [0, 0, fraction * LIFT]
            self._apply_control(reference, np.zeros(3), (470.0, 520.0), 28.0)
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        if len(self._contact_state()[2]) != 2 or self.block[2] < SOURCE[2] + LIFT - 0.035:
            raise RuntimeError("Scripted reset failed to lift the payload with bilateral finger contact")

        if self.observer is not None:
            self._calibrate_observer()

        self.start_time = self.data.time
        self.time = 0.0
        self.reference = SOURCE + [0, 0, LIFT]
        self.gains = self.fixed_gains.copy() if self.baseline == "fixed" else np.array([440.0, 360.0])
        self.support = 0.0
        self.support_time = 0.0
        self.stable_time = 0.0
        self.first_touch = None
        self.touch_vz = None
        self.release_time = None
        self.release_z = None
        self.peak_force = 0.0
        self.placement_peak = 0.0
        self.impulse = 0.0
        self.max_carry_error = 0.0
        self.carry_error_sum = 0.0
        self.carry_samples = 0
        self.pulse_window_error = 0.0
        self.grasp_loss_time = 0.0
        self.previous_action = np.zeros(self.action_space.shape)
        self.done = False
        self.last_trace = []
        self.last_observer_trace = []
        return self._observe(), self._info("running")

    def _configure_plant(self):
        self.model.body_mass[self.payload] = self.case.mass
        self.model.body_inertia[self.payload] = self.nominal_inertia * (self.case.mass / 0.25)
        self.model.body_pos[self.plate, 2] = self.nominal_plate_z + self.case.height
        self.model.pair_friction[self.landing_pair, :2] = self.case.friction

    def _calibrate_observer(self):
        self.observer.reset(self.data.qpos[:7], self.data.qvel[:7], self.np_random)
        rest = SOURCE + [0, 0, LIFT]
        samples = []
        for i in range(150):
            self._apply_control(rest, np.zeros(3), (470.0, 520.0), 28.0)
            mj.mj_step(self.model, self.data)
            self.observer.update(self.data.qpos[:7], self.data.qvel[:7], self.data.ctrl[:7])
            if i >= 75:
                samples.append(self.observer.wrench[2])
        self.observer.calibrate_load(samples)
        mj.mj_forward(self.model, self.data)

    def _observe(self):
        jac, rot = self._jacobians()
        phase = [self.time < CARRY_TIME, self.time >= CARRY_TIME and self.release_time is None,
                 self.release_time is not None]
        if self.observer is None:
            object_features = np.r_[(self.block - self.tcp_pos) / 0.05, self.data.qvel[-6:-3] / 0.4,
                                    (TARGET[:2] - self.block[:2]) / 0.25, (self.block[2] - TARGET[2]) / 0.15]
            support = self.support
        else:
            wrench = self.observer.wrench if self.force_feedback else np.zeros(6)
            object_features = np.r_[wrench[:3] / 12.0, wrench[3:] / 3.0,
                                    (self.observer.mass_kg - NOMINAL_LOAD_KG) / 0.12,
                                    (self.observer.tcp[2] - TARGET[2]) / 0.15,
                                    float(self.observer.contact and self.force_feedback)]
            support = self.observer.support
        obs = np.r_[self.data.qpos[:7] / 3.0, self.data.qvel[:7] / 5.0,
                    (self.reference - self.tcp_pos) / 0.10,
                    jac @ self.data.qvel[:7] / 0.4, rot @ self.data.qvel[:7] / 3.0,
                    object_features, phase, self.time / self.deadline, (support / 8.0 if self.force_feedback else 0.0),
                    self.gains / 500.0]
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _motion_reference(self):
        if self.time < CARRY_TIME:
            u = self.time / CARRY_TIME
            self.reference = SOURCE + min_jerk(u) * (TARGET - SOURCE) + [0, 0, LIFT]
            return (TARGET - SOURCE) * (30.0 * u**2 * (1.0 - u)**2 / CARRY_TIME)
        if self.release_time is not None:
            u = (self.time - self.release_time - 0.24) / 0.55
            self.reference[2] = self.release_z + 0.10 * min_jerk(u)
            u = np.clip(u, 0.0, 1.0)
            return np.array([0.0, 0.0, 0.10 * 30.0 * u**2 * (1.0 - u)**2 / 0.55])
        if self.observer is not None:
            return self._estimated_landing_reference()
        # The same nominal approach profile and lower limit are used by every method.
        speed = 0.035 + 0.09 * ramp((self.reference[2] - TARGET[2] - 0.029) / 0.046)
        new_z = max(TARGET[2] - 0.043, self.reference[2] - speed * DT)
        velocity = (new_z - self.reference[2]) / DT
        self.reference[:] = [*TARGET[:2], new_z]
        return np.array([0.0, 0.0, velocity])

    def _estimated_landing_reference(self):
        z = self.reference[2]
        if self.observer.contact or z <= TARGET[2]:
            rate = np.clip(0.012 * (self.observer.support - SUPPORT_TARGET_N), -0.032, 0.028)
            travel = 0.060 if self.observer.contact else 0.047
            new_z = max(TARGET[2] - travel, z + rate * DT)
        else:
            # Slow down above the full range of possible table heights. Below
            # the nominal endpoint, only the bounded force search can descend.
            speed = 0.032 + 0.088 * ramp((z - TARGET[2] - 0.066) / 0.04)
            new_z = max(TARGET[2], z - speed * DT)
        self.reference[:] = [*TARGET[:2], new_z]
        return np.array([0.0, 0.0, (new_z - z) / DT])

    def _adjust_gains(self, action):
        if self.baseline == "fixed":
            return
        clearance = (self.block[2] if self.observer is None else self.observer.tcp[2]) - TARGET[2]
        approach = float(self.time >= CARRY_TIME) * ramp((0.10 - clearance) / 0.086)
        if self.force_feedback:
            support = self.support if self.observer is None else (self.observer.support if self.observer.contact else 0.0)
            approach = max(approach, ramp(support / 2.0))
        vertical_range = 290.0 if self.observer is None else 220.0
        nominal = np.array([440.0 - 120.0 * approach, 360.0 - vertical_range * approach])
        desired = np.clip(nominal * np.exp(0.55 * action), [140.0, 30.0], [730.0, 620.0])
        if self.release_time is not None:
            desired = self.gains
        # Smooth-rate update runs at the physics rate, including the phase transition.
        self.gains += np.clip((desired - self.gains) * DT / 0.065, -1400.0 * DT, 1400.0 * DT)

    def _info(self, status):
        info = dict(status=status, is_success=status == "settled", first_touch_s=self.first_touch,
                    release_s=self.release_time, contact_speed_z_m_s=self.touch_vz,
                    peak_force_n=self.peak_force if self.first_touch is not None else None,
                    impulse_100ms_n_s=self.impulse if self.first_touch is not None else None,
                    contact_window_complete=self.first_touch is not None and self.time - self.first_touch >= 0.10 - DT / 2,
                    max_carry_error_m=self.max_carry_error,
                    rms_carry_error_m=np.sqrt(self.carry_error_sum / max(1, self.carry_samples)),
                    pulse_window_error_m=self.pulse_window_error,
                    final_xy_error_m=float(np.linalg.norm(self.block[:2] - TARGET[:2])),
                    final_tilt_deg=float(np.degrees(np.arccos(np.clip(self.data.xmat[self.payload, 8], -1, 1)))),
                    elapsed_s=self.time)
        if self.observer is not None:
            info.update(estimated_mass_kg=self.observer.mass_kg, estimated_touch_s=self.observer.contact_s,
                        placement_peak_force_n=self.placement_peak, footprint_margin_mm=self._footprint_margin_mm())
        return info

    def _footprint_margin_mm(self):
        rotation = self.data.geom_xmat[self.payload_geom].reshape(3, 3)
        extent = np.abs(rotation[:2]) @ self.model.geom_size[self.payload_geom]
        offset = self.data.geom_xpos[self.payload_geom, :2] - self.data.geom_xpos[self.target_geom, :2]
        return float(1000 * np.min(self.model.geom_size[self.target_geom, :2] - np.abs(offset) - extent))

    def _ready_to_open(self, touch, xy_error):
        if self.observer is None:
            return touch and self.support >= 0.65 and xy_error <= 0.038
        est_error = np.max(np.abs(self.observer.tcp[:2] - TARGET[:2]))
        return self.observer.contact and 3.0 <= self.observer.support <= 5.0 and est_error <= self.open_xy_limit

    def _score_settled(self):
        if self.placement_peak > self.peak_limit:
            return "impact_limit"
        return "overhang" if self._footprint_margin_mm() < 0.0 else "settled"

    def _force_time(self):
        if self.case.approach_force_gap_m is None:
            return self.time
        if self.release_time is not None:
            return -np.inf
        if self.force_origin is None and self.time >= CARRY_TIME:
            vertical = self.data.geom_xmat[self.payload_geom].reshape(3, 3)[2]
            bottom = self.data.geom_xpos[self.payload_geom, 2] - np.abs(vertical) @ self.model.geom_size[self.payload_geom]
            top = self.data.geom_xpos[self.target_geom, 2] + self.model.geom_size[self.target_geom, 2]
            if bottom - top <= self.case.approach_force_gap_m:
                self.force_origin = self.time
        return -np.inf if self.force_origin is None else round(self.time - self.force_origin, 9)

    def _disturbance(self, time):
        rise_s = self.case.force_ramp_s
        if not self.force_vectors.size and rise_s == 0:
            active = any(start <= time < end for start, end in self.force_intervals)
            return self.force_vector * active
        for i, (start, end) in enumerate(self.force_intervals):
            if start <= time < end:
                force = self.force_vectors[i] if self.force_vectors.size else self.force_vector
                return force if rise_s == 0 else force * ramp((time - start) / rise_s)
        return np.zeros(3)

    def _update_estimator(self, ref_velocity):
        detection_time = self.time if self.time >= CARRY_TIME and self.release_time is None else None
        self.observer.update(self.data.qpos[:7], self.data.qvel[:7], self.data.ctrl[:7], detection_time)

    def _record_estimate(self, disturbance):
        self.last_observer_trace.append((self.time, *self.observer.wrench, self.observer.support,
                                         self.observer.mass_kg, self.observer.contact, disturbance[2]))

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected a finite action of shape {self.action_space.shape}")
        action = np.clip(action, -1.0, 1.0)
        previous_peak, previous_impulse = self.peak_force, self.impulse
        status = "running"
        self.last_trace = []
        self.last_observer_trace = []
        for _ in range(SUBSTEPS):
            velocity = self._motion_reference()
            self._adjust_gains(action)
            force_time = self._force_time()
            disturbance = self._disturbance(force_time)
            self.data.xfrc_applied[self.payload, :3] = disturbance
            self._apply_control(self.reference, velocity, self.gains, self._grip_command())
            before_vz = self.data.qvel[-4]
            mj.mj_step(self.model, self.data)
            self.time = round(self.data.time - self.start_time, 9)
            if self.observer is not None:
                self._update_estimator(velocity)
            force, touch, held_by, collision = self._contact_state()
            self.support += (force - self.support) * (DT / 0.024)
            self.placement_peak = max(self.placement_peak, force)
            if touch and force > 0.0 and self.first_touch is None:
                self.first_touch = self.time
                self.touch_vz = float(before_vz)
            if self.first_touch is not None and round(self.time - self.first_touch, 9) < 0.10:
                self.peak_force = max(self.peak_force, force)
                self.impulse += force * DT

            xy_error = np.linalg.norm(self.block[:2] - TARGET[:2])
            tracking = np.linalg.norm(self.block[:2] - self.reference[:2])
            if self.time <= CARRY_TIME:
                self.max_carry_error = max(self.max_carry_error, tracking)
                self.carry_error_sum += tracking**2
                self.carry_samples += 1
                self.grasp_loss_time = self.grasp_loss_time + DT if len(held_by) < 2 else 0.0
                if (self.case.approach_force_gap_m is None
                        and self.force_window_start <= self.time <= self.force_window_end):
                    self.pulse_window_error = max(self.pulse_window_error, tracking)
            if self.release_time is None and self.time >= CARRY_TIME:
                supported = self._ready_to_open(touch, xy_error)
                self.support_time = self.support_time + DT if supported else 0.0
                if self.support_time >= 0.08:
                    self.release_time, self.release_z = self.time, self.reference[2]
            if self.release_time is not None:
                pose_ok = xy_error <= 0.025 and self.data.xmat[self.payload, 8] >= np.cos(np.deg2rad(7))
                speed_ok = np.linalg.norm(self.data.qvel[-6:-3]) <= 0.015 and np.linalg.norm(self.data.qvel[-3:]) <= 0.15
                fingers_clear = not held_by and self.data.qpos[7:9].min() > 0.034
                stable = touch and force > 0.0 and pose_ok and speed_ok and fingers_clear
                self.stable_time = self.stable_time + DT if stable else 0.0
            self.last_trace.append((self.time, *self.block.copy(), *self.reference.copy(), force, self.support,
                                    *disturbance[:2], *self.gains, self.touch_vz is not None, self.release_time is not None))
            if self.observer is not None:
                self._record_estimate(disturbance)
            if collision:
                status = "table_collision"
            elif self.grasp_loss_time >= 0.05 or self.block[2] < TABLE_Z - 0.047 or (
                self.release_time is None and np.linalg.norm(self.block - self.tcp_pos) > 0.072
            ):
                status = "dropped"
            elif self.stable_time >= 0.30 and self.time - self.release_time >= 0.79:
                status = "settled"
            if status != "running":
                break

        mj.mj_forward(self.model, self.data)
        if self.observer is not None and status == "settled":
            status = self._score_settled()
        terminated = status != "running"
        truncated = not terminated and self.time >= self.deadline
        if truncated:
            status = "timeout"
        self.done = terminated or truncated
        reward = (-0.015 - 0.075 * min(tracking / 0.05, 2.0)
                  - 0.02 * np.linalg.norm(action - self.previous_action)
                  - 0.30 * (self.peak_force - previous_peak) / 10.0
                  - 0.30 * (self.impulse - previous_impulse) / 0.50)
        if self.done:
            reward += 5.0 if status == "settled" else -5.0
        self.previous_action = action
        return self._observe(), float(reward), terminated, truncated, self._info(status)

    def render(self):
        if self.renderer is None:
            self.renderer = mj.Renderer(self.model, height=640, width=1024)
        self.renderer.update_scene(self.data, camera="overview")
        disturbance = self.data.xfrc_applied[self.payload, :3]
        if np.linalg.norm(disturbance) > 0.0:
            scene = self.renderer.scene
            if scene.ngeom < scene.maxgeom:
                arrow = scene.geoms[scene.ngeom]
                direction = disturbance / np.linalg.norm(disturbance)
                mj.mjv_initGeom(arrow, mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.eye(3).ravel(), np.array([1.0, 0.28, 0.16, 1.0], dtype=np.float32))
                mj.mjv_connector(arrow, mj.mjtGeom.mjGEOM_ARROW, 0.006,
                                 self.block - 0.16 * direction, self.block - 0.037 * direction)
                scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
