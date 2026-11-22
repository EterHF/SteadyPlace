"""Observer-based placement with controllable approach and grip."""

from gymnasium import spaces
import numpy as np

from sim.carry_place import CARRY_TIME, DT, SUPPORT_TARGET_N, TARGET, CarryPlaceEnv, Scenario, ramp
from sim.momentum_observer import NOMINAL_LOAD_KG


DEADLINE_S = 7.2
PEAK_LIMIT_N = 10.0
MIN_SPEED, MAX_SPEED = .012, .065
MIN_BRAKE, MAX_BRAKE = .028, .132
REFERENCE_TRAVEL_M = .003
SUPPORT_TCP_TRAVEL_M = .0045
PROFILES = ("quiet", "sustained", "random_pulses")


def disturbance_program(profile, rng):
    if profile == "quiet":
        return dict(pulse=0.0)
    z_force = float(rng.choice([-2.3, 2.3]))
    if profile == "sustained":
        magnitude = float(rng.uniform(6.0, 8.0))
        return dict(pulse=np.sqrt(magnitude**2 - z_force**2), angle_deg=float(rng.uniform(-180, 180)),
                    force_intervals=((1.46, 4.80),), force_z_n=z_force, force_ramp_s=.10)
    if profile != "random_pulses":
        raise ValueError(f"Unknown disturbance profile {profile}")
    starts = rng.uniform([1.09, 1.83, 2.75, 3.65, 4.92], [1.32, 1.92, 3.12, 4.15, 5.35])
    durations = rng.uniform([.13, .24, .15, .15, .15], [.22, .33, .27, .27, .27])
    magnitudes = rng.uniform(6.0, 8.0, len(starts))
    angle = rng.uniform(-np.pi, np.pi, len(starts))
    xy = np.sqrt(magnitudes**2 - z_force**2)[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    return dict(pulse=0.0, force_ramp_s=.022, force_z_n=z_force,
                force_intervals=tuple((round(a, 6), round(a + d, 6)) for a, d in zip(starts, durations)),
                force_vectors_xy_n=tuple(tuple(map(float, force)) for force in xy))


class CoupledCarryPlaceEnv(CarryPlaceEnv):
    """Expose approach speed, brake onset and grip beside two impedance actions."""

    def __init__(self, *, estimated_mass_feedforward=True, target_width_m=.064, **kwargs):
        super().__init__(observer_feedback=True, target_width_m=target_width_m, **kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (42,), dtype=np.float32)
        self.deadline = DEADLINE_S
        self.peak_limit = PEAK_LIMIT_N
        # Allow for unmeasured pose offset within the grasp when judging
        # whether the TCP is centered enough to open over a narrow pad.
        if target_width_m is not None:
            self.open_xy_limit = max(0.0, self.open_xy_limit - .003)
        self.estimated_mass_feedforward = estimated_mass_feedforward
        bodies = [*(f"link{i}" for i in range(1, 8)), "hand", "left_finger", "right_finger"]
        self.arm_bodies = [self.model.body(name).id for name in bodies]
        self.arm_mass = self.model.body_mass[self.arm_bodies].copy()
        self.arm_inertia = self.model.body_inertia[self.arm_bodies].copy()

    def reset(self, *, seed=None, options=None):
        self.near_speed = (MIN_SPEED + MAX_SPEED) / 2
        self.brake_onset = (MIN_BRAKE + MAX_BRAKE) / 2
        self.grip = 28.0
        self.feedforward = 0.0
        self.contact_anchor = None
        self.low_support_s = 0.0
        self.mass_updates = 0
        self.contact_rejections = 0
        return super().reset(seed=seed, options=options)

    def _configure_plant(self):
        super()._configure_plant()
        self.model.body_mass[self.arm_bodies] = self.arm_mass * self.case.arm_scale
        self.model.body_inertia[self.arm_bodies] = self.arm_inertia * self.case.arm_scale
        self.model.body_ipos[self.payload] = self.case.payload_com_m

    def _sample(self):
        rng = self.np_random
        return Scenario(mass=float(rng.uniform(.18, .34)), friction=float(rng.uniform(.46, .79)),
                        height=float(rng.uniform(-.04, .04)), arm_scale=float(rng.uniform(.96, 1.04)),
                        payload_com_m=tuple(map(float, rng.uniform([-.008, -.008, -.005], [.008, .008, .005]))),
                        **disturbance_program(rng.choice(PROFILES), rng))

    def _calibrate_observer(self):
        # Start at the prior as transport begins, with no identification hold.
        self.observer.reset(self.data.qpos[:7], self.data.qvel[:7], self.np_random)

    def _observe(self):
        obs = super()._observe()
        controls = np.array([self.near_speed / MAX_SPEED, self.brake_onset / MAX_BRAKE, self.grip / 60], np.float32)
        return np.concatenate((obs, controls))

    def _vertical_feedforward(self):
        return self.feedforward

    def _grip_command(self):
        return self.grip if self.release_time is None else 255.0

    def _adjust_gains(self, action):
        super()._adjust_gains(action[:2])
        if self.release_time is None:
            alpha = DT / (.05 + DT)
            speed = np.interp(action[2], (-1, 1), (MIN_SPEED, MAX_SPEED))
            brake = np.interp(action[3], (-1, 1), (MIN_BRAKE, MAX_BRAKE))
            grip = np.interp(action[4], (-1, 0, 1), (16, 28, 60))
            self.near_speed += alpha * (speed - self.near_speed)
            self.brake_onset += alpha * (brake - self.brake_onset)
            self.grip += alpha * (grip - self.grip)
            target = self._payload_feedforward_n()
            if self.observer.contact:
                press = np.clip(SUPPORT_TARGET_N + .6 * (SUPPORT_TARGET_N - self.observer.support), 0., 7.)
                target -= press
        else:
            target = 0.0
        self.feedforward += DT / (.075 + DT) * (target - self.feedforward)

    def _payload_feedforward_n(self):
        mass = self.observer.mass_kg if self.estimated_mass_feedforward else NOMINAL_LOAD_KG
        return mass * self.observer.gravity

    def _estimated_landing_reference(self):
        z = self.reference[2]
        if self.observer.contact:
            if self.contact_anchor is None:
                self.contact_anchor = float(self.observer.tcp[2])
                z = self.contact_anchor
            rate = np.clip(.008 * (self.observer.support - SUPPORT_TARGET_N), -.014, .014)
            new_z = np.clip(z + rate * DT, self.contact_anchor - REFERENCE_TRAVEL_M,
                            self.contact_anchor + REFERENCE_TRAVEL_M)
        else:
            gap = z - TARGET[2]
            blend = ramp((gap - self.brake_onset + .04) / .04)
            speed = self.near_speed + (.13 - self.near_speed) * blend
            new_z = max(TARGET[2] - .052, z - speed * DT)
        self.reference[:] = [*TARGET[:2], new_z]
        return np.array([0., 0., (new_z - z) / DT])

    def _update_estimator(self, ref_velocity):
        observer = self.observer
        dq = self.data.qvel[:7]
        observer.update(self.data.qpos[:7], dq, self.data.ctrl[:7])
        if self.release_time is not None:
            return
        vz = float(observer.jacobian[2, :7] @ dq)
        if self.time < CARRY_TIME:
            if abs(vz) < .045 and np.linalg.norm(observer.wrench[:2]) < 3.25:
                reading = np.clip(-observer.wrench[2] / observer.gravity, .05, .65)
                observer.mass_kg += DT / (.35 + DT) * (float(reading) - observer.mass_kg)
                self.mass_updates += 1
            return
        if observer.contact:
            self.low_support_s = self.low_support_s + DT if observer.support < .45 else 0.0
            if self.low_support_s >= .10 or not self._tcp_holds_at_contact():
                observer.contact = False
                observer.contact_duration = 0.0
                self.contact_anchor = None
                self.low_support_s = 0.0
                self.reference[2] = observer.tcp[2]
                self.contact_rejections += 1
        else:
            in_band = observer.tcp[2] <= TARGET[2] + .055
            arrested = vz >= -max(.006, .65 * abs(ref_velocity[2]))
            observer.update_contact(self.time, in_band and arrested)

    def _tcp_holds_at_contact(self):
        return self.contact_anchor is not None and abs(self.contact_anchor - self.observer.tcp[2]) <= SUPPORT_TCP_TRAVEL_M

    def _ready_to_open(self, touch, xy_error):
        press = self._payload_feedforward_n() - self.feedforward
        stationary = abs(float(self.observer.jacobian[2, :7] @ self.data.qvel[:7])) <= .005
        return (super()._ready_to_open(touch, xy_error) and self._tcp_holds_at_contact() and press >= 2.5
                and stationary)

    def _score_settled(self):
        if self.first_touch is None or self.release_time < self.first_touch:
            return "premature_release"
        return super()._score_settled()

    def _force_time(self):
        return -np.inf if self.release_time is not None else super()._force_time()

    def _record_estimate(self, disturbance):
        super()._record_estimate(disturbance)
        self.last_observer_trace[-1] += (self.near_speed, self.brake_onset, self.grip, self.feedforward,
                                         float(self.observer.tcp[2]),
                                         self.contact_anchor if self.contact_anchor is not None else np.nan)
