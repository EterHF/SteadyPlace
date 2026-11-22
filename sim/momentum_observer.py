"""Proprioceptive, joint-space disturbance observer for the Panda arm."""

import mujoco as mj
import numpy as np


NOMINAL_LOAD_KG = 0.25
OBSERVER_GAIN = 50.0
WRENCH_LAG_S = 0.024
WRENCH_NOISE_STD = np.array([0.15, 0.15, 0.15, 0.012, 0.012, 0.012])
CONTACT_LEVEL_N = 1.0
CONTACT_DEBOUNCE_S = 0.020


def nominal_bias(model, data, result):
    """Compute inertial and gravitational bias from a model and its encoder state."""
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    mj.mj_comVel(model, data)
    mj.mj_rne(model, data, 0, result)


class MomentumObserver:
    """Integrate nominal momentum using only joint encoders and commanded motors."""

    def __init__(self, nominal_model, dt):
        self.model = nominal_model
        self.data = mj.MjData(nominal_model)
        self.dt = dt
        self.tcp_id = nominal_model.site("tcp").id
        self.inertia = np.empty((nominal_model.nv, nominal_model.nv))
        self.rne = np.empty(nominal_model.nv)
        self.jacobian = np.zeros((6, nominal_model.nv))
        self.gravity = abs(float(self.model.opt.gravity[2]))
        self.nominal_gravity = np.r_[NOMINAL_LOAD_KG * self.model.opt.gravity, np.zeros(3)]
        self.wrench_regularizer = np.eye(6) * 0.002**2

    @property
    def tcp(self):
        return self.data.site_xpos[self.tcp_id]

    @property
    def support(self):
        # Positive vertical acceleration and additional loads are indistinguishable
        # from contact in this proxy. The contact detector is armed only in descent.
        return max(0.0, self.wrench[2] + self.mass_kg * self.gravity)

    def _nominal_dynamics(self, q, velocity):
        self.data.qpos[:7] = q
        self.data.qvel[:7] = velocity
        nominal_bias(self.model, self.data, self.rne)
        mj.mj_makeM(self.model, self.data)
        mj.mj_fullM(self.model, self.inertia, self.data.qM)
        return (self.inertia[:7, :7].copy(), self.rne[:7] + self.model.dof_damping[:7] * velocity)

    def reset(self, q, velocity, rng):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[7:9] = 0.018
        self.mass_matrix, self.bias = self._nominal_dynamics(q, velocity)
        self.velocity = velocity.copy()
        self.integral = self.mass_matrix @ self.velocity
        self.residual = np.zeros(7)
        self.wrench = self.nominal_gravity.copy()
        self.mass_kg = NOMINAL_LOAD_KG
        self.contact = False
        self.contact_s = None
        self.contact_duration = 0.0
        self.rng = rng

    def update(self, q, velocity, torque_cmd, time=None):
        matrix, bias = self._nominal_dynamics(q, velocity)
        # p_dot = tau_cmd - bias + M_dot q_dot + tau_ext. Approximate the
        # integral of M_dot q_dot by a matrix increment and midpoint velocity.
        matrix_increment = (matrix - self.mass_matrix) @ (0.5 * (velocity + self.velocity))
        self.integral += self.dt * (torque_cmd - self.bias + self.residual) + matrix_increment
        self.residual = OBSERVER_GAIN * (matrix @ velocity - self.integral)
        self.mass_matrix, self.bias, self.velocity = matrix, bias, velocity.copy()

        mj.mj_jacSite(self.model, self.data, self.jacobian[:3], self.jacobian[3:], self.tcp_id)
        jac = self.jacobian[:, :7]
        # The nominal model already accounts for the rigid load. Add its static
        # weight back to report a TCP-equivalent wrench including payload weight.
        raw = np.linalg.solve(jac @ jac.T + self.wrench_regularizer, jac @ self.residual) + self.nominal_gravity
        noisy = raw + self.rng.normal(scale=WRENCH_NOISE_STD)
        self.wrench += (self.dt / (WRENCH_LAG_S + self.dt)) * (noisy - self.wrench)
        if time is not None:
            self.update_contact(time)

    def update_contact(self, time, candidate=True):
        if not self.contact:
            self.contact_duration = self.contact_duration + self.dt if candidate and self.support > CONTACT_LEVEL_N else 0.0
            if self.contact_duration >= CONTACT_DEBOUNCE_S:
                self.contact = True
                if self.contact_s is None:
                    self.contact_s = time

    def calibrate_load(self, vertical_samples):
        self.mass_kg = float(np.clip(-np.median(vertical_samples) / self.gravity, 0.05, 0.65))
