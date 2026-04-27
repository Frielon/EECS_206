import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    KF_PROCESS_NOISE,
    KF_MEASUREMENT_NOISE,
    GRAVITY_MM_S2,
    ROLLING_FACTOR,
)


def tilt_to_accel(theta_x, theta_y, g=GRAVITY_MM_S2, alpha=ROLLING_FACTOR):
    """
    Convert plate tilt angles (radians) to in-plane acceleration of the ball
    center, expressed in the table frame (mm/s^2).

    Sign convention (matches config.MARKER_POSITIONS_TABLE: +x right, +y up):
        theta_y > 0  =>  +x edge dips down  =>  a_x positive
        theta_x > 0  =>  +y edge dips down  =>  a_y negative
    Flip a sign here if your physical setup defines rotation differently.
    """
    a_x =  alpha * g * math.sin(theta_y)
    a_y = -alpha * g * math.sin(theta_x)
    return a_x, a_y


class BallKalmanFilter:
    """
    Linear Kalman filter for 2D ball tracking with plate tilt as a control input.

    State:        [x, y, vx, vy]      (mm, mm/s)
    Measurement:  [x, y]              (mm)
    Control:      [a_x, a_y]          (mm/s^2, table frame)

    Without any call to set_control(), u = 0 and the filter degrades cleanly to
    the original constant-velocity model.
    """

    def __init__(self, dt=1.0 / 60.0):
        self.initialized = False
        self.x = np.zeros(4)
        self.u = np.zeros(2)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.R = np.eye(2) * (KF_MEASUREMENT_NOISE ** 2)
        self.P = np.eye(4) * 100.0

        self._build_dt_matrices(dt)

    def _build_dt_matrices(self, dt):
        """Rebuild F, B, and Q whenever dt changes."""
        self.dt = dt

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        self.B = np.array([
            [0.5 * dt * dt, 0.0          ],
            [0.0,           0.5 * dt * dt],
            [dt,            0.0          ],
            [0.0,           dt           ],
        ], dtype=np.float64)

        # Continuous-white-noise-acceleration Q. KF_PROCESS_NOISE is sigma_a
        # in mm/s^2 (the *unmodeled* acceleration, since gravity is in B*u).
        q2 = KF_PROCESS_NOISE ** 2
        self.Q = np.array([
            [dt**4 / 4, 0,         dt**3 / 2, 0        ],
            [0,         dt**4 / 4, 0,         dt**3 / 2],
            [dt**3 / 2, 0,         dt**2,     0        ],
            [0,         dt**3 / 2, 0,         dt**2    ],
        ], dtype=np.float64) * q2

    def reset(self, x, y):
        """Initialize / reset the filter at a known position."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.u = np.zeros(2)
        self.initialized = True

    def set_control(self, a_x, a_y):
        """Set the modeled in-plane acceleration (mm/s^2, table frame)."""
        self.u = np.array([a_x, a_y], dtype=np.float64)

    def predict(self, dt=None):
        """Predict step. If dt is given, F/B/Q are rebuilt for the new dt."""
        if dt is not None and dt != self.dt:
            self._build_dt_matrices(dt)
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_x, z_y):
        """Update step: incorporate a new measurement."""
        z = np.array([z_x, z_y])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return tuple(self.x)

    def get_position(self):
        return (self.x[0], self.x[1])

    def get_velocity(self):
        return (self.x[2], self.x[3])
