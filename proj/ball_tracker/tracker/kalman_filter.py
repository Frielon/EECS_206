import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import KF_PROCESS_NOISE, KF_MEASUREMENT_NOISE


class BallKalmanFilter:
    """
    Linear Kalman filter for 2D ball tracking.
    State:       [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, dt=1.0/60.0):
        self.dt = dt
        self.initialized = False

        # State vector [x, y, vx, vy]
        self.x = np.zeros(4)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Measurement matrix: we observe [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # Process noise covariance
        q = KF_PROCESS_NOISE
        self.Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ]) * q**2

        # Measurement noise covariance
        r = KF_MEASUREMENT_NOISE
        self.R = np.eye(2) * r**2

        # Error covariance
        self.P = np.eye(4) * 100.0   # large initial uncertainty

    def reset(self, x, y):
        """Initialize / reset the filter at a known position."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 100.0
        self.initialized = True

    def predict(self):
        """Predict step: propagate state forward by dt."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_x, z_y):
        """
        Update step: incorporate a new measurement.
        Call predict() first, then update().
        """
        z = np.array([z_x, z_y])
        y = z - self.H @ self.x                     # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """Returns (x, y, vx, vy)."""
        return tuple(self.x)

    def get_position(self):
        """Returns (x, y) in mm."""
        return (self.x[0], self.x[1])

    def get_velocity(self):
        """Returns (vx, vy) in mm/s."""
        return (self.x[2], self.x[3])
