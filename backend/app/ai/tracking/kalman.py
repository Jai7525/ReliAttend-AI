"""
ReliAttend AI — 2D Bounding Box Kalman Filter (Phase 9 - Substage 9.1)
Pure NumPy implementation of constant-velocity motion model for face bounding boxes.
Tracks state vector: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]^T
Zero external C++ / SciPy dependencies required.
"""

from typing import Tuple
import numpy as np


class KalmanBoxTracker:
    """
    2D Bounding Box Kalman Filter for tracking human faces.
    State representation:
        x = [x_center, y_center, s, h, vx, vy, vs, vh]^T
    where:
        - (x_center, y_center) is the center of the bounding box.
        - s is the aspect ratio (width / height).
        - h is the box height.
        - (vx, vy, vs, vh) are their respective velocities over discrete frame steps.
    Measurement representation:
        z = [x_center, y_center, s, h]^T
    """

    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int]):
        """
        Initialize Kalman filter from an initial bounding box (x, y, w, h).
        """
        # Dimensions
        self.dim_x = 8  # 4 state variables + 4 velocities
        self.dim_z = 4  # 4 observed variables

        # State transition matrix F (constant-velocity model: x_t = x_{t-1} + v * dt, dt=1)
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement function matrix H (observes position, scale, and height directly)
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Measurement uncertainty covariance R
        # Face bounding boxes have some pixel noise (typically ~1-10 pixels)
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0, 0] = 1.0    # x_center
        self.R[1, 1] = 1.0    # y_center
        self.R[2, 2] = 10.0   # aspect ratio (more jittery)
        self.R[3, 3] = 1.0    # height

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01  # Lower noise for velocity
        self.Q[:4, :4] *= 0.001

        # State estimation covariance P
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0  # High initial uncertainty on velocity

        # Initial state vector x
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.x[:4] = self._bbox_to_z(bbox)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    @staticmethod
    def _bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Converts (x, y, w, h) bounding box into measurement vector [x_center, y_center, s, h]^T.
        """
        x, y, w, h = bbox
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        x_center = float(x) + w / 2.0
        y_center = float(y) + h / 2.0
        s = w / h  # aspect ratio
        return np.array([[x_center], [y_center], [s], [h]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Converts measurement vector [x_center, y_center, s, h]^T back to (x, y, w, h) integer box.
        """
        x_center = float(z[0, 0])
        y_center = float(z[1, 0])
        s = max(0.01, float(z[2, 0]))
        h = max(1.0, float(z[3, 0]))
        w = s * h
        x = int(round(x_center - w / 2.0))
        y = int(round(y_center - h / 2.0))
        return (x, y, int(round(w)), int(round(h)))

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Advances state vector by one time step and returns predicted bounding box (x, y, w, h).
        """
        # Guard: prevent negative height or inverted aspect ratio
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        # State extrapolation: x = F * x
        self.x = np.dot(self.F, self.x)

        # Covariance extrapolation: P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        predicted_box = self._z_to_bbox(self.x[:4])
        self.history.append(predicted_box)
        return predicted_box

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """
        Updates the state estimate with a newly observed detection bounding box.
        Standard Kalman update equations:
            y = z - H * x        (innovation)
            S = H * P * H^T + R  (innovation covariance)
            K = P * H^T * S^-1   (optimal Kalman gain)
            x = x + K * y        (updated state)
            P = (I - K * H) * P  (updated covariance)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)

        # Innovation: y = z - H * x
        y = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P * H^T * S^-1
        # Using np.linalg.solve for numerical stability over explicit matrix inversion
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        # Update state: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Update covariance: P = (I - K * H) * P
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self) -> Tuple[int, int, int, int]:
        """
        Returns the current bounding box estimate (x, y, w, h).
        """
        return self._z_to_bbox(self.x[:4])

