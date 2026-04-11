# Ball-on-Plate Vision System: Coding Guidance

A complete guide to measuring ball position and velocity on a rotatable table using a single camera, ArUco markers, and OpenCV.

---

## System Overview

```
Camera (overhead)
     │
     ▼
┌──────────────┐
│ ● ball       │   ← Table with ArUco markers at corners
│ [M0]    [M1] │
│              │
│ [M2]    [M3] │
└──────────────┘
     ▲
  Tilt motors (controlled by robot)
```

**Pipeline per frame:**
1. Capture image → 2. Detect ArUco markers → 3. Compute homography → 4. Detect ball → 5. Transform to table frame → 6. Kalman filter → 7. Output (x, y, vx, vy)

---

## 1. Project Setup

### Dependencies

```bash
pip install opencv-python opencv-contrib-python numpy
```

### Directory Structure

```
ball_tracker/
├── calibration/
│   ├── calibrate_camera.py       # Step 1: Camera intrinsics
│   ├── generate_markers.py       # Print ArUco markers
│   └── calib_images/             # Checkerboard photos
├── tracker/
│   ├── ball_detector.py          # Color-based ball detection
│   ├── table_frame.py            # ArUco detection + homography
│   ├── kalman_filter.py          # State estimation
│   └── pipeline.py               # Main tracking loop
├── config.py                     # All tunable parameters
└── main.py                       # Entry point
```

---

## 2. Configuration (config.py)

Centralize every tunable parameter so you can adjust without digging through code.

```python
import numpy as np

# --- Camera ---
CAMERA_INDEX = 0              # /dev/video0, or a path to video file
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60               # Request 60 FPS if hardware supports it

# --- ArUco Markers ---
ARUCO_DICT_NAME = "DICT_4X4_50"
MARKER_IDS = [0, 1, 2, 3]     # IDs of the 4 markers on the table

# Known positions of each marker center in TABLE coordinates (mm)
# Origin = table center, X = right, Y = up (looking from camera)
MARKER_POSITIONS_TABLE = {
    0: np.array([-120.0, -120.0]),   # bottom-left
    1: np.array([ 120.0, -120.0]),   # bottom-right
    2: np.array([-120.0,  120.0]),   # top-left
    3: np.array([ 120.0,  120.0]),   # top-right
}

# --- Ball Detection (HSV thresholds) ---
# Tune these for your specific ball color and lighting
BALL_HSV_LOWER = np.array([5, 150, 150])    # orange ball example
BALL_HSV_UPPER = np.array([15, 255, 255])
BALL_MIN_RADIUS_PX = 5        # ignore detections smaller than this
BALL_MAX_RADIUS_PX = 80

# --- Kalman Filter ---
# Process noise (how much we expect the ball to accelerate between frames)
KF_PROCESS_NOISE = 5.0
# Measurement noise (how much we trust the camera measurement)
KF_MEASUREMENT_NOISE = 2.0

# --- Control Output ---
CONTROL_LOOP_HZ = 60
```

---

## 3. Camera Calibration (calibration/calibrate_camera.py)

Lens distortion will ruin your position accuracy. Calibrate once, save, and reuse.

```python
"""
Usage:
  1. Print a checkerboard (e.g., 9x6 inner corners, 25 mm squares)
  2. Run this script, hold the board in ~20 different poses, press 's' to save each
  3. Press 'c' to compute calibration, saves to calibration_data.npz
"""
import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9, 6)       # inner corners
SQUARE_SIZE = 25.0          # mm

def collect_images(camera_index=0, save_dir="calib_images"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    count = 0

    print("Press 's' to save a frame, 'c' to calibrate, 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and found:
            path = os.path.join(save_dir, f"calib_{count:03d}.png")
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            count += 1
        elif key == ord('c'):
            break
        elif key == ord('q'):
            cap.release()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return save_dir


def calibrate(image_dir="calib_images"):
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_points = []
    img_points = []

    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    print(f"Found {len(images)} calibration images")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(objp)
            img_points.append(corners_refined)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"Camera matrix:\n{camera_matrix}")

    np.savez("calibration_data.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)
    print("Saved calibration_data.npz")
    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    save_dir = collect_images()
    if save_dir:
        calibrate(save_dir)
```

---

## 4. Generate & Print ArUco Markers (calibration/generate_markers.py)

```python
import cv2
import numpy as np

def generate_markers(marker_ids=[0, 1, 2, 3], size_px=200):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for mid in marker_ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, mid, size_px)
        # Add white border for easier detection
        bordered = cv2.copyMakeBorder(img, 40, 40, 40, 40,
                                      cv2.BORDER_CONSTANT, value=255)
        filename = f"marker_{mid}.png"
        cv2.imwrite(filename, bordered)
        print(f"Saved {filename}  (print at ~40-50mm)")

if __name__ == "__main__":
    generate_markers()
```

Print these, cut them out, and glue them at the measured positions on your table.

---

## 5. Table Frame Estimation (tracker/table_frame.py)

This is the core geometric module. It detects markers and computes the pixel-to-table homography.

```python
import cv2
import numpy as np
from config import ARUCO_DICT_NAME, MARKER_IDS, MARKER_POSITIONS_TABLE


class TableFrame:
    def __init__(self):
        dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.homography = None
        self.markers_found = 0

    def update(self, frame):
        """
        Detect ArUco markers and compute homography.
        Returns True if at least 3 markers found (enough for homography).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None:
            self.markers_found = 0
            return False

        ids = ids.flatten()

        # Collect matched point pairs: pixel <-> table coordinates
        pts_pixel = []
        pts_table = []

        for i, marker_id in enumerate(ids):
            if marker_id in MARKER_POSITIONS_TABLE:
                # Center of the marker in pixel coordinates
                marker_corners = corners_list[i][0]   # shape (4, 2)
                center_px = marker_corners.mean(axis=0)

                pts_pixel.append(center_px)
                pts_table.append(MARKER_POSITIONS_TABLE[marker_id])

        self.markers_found = len(pts_pixel)

        if self.markers_found < 3:
            return False

        pts_pixel = np.array(pts_pixel, dtype=np.float64)
        pts_table = np.array(pts_table, dtype=np.float64)

        # Compute homography: pixel -> table frame
        self.homography, status = cv2.findHomography(pts_pixel, pts_table)
        return self.homography is not None

    def pixel_to_table(self, u, v):
        """
        Convert pixel coordinates to table-frame coordinates (mm).
        Returns (x, y) in table frame, or None if no homography available.
        """
        if self.homography is None:
            return None

        pt = np.array([[[u, v]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        x, y = transformed[0, 0]
        return (x, y)

    def draw_debug(self, frame):
        """Draw detected markers and axes on the frame for debugging."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)
        return frame
```

---

## 6. Ball Detection (tracker/ball_detector.py)

```python
import cv2
import numpy as np
from config import BALL_HSV_LOWER, BALL_HSV_UPPER, BALL_MIN_RADIUS_PX, BALL_MAX_RADIUS_PX


class BallDetector:
    def __init__(self):
        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame):
        """
        Detect the ball in the frame.
        Returns (u, v, radius) in pixels, or None if not found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

        # Clean up the mask
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Pick the largest contour
        largest = max(contours, key=cv2.contourArea)
        ((u, v), radius) = cv2.minEnclosingCircle(largest)

        # Validate size
        if radius < BALL_MIN_RADIUS_PX or radius > BALL_MAX_RADIUS_PX:
            return None

        return (float(u), float(v), float(radius))

    def tune_hsv(self, frame):
        """
        Interactive HSV tuner. Run this once to find the right thresholds.
        Press 'q' to quit and print the selected values.
        """
        cv2.namedWindow("HSV Tuner")
        cv2.createTrackbar("H_lo", "HSV Tuner", int(BALL_HSV_LOWER[0]), 179, lambda x: None)
        cv2.createTrackbar("S_lo", "HSV Tuner", int(BALL_HSV_LOWER[1]), 255, lambda x: None)
        cv2.createTrackbar("V_lo", "HSV Tuner", int(BALL_HSV_LOWER[2]), 255, lambda x: None)
        cv2.createTrackbar("H_hi", "HSV Tuner", int(BALL_HSV_UPPER[0]), 179, lambda x: None)
        cv2.createTrackbar("S_hi", "HSV Tuner", int(BALL_HSV_UPPER[1]), 255, lambda x: None)
        cv2.createTrackbar("V_hi", "HSV Tuner", int(BALL_HSV_UPPER[2]), 255, lambda x: None)

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lo = np.array([cv2.getTrackbarPos("H_lo", "HSV Tuner"),
                           cv2.getTrackbarPos("S_lo", "HSV Tuner"),
                           cv2.getTrackbarPos("V_lo", "HSV Tuner")])
            hi = np.array([cv2.getTrackbarPos("H_hi", "HSV Tuner"),
                           cv2.getTrackbarPos("S_hi", "HSV Tuner"),
                           cv2.getTrackbarPos("V_hi", "HSV Tuner")])

            mask = cv2.inRange(hsv, lo, hi)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("HSV Tuner", np.hstack([frame, result]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"BALL_HSV_LOWER = np.array({lo.tolist()})")
                print(f"BALL_HSV_UPPER = np.array({hi.tolist()})")
                break

        cap.release()
        cv2.destroyAllWindows()
```

---

## 7. Kalman Filter (tracker/kalman_filter.py)

State: [x, y, vx, vy]. Measurement: [x, y]. This gives you filtered position and estimated velocity.

```python
import numpy as np
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
        #   x' = x + vx * dt
        #   y' = y + vy * dt
        #  vx' = vx
        #  vy' = vy
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
```

---

## 8. Main Pipeline (tracker/pipeline.py)

Ties everything together into the real-time tracking loop.

```python
import cv2
import numpy as np
import time

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from tracker.table_frame import TableFrame
from tracker.ball_detector import BallDetector
from tracker.kalman_filter import BallKalmanFilter


class TrackingPipeline:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.table = TableFrame()
        self.ball = BallDetector()
        self.kf = BallKalmanFilter(dt=1.0 / CAMERA_FPS)

        self.prev_time = time.time()

    def undistort(self, frame):
        """Remove lens distortion if calibration data is available."""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return frame

    def process_frame(self, frame):
        """
        Process one frame. Returns a dict with tracking results:
          {
            'ball_found': bool,
            'position': (x, y) or None,       # mm in table frame
            'velocity': (vx, vy) or None,      # mm/s in table frame
            'markers_found': int,
            'dt': float,                        # seconds since last frame
            'ball_pixel': (u, v) or None,
          }
        """
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        # Update Kalman dt if frame rate varies
        self.kf.dt = dt
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

        result = {
            'ball_found': False,
            'position': None,
            'velocity': None,
            'markers_found': 0,
            'dt': dt,
            'ball_pixel': None,
        }

        # Step 1: Undistort
        frame = self.undistort(frame)

        # Step 2: Detect markers and compute homography
        markers_ok = self.table.update(frame)
        result['markers_found'] = self.table.markers_found

        # Step 3: Detect ball in pixel space
        detection = self.ball.detect(frame)

        if detection is not None and markers_ok:
            u, v, radius = detection
            result['ball_pixel'] = (u, v)

            # Step 4: Transform to table coordinates
            table_pos = self.table.pixel_to_table(u, v)

            if table_pos is not None:
                x, y = table_pos

                # Step 5: Kalman filter
                if not self.kf.initialized:
                    self.kf.reset(x, y)
                else:
                    self.kf.predict()
                    self.kf.update(x, y)

                result['ball_found'] = True
                result['position'] = self.kf.get_position()
                result['velocity'] = self.kf.get_velocity()

        elif self.kf.initialized:
            # Ball not detected: predict only (coasts on last velocity)
            self.kf.predict()
            result['position'] = self.kf.get_position()
            result['velocity'] = self.kf.get_velocity()

        return result

    def run(self, callback=None, show_debug=True):
        """
        Main loop. Calls callback(result) every frame if provided.
        This is where you plug in your controller.
        """
        print("Starting tracking pipeline... Press 'q' to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera read failed")
                break

            result = self.process_frame(frame)

            # Send to controller
            if callback is not None:
                callback(result)

            # Debug visualization
            if show_debug:
                debug_frame = self.draw_debug(frame, result)
                cv2.imshow("Ball Tracker", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_debug(self, frame, result):
        """Overlay tracking info on the frame."""
        frame = self.table.draw_debug(frame)

        if result['ball_pixel'] is not None:
            u, v = result['ball_pixel']
            cv2.circle(frame, (int(u), int(v)), 10, (0, 255, 0), 2)

        # Display text info
        info_lines = [
            f"Markers: {result['markers_found']}",
            f"dt: {result['dt']*1000:.1f} ms",
        ]
        if result['position']:
            x, y = result['position']
            info_lines.append(f"Pos: ({x:.1f}, {y:.1f}) mm")
        if result['velocity']:
            vx, vy = result['velocity']
            info_lines.append(f"Vel: ({vx:.1f}, {vy:.1f}) mm/s")

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
```

---

## 9. Entry Point (main.py)

```python
import numpy as np
from tracker.pipeline import TrackingPipeline


def controller_callback(result):
    """
    This is where your control loop goes.
    Called every frame with the latest tracking data.
    """
    if not result['ball_found']:
        return

    x, y = result['position']
    vx, vy = result['velocity']

    # --- YOUR PID / LQR CONTROLLER HERE ---
    # Example: simple proportional control
    # target is the table center (0, 0)
    Kp = 0.01
    Kd = 0.005

    tilt_x = -Kp * x - Kd * vx
    tilt_y = -Kp * y - Kd * vy

    # Send tilt_x, tilt_y to your motor driver
    # e.g., serial_port.write(f"{tilt_x:.4f},{tilt_y:.4f}\n".encode())


def main():
    # Load calibration if available
    try:
        data = np.load("calibration/calibration_data.npz")
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Loaded camera calibration")
    except FileNotFoundError:
        print("No calibration found, running without undistortion")
        camera_matrix = None
        dist_coeffs = None

    pipeline = TrackingPipeline(camera_matrix, dist_coeffs)
    pipeline.run(callback=controller_callback, show_debug=True)


if __name__ == "__main__":
    main()
```

---

## 10. Step-by-Step Bring-Up Checklist

Work through these in order. Don't skip ahead — each step validates the previous one.

### Phase 1: Camera & Calibration
1. Verify the camera opens and streams at the desired FPS.
2. Print the checkerboard, run `calibrate_camera.py`, confirm reprojection error is < 0.5 px.
3. Print the 4 ArUco markers, glue them to the table at measured positions.

### Phase 2: Marker Detection
4. Run the pipeline, verify all 4 markers are detected (green outlines in debug view).
5. Check the homography by clicking known points on the table and printing their table-frame coordinates — they should match your ruler measurements.

### Phase 3: Ball Detection
6. Place the ball on the table, run `tune_hsv` to find reliable HSV thresholds.
7. Verify ball detection is stable: green circle tracks the ball smoothly.
8. Check for false positives — the detector shouldn't fire on the markers or table edges.

### Phase 4: Full Pipeline
9. Roll the ball slowly by hand, confirm position readout makes sense (values near 0 at center, ±120 at edges).
10. Check velocity: push the ball, velocity should spike and decay.
11. Measure latency: print timestamps at capture vs. output, aim for < 20 ms total.

### Phase 5: Control Integration
12. Wire up the controller callback to your motor driver.
13. Start with very low gains (Kp = 0.001) and increase gradually.
14. Tune the Kalman filter noise parameters if velocity is too noisy or too laggy.

---

## 11. Troubleshooting Guide

| Problem | Likely Cause | Fix |
|---|---|---|
| Markers not detected | Glare, blur, too far | Matte print, better lighting, increase marker size |
| Ball detection flickers | HSV thresholds too tight | Widen range, add Gaussian blur before HSV conversion |
| Position jumps | Homography unstable with < 4 markers | Ensure markers aren't occluded; add more markers |
| Velocity is noisy | Measurement noise too low in KF | Increase `KF_MEASUREMENT_NOISE` |
| Velocity lags behind | Process noise too low in KF | Increase `KF_PROCESS_NOISE` |
| High latency | Resolution too high, slow processing | Reduce resolution, crop ROI, skip undistortion |
| Ball sticks to edge | Ball rolls over a marker | Move markers further to corners, outside ball range |

---

## 12. Performance Optimization Tips

- **Crop the frame** to just the table region before processing. This reduces pixel count significantly.
- **Downscale** if you don't need full resolution (320×240 is often enough for position).
- **Skip undistortion** during rapid prototyping — it's expensive and the effect is small for low-distortion lenses.
- **Thread the capture**: use a separate thread to read frames so the processing loop isn't blocked by USB latency.
- **Profile** with `cv2.getTickCount()` to find bottlenecks.

---

## Notes on Coordinate Frames

```
Table frame (what your controller sees):
        +Y
        ▲
        │
  ──────┼──────► +X
        │
        │
   Origin = table center
   Units = millimeters

Camera frame:
   Origin = camera optical center
   Z = optical axis (pointing at table)
   You generally don't need this — the homography skips it entirely.
```

The homography maps directly from **pixel** to **table frame**, bypassing the camera frame. This is simpler than the full intrinsic/extrinsic decomposition and works perfectly for a planar scene.
