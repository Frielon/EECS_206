import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from tracker.table_frame import TableFrame
from tracker.ball_detector import BallDetector
from tracker.kalman_filter import BallKalmanFilter, tilt_to_accel


class TrackingPipeline:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # Discard initial frames so camera auto-exposure settles
        for _ in range(30):
            self.cap.read()

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

        self.table = TableFrame()
        self.ball = BallDetector()
        self.kf = BallKalmanFilter(dt=1.0 / CAMERA_FPS)

        # Latest commanded plate tilt (radians). Updated by the controller via
        # set_plate_angles(); fed to the KF as a control input each frame.
        self.plate_angles = (0.0, 0.0)

        self.prev_time = time.time()

    def set_plate_angles(self, theta_x, theta_y):
        """
        Tell the tracker the current commanded plate tilt (radians). Call this
        from your controller whenever the command changes; the KF uses it to
        predict gravity-induced acceleration between frames.
        """
        self.plate_angles = (float(theta_x), float(theta_y))

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

        # Feed the latest commanded plate tilt into the KF as a control input.
        # Gravity is now modeled deterministically; Q only handles slip/friction.
        a_x, a_y = tilt_to_accel(*self.plate_angles)
        self.kf.set_control(a_x, a_y)

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
            result['ball_radius'] = radius

            # Step 4: Transform to table coordinates
            table_pos = self.table.pixel_to_table(u, v)

            if table_pos is not None:
                x, y = table_pos

                # Step 5: Kalman filter
                if not self.kf.initialized:
                    self.kf.reset(x, y)
                else:
                    self.kf.predict(dt=dt)
                    self.kf.update(x, y)

                result['ball_found'] = True
                result['position'] = self.kf.get_position()
                result['velocity'] = self.kf.get_velocity()

        elif self.kf.initialized:
            # Ball not detected: predict only (coasts on last velocity + tilt).
            self.kf.predict(dt=dt)
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
            r = int(result.get('ball_radius', 10))
            cv2.circle(frame, (int(u), int(v)), r, (0, 255, 0), 2)

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
