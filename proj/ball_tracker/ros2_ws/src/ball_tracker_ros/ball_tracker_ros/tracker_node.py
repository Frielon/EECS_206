import rclpy
from rclpy.node import Node
from ball_tracker_msgs.msg import BallState
from geometry_msgs.msg import Vector3Stamped

import cv2
import numpy as np
import time
import sys
import os

# Add the ball_tracker package root to path so we can import config & tracker modules
# Navigate from ros2_ws (found via environment or known structure) up to ball_tracker/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Walk up until we find ros2_ws, then go one level above it
_d = _THIS_DIR
while os.path.basename(_d) != 'ros2_ws' and _d != '/':
    _d = os.path.dirname(_d)
BALL_TRACKER_ROOT = os.path.dirname(_d)  # parent of ros2_ws = ball_tracker/
sys.path.insert(0, BALL_TRACKER_ROOT)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    KF_PROCESS_NOISE, KF_MEASUREMENT_NOISE,
)
from tracker.table_frame import TableFrame
from tracker.ball_detector import BallDetector
from tracker.kalman_filter import BallKalmanFilter, tilt_to_accel


class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker')

        # -- Declare ROS2 parameters (overridable at launch) --
        self.declare_parameter('camera_index', CAMERA_INDEX)
        self.declare_parameter('camera_width', CAMERA_WIDTH)
        self.declare_parameter('camera_height', CAMERA_HEIGHT)
        self.declare_parameter('camera_fps', CAMERA_FPS)
        self.declare_parameter('show_debug', True)
        self.declare_parameter('calibration_file', '')
        # Drop the tilt control input if the latest /plate_cmd is older than
        # this many seconds (controller stopped, paused, or crashed).
        self.declare_parameter('plate_cmd_timeout_s', 0.3)

        cam_idx = self.get_parameter('camera_index').value
        cam_w = self.get_parameter('camera_width').value
        cam_h = self.get_parameter('camera_height').value
        cam_fps = self.get_parameter('camera_fps').value
        self.show_debug = self.get_parameter('show_debug').value
        calib_file = self.get_parameter('calibration_file').value
        self.plate_cmd_timeout_s = float(self.get_parameter('plate_cmd_timeout_s').value)

        # -- Camera --
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        self.cap.set(cv2.CAP_PROP_FPS, cam_fps)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {cam_idx}')
            raise RuntimeError(f'Cannot open camera {cam_idx}')

        # -- Calibration --
        self.camera_matrix = None
        self.dist_coeffs = None
        if calib_file:
            self._load_calibration(calib_file)
        else:
            # Try default path
            default_path = os.path.join(BALL_TRACKER_ROOT, 'calibration', 'calibration_data.npz')
            if os.path.isfile(default_path):
                self._load_calibration(default_path)

        # -- Tracker components --
        self.table = TableFrame()
        self.ball = BallDetector()
        self.kf = BallKalmanFilter(dt=1.0 / cam_fps)
        self.prev_time = time.time()

        # -- Publisher --
        self.pub_ball_state = self.create_publisher(BallState, 'ball_state', 10)

        # -- Subscriber: tilt command from controller (drives KF control input) --
        self.plate_angles = (0.0, 0.0)        # (theta_x, theta_y), radians
        self.plate_cmd_stamp = None           # builtin_interfaces/Time of last cmd
        self.sub_plate_cmd = self.create_subscription(
            Vector3Stamped, '/plate_cmd', self._plate_cmd_callback, 10
        )

        # -- Timer drives the loop at camera fps --
        timer_period = 1.0 / cam_fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Ball tracker started: camera={cam_idx}, {cam_w}x{cam_h}@{cam_fps}fps'
        )

    def _plate_cmd_callback(self, msg: Vector3Stamped):
        """Cache the latest commanded plate tilt for the KF predict step."""
        self.plate_angles = (float(msg.vector.x), float(msg.vector.y))
        self.plate_cmd_stamp = msg.header.stamp

    def _current_control(self):
        """
        Convert the latest commanded tilt to table-frame acceleration. Returns
        (0, 0) if no command has been received or if the last one is stale,
        which makes the KF degrade cleanly to constant-velocity.
        """
        if self.plate_cmd_stamp is None:
            return 0.0, 0.0
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = (
            int(self.plate_cmd_stamp.sec) * 1_000_000_000
            + int(self.plate_cmd_stamp.nanosec)
        )
        age_s = (now_ns - stamp_ns) * 1e-9
        if age_s > self.plate_cmd_timeout_s:
            return 0.0, 0.0
        return tilt_to_accel(*self.plate_angles)

    def _load_calibration(self, path):
        try:
            data = np.load(path)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.get_logger().info(f'Loaded calibration from {path}')
        except Exception as e:
            self.get_logger().warn(f'Failed to load calibration: {e}')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera read failed')
            return

        # Undistort
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

        # Timing
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        # Push the latest tilt command into the KF as a control input. The
        # filter's predict(dt=...) call below also rebuilds F/B/Q for the
        # current dt, so we no longer mutate self.kf.F directly.
        a_x, a_y = self._current_control()
        self.kf.set_control(a_x, a_y)

        # Detect markers + homography
        markers_ok = self.table.update(frame)

        # Detect ball
        detection = self.ball.detect(frame)

        # Build message
        msg = BallState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'table'
        msg.markers_found = self.table.markers_found
        msg.ball_found = False
        msg.pixel_u = 0.0
        msg.pixel_v = 0.0

        if detection is not None and markers_ok:
            u, v, radius = detection
            msg.pixel_u = u
            msg.pixel_v = v

            table_pos = self.table.pixel_to_table(u, v)
            if table_pos is not None:
                x, y = table_pos
                if not self.kf.initialized:
                    self.kf.reset(x, y)
                else:
                    self.kf.predict(dt=dt)
                    self.kf.update(x, y)

                msg.ball_found = True
                msg.x, msg.y = self.kf.get_position()
                msg.vx, msg.vy = self.kf.get_velocity()

        elif self.kf.initialized:
            # Coast on prediction (still uses the current tilt command via B*u)
            self.kf.predict(dt=dt)
            msg.x, msg.y = self.kf.get_position()
            msg.vx, msg.vy = self.kf.get_velocity()

        self.pub_ball_state.publish(msg)

        # Debug window
        if self.show_debug:
            self._draw_debug(frame, msg)

    def _draw_debug(self, frame, msg):
        frame = self.table.draw_debug(frame)

        if msg.pixel_u != 0.0 or msg.pixel_v != 0.0:
            cv2.circle(frame, (int(msg.pixel_u), int(msg.pixel_v)), 10, (0, 255, 0), 2)

        lines = [f'Markers: {msg.markers_found}']
        if msg.ball_found or (msg.x != 0.0 or msg.y != 0.0):
            lines.append(f'Pos: ({msg.x:.1f}, {msg.y:.1f}) mm')
            lines.append(f'Vel: ({msg.vx:.1f}, {msg.vy:.1f}) mm/s')

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Ball Tracker', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
