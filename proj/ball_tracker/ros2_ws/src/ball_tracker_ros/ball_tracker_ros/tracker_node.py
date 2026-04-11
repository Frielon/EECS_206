import rclpy
from rclpy.node import Node
from ball_tracker_msgs.msg import BallState

import cv2
import numpy as np
import time
import sys
import os

# Add the ball_tracker package root to path so we can import config & tracker modules
BALL_TRACKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, BALL_TRACKER_ROOT)

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    KF_PROCESS_NOISE, KF_MEASUREMENT_NOISE,
)
from tracker.table_frame import TableFrame
from tracker.ball_detector import BallDetector
from tracker.kalman_filter import BallKalmanFilter


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

        cam_idx = self.get_parameter('camera_index').value
        cam_w = self.get_parameter('camera_width').value
        cam_h = self.get_parameter('camera_height').value
        cam_fps = self.get_parameter('camera_fps').value
        self.show_debug = self.get_parameter('show_debug').value
        calib_file = self.get_parameter('calibration_file').value

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

        # -- Timer drives the loop at camera fps --
        timer_period = 1.0 / cam_fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Ball tracker started: camera={cam_idx}, {cam_w}x{cam_h}@{cam_fps}fps'
        )

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

        # Update Kalman dt
        self.kf.dt = dt
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt

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
                    self.kf.predict()
                    self.kf.update(x, y)

                msg.ball_found = True
                msg.x, msg.y = self.kf.get_position()
                msg.vx, msg.vy = self.kf.get_velocity()

        elif self.kf.initialized:
            # Coast on prediction
            self.kf.predict()
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
