import numpy as np

# --- Camera ---
CAMERA_INDEX = 0             # /dev/video0, or a path to video file
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
BALL_HSV_LOWER = np.array([14, 148, 165])    # orange ball example
BALL_HSV_UPPER = np.array([19, 255, 255])
BALL_MIN_RADIUS_PX = 5        # ignore detections smaller than this
BALL_MAX_RADIUS_PX = 80

# --- Kalman Filter ---
# Process noise (how much we expect the ball to accelerate between frames)
KF_PROCESS_NOISE = 5.0
# Measurement noise (how much we trust the camera measurement)
KF_MEASUREMENT_NOISE = 2.0

# --- Control Output ---
CONTROL_LOOP_HZ = 60
