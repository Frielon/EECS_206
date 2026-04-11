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
