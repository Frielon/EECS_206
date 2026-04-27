import numpy as np
from tracker.pipeline import TrackingPipeline


def make_controller_callback(pipeline):
    """
    Build a control-loop callback bound to a specific pipeline so the
    commanded plate tilt can be fed back into the KF each frame.
    """
    Kp = 0.01
    Kd = 0.005

    def controller_callback(result):
        if not result['ball_found']:
            return

        x, y   = result['position']
        vx, vy = result['velocity']

        # Simple PD: tilt about x-axis controls y motion, tilt about y-axis controls x.
        theta_x = -Kp * y - Kd * vy
        theta_y =  Kp * x + Kd * vx

        # Tell the tracker about the commanded tilt so the KF can use it as
        # the control input (a = alpha * g * sin(theta)) for the next predict.
        pipeline.set_plate_angles(theta_x, theta_y)

        # Send theta_x, theta_y to your motor driver here, e.g.:
        # serial_port.write(f"{theta_x:.4f},{theta_y:.4f}\n".encode())

    return controller_callback


def main():
    # Load calibration if available
    try:
        data = np.load("/home/cc/ee106a/sp26/class/ee106a-acq/proj/EECS_206/proj/ball_tracker/calibration/calibration_data.npz")
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Loaded camera calibration")
    except FileNotFoundError:
        print("No calibration found, running without undistortion")
        camera_matrix = None
        dist_coeffs = None

    pipeline = TrackingPipeline(camera_matrix, dist_coeffs)
    pipeline.run(callback=make_controller_callback(pipeline), show_debug=True)


if __name__ == "__main__":
    main()
