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

CHECKERBOARD = (8, 5)       # inner corners (for a 9x6 squares board)
SQUARE_SIZE = 30.0          # mm (3.0 cm)


def collect_images(camera_index=0, save_dir="calib_images"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    count = 0

    print("Controls (press in terminal or camera window):")
    print("  s = save frame    c = calibrate    q = quit")
    print("-----------------------------------------------")

    import threading, sys

    key_pressed = [None]

    def read_keys():
        """Read single keypresses from terminal as fallback."""
        while True:
            line = input()
            if line:
                key_pressed[0] = line[0]

    t = threading.Thread(target=read_keys, daemon=True)
    t.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)
            cv2.putText(display, "CORNERS FOUND - press 's' to save",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No corners detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Saved: {count} frames",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        # Also check terminal input
        if key_pressed[0] is not None:
            key = ord(key_pressed[0])
            key_pressed[0] = None

        if key == ord('s'):
            if found:
                path = os.path.join(save_dir, f"calib_{count:03d}.png")
                cv2.imwrite(path, frame)
                print(f"[{count+1}] Saved {path}")
                count += 1
            else:
                print("No corners detected - move the checkerboard and try again")
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

    gray = None
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

    if not obj_points:
        print("No valid calibration images found!")
        return None, None

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
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save_dir = collect_images(camera_index=cam_idx)
    if save_dir:
        calibrate(save_dir)
