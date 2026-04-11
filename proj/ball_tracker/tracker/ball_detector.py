import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

    def tune_hsv(self, camera_index=0):
        """
        Interactive HSV tuner with slider bars.
        Drag the sliders to adjust thresholds. Type 'q' + enter in terminal to quit.
        """
        import threading

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {camera_index}")
            return

        # Create window with trackbars
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 400, 300)
        cv2.createTrackbar("H_lo", "Controls", int(BALL_HSV_LOWER[0]), 179, lambda x: None)
        cv2.createTrackbar("S_lo", "Controls", int(BALL_HSV_LOWER[1]), 255, lambda x: None)
        cv2.createTrackbar("V_lo", "Controls", int(BALL_HSV_LOWER[2]), 255, lambda x: None)
        cv2.createTrackbar("H_hi", "Controls", int(BALL_HSV_UPPER[0]), 179, lambda x: None)
        cv2.createTrackbar("S_hi", "Controls", int(BALL_HSV_UPPER[1]), 255, lambda x: None)
        cv2.createTrackbar("V_hi", "Controls", int(BALL_HSV_UPPER[2]), 255, lambda x: None)

        running = [True]

        def wait_for_quit():
            """Wait for 'q' + enter in terminal to quit."""
            while running[0]:
                try:
                    line = input()
                    if line.strip() == 'q':
                        running[0] = False
                        break
                except EOFError:
                    break

        t = threading.Thread(target=wait_for_quit, daemon=True)
        t.start()

        print("=== HSV Tuner ===")
        print("Drag the sliders in the 'Controls' window to adjust thresholds.")
        print("Type 'q' + enter in terminal to quit and print values.")

        while running[0]:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Read slider values
            lo = np.array([
                cv2.getTrackbarPos("H_lo", "Controls"),
                cv2.getTrackbarPos("S_lo", "Controls"),
                cv2.getTrackbarPos("V_lo", "Controls"),
            ])
            hi = np.array([
                cv2.getTrackbarPos("H_hi", "Controls"),
                cv2.getTrackbarPos("S_hi", "Controls"),
                cv2.getTrackbarPos("V_hi", "Controls"),
            ])

            mask = cv2.inRange(hsv, lo, hi)
            masked = cv2.bitwise_and(frame, frame, mask=mask)

            # Show values on camera feed
            cv2.putText(frame, f"Lo: H={lo[0]} S={lo[1]} V={lo[2]}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Hi: H={hi[0]} S={hi[1]} V={hi[2]}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            px_count = cv2.countNonZero(mask)
            cv2.putText(frame, f"Mask pixels: {px_count}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("HSV Tuner", np.hstack([frame, masked]))
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

        lo_list = [int(x) for x in lo]
        hi_list = [int(x) for x in hi]
        print(f"\n=== Copy these into config.py ===")
        print(f"BALL_HSV_LOWER = np.array({lo_list})")
        print(f"BALL_HSV_UPPER = np.array({hi_list})")
