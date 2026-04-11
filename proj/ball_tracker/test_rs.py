import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for _ in range(30):
    cap.read()

ret, frame = cap.read()
cap.release()

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower = np.array([8, 180, 120])
upper = np.array([25, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_clean = cv2.erode(mask, kernel, iterations=1)
mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

# Draw centroid and minEnclosingCircle on frame
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest = max(contours, key=cv2.contourArea)
    ((cu, cv_), radius) = cv2.minEnclosingCircle(largest)
    M = cv2.moments(largest)
    if M["m00"] > 0:
        mu = M["m10"] / M["m00"]
        mv = M["m01"] / M["m00"]
        # Red = minEnclosingCircle center, Blue = centroid
        cv2.circle(frame, (int(cu), int(cv_)), int(radius), (0, 0, 255), 2)  # red circle
        cv2.circle(frame, (int(mu), int(mv)), 5, (255, 0, 0), -1)  # blue dot = centroid
        cv2.circle(frame, (int(cu), int(cv_)), 5, (0, 0, 255), -1)  # red dot = enclosing center
        print(f"Enclosing circle center: ({cu:.0f}, {cv_:.0f}), radius: {radius:.0f}")
        print(f"Centroid: ({mu:.0f}, {mv:.0f})")
        print(f"Offset: ({mu-cu:.0f}, {mv-cv_:.0f})")

cv2.imwrite("debug_centers.png", frame)
cv2.imwrite("debug_mask_current.png", mask_clean)
print("Saved debug_centers.png and debug_mask_current.png")
