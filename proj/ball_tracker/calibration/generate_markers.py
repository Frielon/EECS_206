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
