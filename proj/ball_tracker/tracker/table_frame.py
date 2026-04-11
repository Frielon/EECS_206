import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import ARUCO_DICT_NAME, MARKER_IDS, MARKER_POSITIONS_TABLE


class TableFrame:
    def __init__(self):
        dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.homography = None
        self.markers_found = 0

    def update(self, frame):
        """
        Detect ArUco markers and compute homography.
        Returns True if at least 3 markers found (enough for homography).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None:
            self.markers_found = 0
            return False

        ids = ids.flatten()

        # Collect matched point pairs: pixel <-> table coordinates
        pts_pixel = []
        pts_table = []

        for i, marker_id in enumerate(ids):
            if marker_id in MARKER_POSITIONS_TABLE:
                # Center of the marker in pixel coordinates
                marker_corners = corners_list[i][0]   # shape (4, 2)
                center_px = marker_corners.mean(axis=0)

                pts_pixel.append(center_px)
                pts_table.append(MARKER_POSITIONS_TABLE[marker_id])

        self.markers_found = len(pts_pixel)

        if self.markers_found < 4:
            return False

        pts_pixel = np.array(pts_pixel, dtype=np.float64)
        pts_table = np.array(pts_table, dtype=np.float64)

        # Compute homography: pixel -> table frame
        self.homography, status = cv2.findHomography(pts_pixel, pts_table)
        return self.homography is not None

    def pixel_to_table(self, u, v):
        """
        Convert pixel coordinates to table-frame coordinates (mm).
        Returns (x, y) in table frame, or None if no homography available.
        """
        if self.homography is None:
            return None

        pt = np.array([[[u, v]]], dtype=np.float64)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        x, y = transformed[0, 0]
        return (x, y)

    def draw_debug(self, frame):
        """Draw detected markers and axes on the frame for debugging."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners_list, ids)
        return frame
