import cv2
import numpy as np


class VisionManager:
    def __init__(self, desired_width=500, desired_height=500):
        self.points = []
        self.homography_matrix = None
        # The dimensions of the "Virtual Table" in pixels
        self.width = desired_width
        self.height = desired_height

        # Define where the 4 points should land in the warped view
        self.pts_dst = np.array([
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height]
        ], dtype="float32")

    def click_corner(self, event, x, y, flags, param):
        """Mouse callback to capture 4 corners of the workspace."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Point {len(self.points)} captured: ({x}, {y})")

            if len(self.points) == 4:
                self.calculate_homography()

    def calculate_homography(self):
        """Computes the 3x3 Homography matrix."""
        pts_src = np.array(self.points, dtype="float32")
        self.homography_matrix = cv2.getPerspectiveTransform(pts_src, self.pts_dst)
        print("Homography Matrix successfully calculated!")

    def get_warped_frame(self, frame):
        """Applies the perspective transformation to a raw frame."""
        if self.homography_matrix is not None:
            return cv2.warpPerspective(frame, self.homography_matrix, (self.width, self.height))
        return None

    def draw_points(self, frame):
        """Visual feedback: Draw dots on the original view."""
        for pt in self.points:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        return frame