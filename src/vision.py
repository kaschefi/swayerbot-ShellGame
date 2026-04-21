import cv2
import numpy as np
from ultralytics import YOLO

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
        # Define Orange Color Range (HSV) for the Ping Pong Ball
        self.orange_lower = np.array([5, 150, 150])
        self.orange_upper = np.array([15, 255, 255])


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

    def detect_ball(self, warped_frame):
        """Finds the orange ball using color masking."""
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.orange_lower, self.orange_upper)

        # Clean up noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest orange blob
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 5:  # Filter out tiny noise
                return (int(x), int(y))
        return None

    def detect_cups(self, warped_frame):
        """Finds the 3 largest red objects on the table."""
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cv2.imshow("Blue Mask Debug", mask)

        # Clean up noise (important if the blue is dark)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area (largest first) and take top 3
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        cups = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 400:  # Filter small noise
                x, y, w, h = cv2.boundingRect(cnt)
                cups.append({
                    'box': (x, y, x + w, y + h),
                    'pos': (int(x + w / 2), y + h)  # Bottom center
                })
        return cups