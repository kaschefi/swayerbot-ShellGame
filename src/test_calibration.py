import cv2
from src.vision import VisionManager

# Initialize the camera and our manager
cap = cv2.VideoCapture(0) #laptop's built-in webcam
vm = VisionManager()

cv2.namedWindow("Original")
cv2.setMouseCallback("Original", vm.click_corner)

print("--- CALIBRATION MODE ---")
print("Click the corners of your table in order:")
print("1. Top-Left -> 2. Top-Right -> 3. Bottom-Right -> 4. Bottom-Left")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Draw the green dots where you clicked
    display_frame = vm.draw_points(frame.copy())
    cv2.imshow("Original", display_frame)

    # If calibrated, show the Bird's Eye View
    warped = vm.get_warped_frame(frame)
    if warped is not None:
        cv2.imshow("Bird's Eye View", warped)

    # Press 'q' to quit or 'r' to reset points
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        vm.points = []
        vm.homography_matrix = None
        print("Points reset. Recalibrate now.")

cap.release()
cv2.destroyAllWindows()