import cv2
from src.vision import VisionManager

# 1. Setup
cap = cv2.VideoCapture(0)
vm = VisionManager()

cv2.namedWindow("Original")
cv2.setMouseCallback("Original", vm.click_corner)

print("--- INTEGRATED TEST MODE ---")
print("1. Click 4 corners to calibrate.")
print("2. Once calibrated, detection will start automatically in the Warped window.")
print("Commands: [q] Quit, [r] Reset Calibration")

while True:
    # Read frame ONLY ONCE per loop
    ret, frame = cap.read()
    if not ret:
        break

    # ---ORIGINAL VIEW & CALIBRATION ---
    display_frame = vm.draw_points(frame.copy())
    cv2.imshow("Original", display_frame)

    # --- WARPED VIEW & DETECTION ---
    warped = vm.get_warped_frame(frame)

    if warped is not None:
        # Create a copy for drawing detections so we don't mess up the raw warped frame
        detection_view = warped.copy()

        # Detect Ball
        ball_pos = vm.detect_ball(detection_view)
        if ball_pos:
            cv2.circle(detection_view, ball_pos, 10, (0, 255, 255), -1)

        #  Detect Cups
        cups = vm.detect_cups(detection_view)
        for cup in cups:
            x1, y1, x2, y2 = cup['box']
            cv2.rectangle(detection_view, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(detection_view, cup['pos'], 5, (0, 0, 255), -1)

        cv2.imshow("Bird's Eye Detection", detection_view)

    # --- KEYBOARD HANDLING ---
    # Call waitKey ONLY ONCE per loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        vm.points = []
        vm.homography_matrix = None
        print("Resetting calibration...")

cap.release()
cv2.destroyAllWindows()