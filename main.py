import cv2
import numpy as np
from src.vision import VisionManager
from src.tracker import Tracker

cap = cv2.VideoCapture(0)
vm = VisionManager()
tracker = Tracker()

# Game State Variables
state = "CALIBRATING"
winning_id = None

# Set up the calibration window
cv2.namedWindow("Original")
cv2.setMouseCallback("Original", vm.click_corner)

print("--- SAWYER SHELL GAME: BOOTING UP ---")

while True:
    ret, frame = cap.read()
    if not ret: break

    # ALWAYS DRAW THE ORIGINAL VIEW FOR CALIBRATION
    display_frame = vm.draw_points(frame.copy())
    cv2.imshow("Original", display_frame)

    # PROCESS THE WARPED VIEW
    warped = vm.get_warped_frame(frame)
    if warped is not None:
        view = warped.copy()

        ball_pos = vm.detect_ball(warped)
        cup_blobs = vm.detect_cups(warped)
        cup_centers = [c['pos'] for c in cup_blobs]


        tracked_cups = tracker.update(cup_centers)
        print(f"VISION SEES: {len(cup_centers)} cups | TRACKER HOLDS: {len(tracked_cups)} IDs")

        if state == "LOCKED":
            if ball_pos:
                winning_id = tracker.assign_winner(ball_pos)
                if winning_id is not None:
                    print(f"BALL FOUND! Tracking Cup ID: {winning_id}")
                    state = "TRACKING"
                else:
                    print("Place the ball closer to a cup!")
                    state = "IDLE"
            else:
                print("No ball detected. Show the ball first!")
                state = "IDLE"

        if ball_pos:
            cv2.circle(view, ball_pos, 10, (0, 255, 255), -1)

        # Draw the Cups and their IDs
        for cup_id, pos in tracked_cups.items():
            color = (0, 255, 0) if cup_id == winning_id else (0, 0, 255)
            # Label the cup on screen
            cv2.circle(view, pos, 8, color, -1)
            cv2.putText(view, f"ID:{cup_id}", (pos[0] - 10, pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cup_id == winning_id:
                cv2.putText(view, "WINNER", (pos[0] - 25, pos[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display current State
        cv2.putText(view, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bird's Eye Game View", view)

    # --- KEYBOARD COMMANDS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'):
        vm.points = [];
        vm.homography_matrix = None
        state = "CALIBRATING"
        winning_id = None
        print("Resetting...")

    if key == ord('s') and state == "CALIBRATING":
        state = "IDLE"
        print("Calibration Locked. Ready for Game.")

    if key == ord('x') and state == "IDLE":
        state = "LOCKED"

cap.release()
cv2.destroyAllWindows()