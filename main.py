# --- ADD THESE IMPORTS AT THE TOP ---
import rospy
from src.sawyer_interface import SawyerInterface
# ------------------------------------

import cv2
import numpy as np
from src.vision import VisionManager
from src.tracker import Tracker

# Initialize ROS node for Intera SDK interaction
rospy.init_node("sawyer_shell_game_vision")

cap = cv2.VideoCapture(0)
# Open secondary video capture for the wrist camera
# (0 is typically built-in or external webcam, 1 or 2 is usually the wrist)
wrist_cap = cv2.VideoCapture(1)

vm = VisionManager()
tracker = Tracker()
robot = SawyerInterface(workspace_width_m=0.4, workspace_height_m=0.4)  # Initialize Robot

# Expanded Game State Variables
state = "CALIBRATING"
winning_id = None

cv2.namedWindow("Original")
cv2.setMouseCallback("Original", vm.click_corner)

print("--- SAWYER SHELL GAME: BOOTING UP ---")

while not rospy.is_shutdown():  # Clean ROS loop hook
    ret, frame = cap.read()
    if not ret: break

    display_frame = vm.draw_points(frame.copy())
    cv2.imshow("Original", display_frame)

    warped = vm.get_warped_frame(frame)
    if warped is not None:
        view = warped.copy()

        ball_pos = vm.detect_ball(warped)
        cup_blobs = vm.detect_cups(warped)
        cup_centers = [c['pos'] for c in cup_blobs]

        tracked_cups = tracker.update(cup_centers)

        # --- ROBUST STATE HANDLING & HANDOFF ---
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

        # Phase 1 Execution Triggered by State Machine
        if state == "REVEALING":
            if winning_id is not None and winning_id in tracked_cups:
                win_pixel = tracked_cups[winning_id]
                # Command macro-movement to hover over the cup using the tracker location
                success = robot.go_to_hover(win_pixel[0], win_pixel[1])
                if success:
                    state = "ROBOT_DESCENT"
                else:
                    state = "IDLE"
            else:
                print("Cannot reveal: Cup lost from top-down tracker view.")
                state = "IDLE"

        # Phase 2 Execution (Visual Servoing)
        if state == "ROBOT_DESCENT":
            ret_w, w_frame = wrist_cap.read()
            if ret_w:
                # Reuse your color mask/contour system tuned for the wrist frame view
                # Pass your wrist frame through your cup detector
                wrist_cups = vm.detect_cups(w_frame)

                if wrist_cups:
                    # When hovering directly over, the closest or largest cup in the
                    # wrist view will step-correct directly to the center
                    target_cup_wrist = wrist_cups[0]['pos']

                    # Draw a dot over what the wrist sees for debugging
                    cv2.circle(w_frame, target_cup_wrist, 5, (255, 0, 0), -1)
                    cv2.imshow("Sawyer Wrist Camera View", w_frame)

                    # Run a visual servoing iteration step
                    finished = robot.visual_servoing_step(target_cup_wrist)
                    if finished:
                        state = "IDLE"  # Goal reached, return to idle
                else:
                    # Safe Halt condition: if the cup disappears out of frame, freeze
                    robot.visual_servoing_step(None)
            else:
                state = "IDLE"

        # --- DRAW VISUALIZATIONS ---
        if ball_pos:
            cv2.circle(view, ball_pos, 10, (0, 255, 255), -1)

        for cup_id, pos in tracked_cups.items():
            color = (0, 255, 0) if cup_id == winning_id else (0, 0, 255)
            cv2.circle(view, pos, 8, color, -1)
            cv2.putText(view, f"ID:{cup_id}", (pos[0] - 10, pos[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cup_id == winning_id:
                cv2.putText(view, "WINNER", (pos[0] - 25, pos[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(view, f"STATE: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bird's Eye Game View", view)

    # --- KEYBOARD COMMANDS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'):
        vm.points = [];
        vm.homography_matrix = None
        state = "CALIBRATING";
        winning_id = None
        print("Resetting...")

    if key == ord('s') and state == "CALIBRATING":
        state = "IDLE"
        print("Calibration Locked. Ready for Game.")

    if key == ord('x') and state == "IDLE":
        state = "LOCKED"

    # New hook: Press 'e' when shuffling is completely finished to execute point movement
    if key == ord('e') and state == "TRACKING":
        state = "REVEALING"
        print("Shuffle finished! Directing Sawyer to winning cup...")

cap.release()
wrist_cap.release()
cv2.destroyAllWindows()