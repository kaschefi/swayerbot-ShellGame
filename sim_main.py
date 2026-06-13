#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Trigger, SetBool
import os
os.environ["ROS_MASTER_URI"] = "http://localhost:11311"

# Importing from your project src directory
from src.sawyer_interface import SawyerInterface
from src.vision import VisionManager
from src.tracker import Tracker

class SimShellGameNode:
    def __init__(self):
            rospy.init_node("sawyer_shell_game_vision", anonymous=True)
            self.bridge = CvBridge()
            self.rate = rospy.Rate(20)

            self.vm = VisionManager()
            self.tracker = Tracker()
            
            self.state = "CALIBRATING"
            self.winning_id = None
            self.latest_frame = None

            print("Waiting for Gazebo game controller services to come online...")
            rospy.wait_for_service('/shell_game/hide_ball')
            rospy.wait_for_service('/shell_game/toggle_shuffle')
            
            self.call_hide_srv = rospy.ServiceProxy('/shell_game/hide_ball', Trigger)
            self.call_shuffle_srv = rospy.ServiceProxy('/shell_game/toggle_shuffle', SetBool)

            self.image_sub = rospy.Subscriber(
                "/io/internal_camera/right_hand_camera/image_raw", 
                Image, 
                self.image_callback
            )
            
            cv2.namedWindow("Original")
            cv2.setMouseCallback("Original", self.vm.click_corner)


            print("[SimShellGameNode] Initializing Sawyer Motion Backend...")
            self.robot = SawyerInterface(workspace_width_m=0.5, workspace_height_m=0.5)

            print("--- VISUAL SERVOING SIMULATION READY ---")
            print("Steps: 1. Click 4 corners & press 'S' | 2. Press 'X' to lock ball | 3. Press 'H' to hide | 4. Press '1' to shuffle")

    def image_callback(self, data):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def run(self):
        while not rospy.is_shutdown():
            if self.latest_frame is None:
                self.rate.sleep()
                continue

            frame = self.latest_frame.copy()
            display_frame = self.vm.draw_points(frame.copy())
            cv2.imshow("Original", display_frame)

            warped = self.vm.get_warped_frame(frame)
            if warped is not None:
                view = warped.copy()

                # Detect objects via custom HSV color thresholds
                ball_pos = self.vm.detect_ball(warped)
                cup_blobs = self.vm.detect_cups(warped)
                cup_centers = [c['pos'] for c in cup_blobs]

                # Update the Tracker layers
                tracked_cups = self.tracker.update(cup_centers)

                # --- STATE MACHINE LOGIC ---
                if self.state == "LOCKED":
                    if ball_pos:
                        self.winning_id = self.tracker.assign_winner(ball_pos)
                        if self.winning_id is not None:
                            print(f"BALL LOCKED! Tracking Cup ID: {self.winning_id}")
                            self.state = "TRACKING"
                        else:
                            self.state = "IDLE"
                    else:
                        print("No ball detected yet. Ensure it is in frame before pressing X.")
                        self.state = "IDLE"

                if self.state == "REVEALING":
                    if self.winning_id is not None and self.winning_id in tracked_cups:
                        win_pixel = tracked_cups[self.winning_id]
                        print(f"[Reveal] Moving to hover above cup at pixel {win_pixel}")

                        # Freeze the frame — take one snapshot of position, move, done
                        success = self.robot.go_to_hover(win_pixel[0], win_pixel[1])

                        if success:
                            print("[SUCCESS] Robot is hovering above the winning cup!")
                        else:
                            print("[FAIL] Could not reach target.")

                        self.state = "IDLE"  # either way, stop trying
                    else:
                        print("[Reveal] Winning cup lost from view!")
                        self.state = "IDLE"

                # --- DRAW RENDERS ---
                if ball_pos:
                    cv2.circle(view, ball_pos, 10, (0, 255, 255), -1)

                for cup_id, pos in tracked_cups.items():
                    color = (0, 255, 0) if cup_id == self.winning_id else (0, 0, 255)
                    cv2.circle(view, pos, 8, color, -1)
                    cv2.putText(view, f"ID:{cup_id}", (pos[0] - 10, pos[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if cup_id == self.winning_id:
                        cv2.putText(view, "WINNER", (pos[0] - 25, pos[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(view, f"STATE: {self.state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Bird's Eye Game View", view)

            # --- KEYBOARD MAP INTERACTION ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.robot.stop()
                break
                
            if key == ord('s') and self.state == "CALIBRATING":
                self.state = "IDLE"
                print("Workspace Calibration Saved. Ready for tracking.")
                
            if key == ord('x') and self.state == "IDLE":
                self.state = "LOCKED"
                
            if key == ord('h') or key == ord('H'):
                print("Calling Simulation Hide Service...")
                try:
                    res = self.call_hide_srv()
                    print(f"Service Response: {res.message}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Hide Ball service call failed: {e}")
                
            if key == ord('1'):
                print("Calling Simulation Shuffle Service [START]...")
                try:
                    res = self.call_shuffle_srv(True)
                    print(f"Service Response: {res.message}")
                except rospy.ServiceException as e:
                    rospy.logerr(f"Start Shuffle service call failed: {e}")
                
            if key == ord('2'):
                print("Calling Simulation Shuffle Service [STOP]... Setting state back to TRACKING.")
                try:
                    res = self.call_shuffle_srv(False)
                    print(f"Service Response: {res.message}")
                    self.state = "TRACKING" 
                except rospy.ServiceException as e:
                    rospy.logerr(f"Stop Shuffle service call failed: {e}")
                
            # Press 'E' -> Triggers continuous Visual Servoing loop
            if key == ord('e') and self.state == "TRACKING":
                print("Initiating Visual Servoing tracking approach loop...")
                self.state = "REVEALING"

            self.rate.sleep()

if __name__ == '__main__':
    node = SimShellGameNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()