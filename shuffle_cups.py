#!/usr/bin/env python3
import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse

class GazeboGameController:
    def __init__(self):
        rospy.init_node('gazebo_game_controller')

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.hide_srv = rospy.Service('/shell_game/hide_ball', Trigger, self.handle_hide_request)
        self.shuffle_srv = rospy.Service('/shell_game/toggle_shuffle', SetBool, self.handle_shuffle_request)
        
        # Global arena baseline positioning layout (Meters relative to robot base)
        self.x_pos = 0.65
        self.table_z = 0.8

        self.SWAP_RADIUS = 0.16
        self.SWAP_SPEED = 75.0

        self.current_slots = ["green_cup_1", "green_cup_2", "green_cup_3"]
        self.slot_y_coords = [-0.2, 0.0, 0.2]

        self.cup_poses = {
            "green_cup_1": {'x': self.x_pos, 'y': -0.2, 'z': self.table_z},
            "green_cup_2": {'x': self.x_pos, 'y': 0.0,  'z': self.table_z},
            "green_cup_3": {'x': self.x_pos, 'y': 0.2,  'z': self.table_z}
        }
        self.ball = {'x': self.x_pos, 'y': -0.12, 'z': 0.77}
        
        self.current_phase = "IDLE"
        self.shuffle_frame = 0
        self.swap_phase = 0

        print("--- GAZEBO GAME CONTROLLER RUNNING (V3: ANTI-COLLISION & CONTINUOUS VELOCITY) ---")
        print(f"Active Parameters -> Clearance Arc: {self.SWAP_RADIUS}m | Cycle Duration: {self.SWAP_SPEED} frames\n")

    def handle_hide_request(self, req):
        if self.current_phase == "IDLE":
            self.current_phase = "HIDE"
            return TriggerResponse(success=True, message="Starting Hide Phase Animation")
        return TriggerResponse(success=False, message=f"Cannot hide while in phase: {self.current_phase}")

    def handle_shuffle_request(self, req):
        if req.data:
            self.current_phase = "SHUFFLE"
            self.shuffle_frame = 0
            self.swap_phase = np.random.randint(0, 3) 
            msg = "Shuffle Started"
        else:
            self.current_phase = "STOP"
            msg = "Shuffle Stopped"
            self.print_winning_cup_location()
        return SetBoolResponse(success=True, message=msg)

    def execute_hide_animation(self):
        """Animates lifting the cup and sliding the ball underneath over 120 cycles."""
        for frame in range(120):
            if rospy.is_shutdown() or self.current_phase != "HIDE":
                break
                
            if frame < 40:
                self.cup_poses["green_cup_1"]['z'] = self.table_z + (0.15 * (frame / 40.0))
            elif frame < 80:
                progress = (frame - 40) / 40.0
                self.ball['y'] = -0.12 - (0.08 * progress)
            else:
                progress = (frame - 80) / 40.0
                self.cup_poses["green_cup_1"]['z'] = (self.table_z + 0.15) - (0.15 * progress)
                self.ball['y'] = -0.2
                
            self.publish_states()
            rospy.sleep(0.03)
            
        if self.current_phase == "HIDE":
            self.current_phase = "IDLE"
            print("[GAME INFO] Hiding animation finished. The ball is under green_cup_1.")

    def run_shuffle_step(self):
        """Swaps cups using dynamic s-curve interpolation and wide separation trajectories."""
        self.shuffle_frame += 1
        
        raw_progress = min(self.shuffle_frame / self.SWAP_SPEED, 1.0)

        smooth_progress = (1.0 - np.cos(raw_progress * np.pi)) / 2.0

        if self.swap_phase == 0:
            idxA, idxB = 0, 1
        elif self.swap_phase == 1:
            idxA, idxB = 1, 2
        else:
            idxA, idxB = 0, 2

        cupA_name = self.current_slots[idxA]
        cupB_name = self.current_slots[idxB]
        
        start_yA = self.slot_y_coords[idxA]
        start_yB = self.slot_y_coords[idxB]

        current_yA = start_yA + (start_yB - start_yA) * smooth_progress
        current_yB = start_yB + (start_yA - start_yB) * smooth_progress

        arc_offset = self.SWAP_RADIUS * np.sin(raw_progress * np.pi)

        self.cup_poses[cupA_name]['x'] = self.x_pos + arc_offset
        self.cup_poses[cupA_name]['y'] = current_yA
        
        self.cup_poses[cupB_name]['x'] = self.x_pos - arc_offset
        self.cup_poses[cupB_name]['y'] = current_yB

        for i in range(3):
            if i != idxA and i != idxB:
                static_cup_name = self.current_slots[i]
                self.cup_poses[static_cup_name]['x'] = self.x_pos
                self.cup_poses[static_cup_name]['y'] = self.slot_y_coords[i]

        self.ball['x'] = self.cup_poses["green_cup_1"]['x']
        self.ball['y'] = self.cup_poses["green_cup_1"]['y']
        self.ball['z'] = self.table_z - 0.03

        self.publish_states()

        if raw_progress >= 1.0:
            self.current_slots[idxA] = cupB_name
            self.current_slots[idxB] = cupA_name
            
            self.shuffle_frame = 0
            self.swap_phase = np.random.randint(0, 3)  # Randomized vector flow selection

            self.print_winning_cup_location()

    def print_winning_cup_location(self):
        """Locates green_cup_1 inside the slot tracking arrays and prints its position."""
        try:
            winning_slot_idx = self.current_slots.index("green_cup_1")
            slot_names = ["LEFT SLOT", "CENTER SLOT", "RIGHT SLOT"]
            print(f"[LIVE GAME TRACKER] Target (green_cup_1) is currently in the: {slot_names[winning_slot_idx]}")
        except ValueError:
            pass

    def publish_states(self):
        """Packages coordinate maps into ModelState messages and shoots them to Gazebo."""
        for name, pose in self.cup_poses.items():
            msg = ModelState(model_name=name)
            msg.pose.position.x = pose['x']
            msg.pose.position.y = pose['y']
            msg.pose.position.z = pose['z']
            msg.pose.orientation.w = 1.0
            self.set_state(msg)
            
        b_msg = ModelState(model_name="orange_ball")
        b_msg.pose.position.x = self.ball['x']
        b_msg.pose.position.y = self.ball['y']
        b_msg.pose.position.z = self.ball['z']
        b_msg.pose.orientation.w = 1.0
        self.set_state(b_msg)

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.current_phase == "HIDE":
                self.execute_hide_animation()
            elif self.current_phase == "SHUFFLE":
                self.run_shuffle_step()
            elif self.current_phase == "STOP":
                for i in range(3):
                    name = self.current_slots[i]
                    self.cup_poses[name]['x'] = self.x_pos
                    self.cup_poses[name]['y'] = self.slot_y_coords[i]
                self.publish_states()
                self.current_phase = "IDLE"
            rate.sleep()

if __name__ == '__main__':
    controller = GazeboGameController()
    try:
        controller.spin()
    except rospy.ROSInterruptException:
        pass