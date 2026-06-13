# sawyer_interface.py — full rewrite of the motion parts for Gazebo

import rospy
import numpy as np
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion
from intera_interface import Limb
import sys

class SawyerInterface:
    def __init__(self, workspace_width_m=0.5, workspace_height_m=0.5):
            from sensor_msgs.msg import JointState

            print("[SawyerInterface] Verifying clock synchronization with Gazebo...")
            while rospy.get_time() == 0.0:
                rospy.sleep(0.1)
            print(f"[SawyerInterface] Clock synchronized! Active Sim Time: {rospy.get_time()}")

            print("[SawyerInterface] Waiting for live joint_states topic to hydrate cache...")
            try:
                rospy.wait_for_message("/robot/joint_states", JointState, timeout=5.0)
                print("[SawyerInterface] Joint states received and synchronized!")
            except rospy.ROSException:
                print("[WARN] Timing out waiting for /robot/joint_states. Trying fallback global topic...")
                try:
                    rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
                except rospy.ROSException:
                    print("[ERROR] No joint state topics are broadcasting! Check your Action Server terminal.")

            self.pixel_width  = 500.0
            self.pixel_height = 500.0
            self.scale_x = workspace_width_m  / self.pixel_width
            self.scale_y = workspace_height_m / self.pixel_height

            # Your Gazebo world calibration
            self.origin_x = 0.40
            self.origin_y = -0.25
            self.origin_z = 0.75   # table surface
            self.HOVER_Z = self.origin_z - 0.30 

            # --- MoveIt init (works in Gazebo) ---
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            
            active_ns = rospy.get_namespace()
            print(f"[SawyerInterface] Detected active ROS namespace: '{active_ns}'")
            
            # Pass the detected namespace directly to the commander
            self.group = moveit_commander.MoveGroupCommander(
                name="right_arm", 
                robot_description="robot_description", 
                ns=active_ns
            )
            
            self.group.set_planner_id("RRTConnectkConfigDefault") 
            self.group.set_max_velocity_scaling_factor(0.8)
            self.group.set_max_acceleration_scaling_factor(0.7)
            self.group.set_planning_time(1.5)
            self.group.set_num_planning_attempts(5)
            self.group.set_goal_position_tolerance(0.015)      
            self.group.set_goal_orientation_tolerance(0.05)   
            
            # Enforce fresh state captures immediately
            self.group.set_start_state_to_current_state()
            
            print(f"Available MoveIt Groups: {self.robot.get_group_names()}")
            print(f"Current EE pose: \n{self.group.get_current_pose().pose}")
            print(f"Planning frame: {self.group.get_planning_frame()}")
            print(f"EE link: {self.group.get_end_effector_link()}")
            print("[SawyerInterface] MoveIt commander ready.")

    def pixel_to_world(self, pixel_x, pixel_y):
        """Warped bird's-eye pixel → Sawyer base frame (meters)."""
        # Vertical image axis maps directly to Robot X (forward/backward extension)
        world_x = self.origin_x + (pixel_y * self.scale_x)
        
        # Horizontal image axis maps to Robot Y (left/right translation)
        world_y = self.origin_y + ((self.pixel_width - pixel_x) * self.scale_y)
        return world_x, world_y

    def _build_pose(self, x, y, z):
        p = Pose()
        p.position    = Point(x, y, z)
        p.orientation = self.orientation_down
        return p

    def _plan_and_execute(self, pose):
        self.group.set_pose_target(pose)
        plan = self.group.plan()

        # MoveIt plan() returns (success, plan, time, err) in newer versions
        if isinstance(plan, tuple):
            success, plan, _, _ = plan
        else:
            success = len(plan.joint_trajectory.points) > 0

        if not success:
            print("[MoveIt] Planning failed — target may be out of reach.")
            self.group.clear_pose_targets()
            return False

        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        return True

    def stop(self):
        self.group.stop()
        self.group.clear_pose_targets()

    def go_to_hover(self, pixel_x, pixel_y):
        wx, wy = self.pixel_to_world(pixel_x, pixel_y)
        print(f"[MoveIt] Hovering above cup at world: x={wx:.3f}, y={wy:.3f}, z={self.HOVER_Z:.3f}")


        try:
            native_limb = Limb('right')
            native_pose = native_limb.endpoint_pose()
            
            # Extract live orientations from the native hardware layer
            live_quat = native_pose['orientation']
            print(f"[DEBUG] Verified Live Orientation via Intera SDK: {live_quat}")
            
            pose = Pose()
            pose.position = Point(wx, wy, self.HOVER_Z)
            pose.orientation = Quaternion(
                x=live_quat.x,
                y=live_quat.y,
                z=live_quat.z,
                w=live_quat.w
            )
        except Exception as e:
            print(f"[WARN] Native Limb API lookup failed: {e}. Falling back to default downward orientation.")
            pose = Pose()
            pose.position = Point(wx, wy, self.HOVER_Z)
            pose.orientation = self.orientation_down  # fallback target

        # Seed the trajectory planner directly with the current physical joint values
        self.group.set_start_state_to_current_state()
        print("\n" + "="*50)
        print("COORDINATE DEBUGGING")
        print("="*50)
        print(f"Target Cup Pixel:         ({pixel_x}, {pixel_y})")
        print(f"Calculated World Target:   X={wx:.3f}, Y={wy:.3f}, Z={self.HOVER_Z:.3f}")
        
        # Pull where the robot actually ended up after moving
        native_pose = Limb('right').endpoint_pose()
        actual_pos = native_pose['position']
        print(f"Actual Robot Destination:  X={actual_pos.x:.3f}, Y={actual_pos.y:.3f}, Z={actual_pos.z:.3f}")
        print("="*50 + "\n")
        return self._plan_and_execute(pose)