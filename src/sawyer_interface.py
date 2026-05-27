import intera_interface
import rospy
import numpy as np


class SawyerInterface:
    def __init__(self, workspace_width_m=0.4, workspace_height_m=0.4):
        """
        Handles coordinates and control for the Sawyer arm.
        workspace_width_m / height_m: The real-world dimensions (meters) of your 500x500 frame.
        """
        # Initialize the limb interface
        self.limb = intera_interface.Limb('right')

        # Dimensions of your warped virtual table
        self.pixel_width = 500.0
        self.pixel_height = 500.0

        # Scale factors: meters per pixel
        self.scale_x = workspace_width_m / self.pixel_width
        self.scale_y = workspace_height_m / self.pixel_height

        # The physical (X, Y, Z) position of top-left workspace pixel (0,0)
        # relative to Sawyer's base frame. Update these based on measurements!
        self.origin_x = 0.45  # meters in front of Sawyer
        self.origin_y = -0.20  # meters to Sawyer's right/left
        self.origin_z = -0.10  # table surface height

        # Visual Servoing Parameters
        self.Kp = 0.0004  # Proportional gain (meters per pixel of error)
        self.cam_center_x = 320
        self.cam_center_y = 240

        # Hardcoded static neutral orientation orientation: tool facing straight down.

        self.fixed_orientation = [0.7071, 0.7071, 0.0, 0.0]

    def pixel_to_sawyer_base(self, pixel_x, pixel_y):
        """Transforms a bird's-eye view pixel to Sawyer's base coordinate frame."""
        # Calculate offset within the workspace area
        work_x = pixel_x * self.scale_x
        work_y = pixel_y * self.scale_y

        # Map workspace 2D coordinates to Sawyer's 3D Base Frame
        # Adjusted signs assuming standard setup. Switch signs if axes act inverted.
        sawyer_x = self.origin_x + work_x
        sawyer_y = self.origin_y + work_y

        return sawyer_x, sawyer_y

    def stop(self):
        """Safely stops active movement by commanding zero velocity to joints."""
        print("[Sawyer Safety] Halting arm movement.")
        self.limb.set_joint_velocities({name: 0.0 for name in self.limb.joint_names()})

    def go_to_hover(self, pixel_x, pixel_y):
        """Phase 1: Moves smoothly to hover 15cm directly above the designated target pixel."""
        target_x, target_y = self.pixel_to_sawyer_base(pixel_x, pixel_y)
        hover_z = self.origin_z + 0.15  # 15 cm above the table

        print(f"[Sawyer] Moving to Hover Position: X={target_x:.3f}, Y={target_y:.3f}, Z={hover_z:.3f}")

        target_pose = {
            'position': (target_x, target_y, hover_z),
            'orientation': tuple(self.fixed_orientation)
        }

        # Resolve Inverse Kinematics and execute move
        joint_angles = self.limb.ik_request(target_pose)
        if joint_angles:
            self.limb.move_to_joint_positions(joint_angles)
            return True
        else:
            print("[Sawyer Error] Hover target unreachable by IK solver.")
            return False

    def retract(self):
        """Moves the arm straight up to a safe altitude to clear the workspace."""
        current_pose = self.limb.endpoint_pose()
        pos = current_pose['position']
        safe_z = self.origin_z + 0.30  # 30 cm safe clear height

        print(f"[Sawyer] Retracting arm to safe altitude Z={safe_z:.3f}")

        target_pose = {
            'position': (pos.x, pos.y, safe_z),
            'orientation': tuple(self.fixed_orientation)
        }

        joint_angles = self.limb.ik_request(target_pose)
        if joint_angles:
            self.limb.move_to_joint_positions(joint_angles)
            return True
        return False

    def visual_servoing_step(self, cup_wrist_pixel):
        """
        Phase 2 Step: Takes the cup center seen from the WRIST camera
        and updates joint angles to center the arm while descending.
        """
        if cup_wrist_pixel is None:
            self.stop()
            return False  # Stop descending if wrist camera loses the cup

        u, v = cup_wrist_pixel

        # Calculate position error relative to camera center
        error_u = u - self.cam_center_x
        error_v = v - self.cam_center_y

        # Map pixel errors to tool frame translation deltas
        delta_x = error_v * self.Kp
        delta_y = -error_u * self.Kp
        delta_z = -0.005  # Descend exactly 5mm per control iteration

        current_pose = self.limb.endpoint_pose()
        pos = current_pose['position']

        # Compute absolute target step enforcing the strict top-down orientation
        target_pose = {
            'position': (pos.x + delta_x, pos.y + delta_y, pos.z + delta_z),
            'orientation': tuple(self.fixed_orientation)
        }

        # Execute small rapid step using non-blocking position command
        joint_angles = self.limb.ik_request(target_pose)
        if joint_angles:
            self.limb.set_joint_positions(joint_angles)

            # Check if we have arrived close to the table surface target
            if pos.z <= (self.origin_z + 0.02):  # Stop 2cm above table
                print("[Sawyer] Reached target cup successfully!")
                return True
        return False