# robot_controller.py
import numpy as np # import numpy
from neurapy.robot import Robot
from scipy.spatial.transform import Rotation as R # import rotation
import time # import time
class RobotController: # define the class
    def __init__(self, robot_ip=None): # initialize the class
        self.robot = Robot(robot_ip) if robot_ip else Robot()
        self.robot.set_mode("Automatic")

    def pose_to_matrix(self, tcp_pose, gripper_offset_z): # define the function for pose to matrix
        """
        Converts a TCP pose [X,Y,Z,Roll,Pitch,Yaw] into a 4x4 transformation matrix
        and explicitly accounts for gripper offset along TCP's Z-direction.
        
        Args:
            tcp_pose (list): TCP pose as [X,Y,Z,roll,pitch,yaw] 
                            (translation in meters, rotation in radians).
            gripper_offset_z (float): Gripper offset in meters along TCP's Z-axis.
                                        Default is 0.100 m (100mm). Adjust as needed.
                                        
        Returns:
            np.array: 4x4 corrected homogeneous transformation matrix (base to gripper-tip).
        """
        translation = np.array(tcp_pose[:3])  # [X,Y,Z]
        rpy = np.array(tcp_pose[3:])          # [Roll,Pitch,Yaw]

        # Compute rotation matrix from Euler angles
        rotation_matrix = R.from_euler('xyz', rpy).as_matrix()

        # Original TCP transformation (base→TCP)
        T_tcp = np.eye(4)
        T_tcp[:3,:3] = rotation_matrix
        T_tcp[:3,3] = translation

        # Account explicitly for your gripper offset along the TCP's local Z-axis
        T_gripper_offset = np.eye(4)
        T_gripper_offset[:3, 3] = [0, 0, gripper_offset_z]  # Z offset of gripper w.r.t. TCP

        # Final transformation (base→gripper_tip)
        T_final = T_tcp @ T_gripper_offset

        return T_final # return the matrix

    def move_to_pose(self, pose_xyzrpy, speed=0.5): # define the function for move to pose
        
        # Move to the desired pose
        linear_property = {
            "speed": speed,
            "acceleration": 0.1,
            "jerk": 100,
            "rotation_speed": 1.57,
            "rotation_acceleration": 5.0,
            "rotation_jerk": 100,
            "blending": False,
            "target_pose": [pose_xyzrpy],
            "current_joint_angles": self.robot.get_current_joint_angles(),
            "weaving": False,
            "pattern": 1,
            "amplitude_left": 0.003,
            "amplitude_right": 0.003,
            "frequency": 1.5,
            "dwell_time_left": 0.0,
            "dwell_time_right": 0.0,
            "elevation": 0.0,
            "azimuth": 0.0
        }
        self.robot.move_linear_from_current_position(**linear_property) # move to the desired pose
        #time.sleep(2)
        io_set = self.robot.set_tool_digital_outputs([1.0,0.0,0.0]) # set the tool digital outputs
        time.sleep(1) # set the time
        self.robot.move_joint("New_capture") # move to the capture position
        self.robot.move_joint("P30") # move to the P30 position
        io_set = self.robot.set_tool_digital_outputs([0.0,1.0,0.0]) # set the tool digital outputs
        self.robot.move_joint("New_capture") # move to the capture position
        self.robot.stop() # stop the robot

