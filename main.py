# main.py
import numpy as np # import numpy
import cv2 # import opencv
from neurapy.robot import Robot
from object_detector import ObjectDetector # import object_detector
from robot_controller import RobotController # import robot_controller

def main(): # define the main function
    # Initialize detector & robot
    r=Robot()
    r.set_mode("Automatic") # setting the robot to automatic
    r.move_joint("New_capture") # moving the robot to the capture position
    calibration_matrix = r"C:\Users\HeraldSuriaraj\Documents\neurapy-windows-v4.20.0\neurapy-windows-v4.20.0\cam_to_tcp_transform.npy"
    detector=ObjectDetector(calibration_matrix) # setting the detector
    robot_control=RobotController() # setting the robot
    

    try:
        while True: # giving while condition
            detection=detector.get_detection() # getting the detection
            if detection is None: # if there is no detection
                print("[DEBUG] No object detected.") # print the statement
                continue # continue the loop

            # clearly print detected position and orientation
            print("[DETECTED] Camera XYZ:", detection["position_camera"],"Angle deg:",detection["orientation_deg"])

            # Obtain current TCP pose
            tcp_pose_current=robot_control.robot.get_tcp_pose()
            T_tcp_to_base=robot_control.pose_to_matrix(tcp_pose_current,gripper_offset_z=-0.105)

            pos_cam_hom=np.array([*detection["position_camera"],1]) # setting the position
            base_coords=T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom # getting the base coordinates

            yaw_rad=np.deg2rad(detection["orientation_deg"])

            target_pose=[base_coords[0],base_coords[1],base_coords[2],0,np.pi,yaw_rad] # setting the target pose
            print("[TARGET POSE]",target_pose) # print the target pose

            robot_control.move_to_pose(target_pose,speed=0.2) # move the robot to the target pose with speed of 0.2

            img=detection["color_image"] # setting the image
            cv2.circle(img, detection["pixel"],5,(0,255,0),-1)
            cv2.imshow("Detection",img) # show the image
            if cv2.waitKey(1)&0xFF==ord('q'): # close the window
                break

    except Exception as ex:# catching the exception
        print("[ERROR]",ex) # print the exception
    finally:
        detector.release() # release the detector
        cv2.destroyAllWindows() # close all the windows

if __name__=="__main__":
    main() # calling the main function