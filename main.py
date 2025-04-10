# main.py
import numpy as np
import cv2
from neurapy.robot import Robot
from object_detector import ObjectDetector
from robot_controller import RobotController

def main():
    # Initialize detector & robot
    r=Robot()
    r.set_mode("Automatic")
    r.move_joint("New_capture")
    calibration_matrix = r"C:\Users\HeraldSuriaraj\Documents\neurapy-windows-v4.20.0\neurapy-windows-v4.20.0\cam_to_tcp_transform.npy"
    detector=ObjectDetector(calibration_matrix)
    robot_control=RobotController()
    

    try:
        while True:
            detection=detector.get_detection()
            if detection is None:
                print("[DEBUG] No object detected.")
                continue

            # clearly print detected position and orientation
            print("[DETECTED] Camera XYZ:", detection["position_camera"],"Angle deg:",detection["orientation_deg"])

            # Obtain current TCP pose
            tcp_pose_current=robot_control.robot.get_tcp_pose()
            T_tcp_to_base=robot_control.pose_to_matrix(tcp_pose_current,gripper_offset_z=-0.105)

            pos_cam_hom=np.array([*detection["position_camera"],1])
            base_coords=T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom

            yaw_rad=np.deg2rad(detection["orientation_deg"])

            target_pose=[base_coords[0],base_coords[1],base_coords[2],0,np.pi,yaw_rad]
            print("[TARGET POSE]",target_pose)

            robot_control.move_to_pose(target_pose,speed=0.2)

            img=detection["color_image"]
            cv2.circle(img, detection["pixel"],5,(0,255,0),-1)
            cv2.imshow("Detection",img)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break

    except Exception as ex:
        print("[ERROR]",ex)
    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()