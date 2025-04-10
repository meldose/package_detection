# object_detector.py
import numpy as np # import numpy
import cv2 # import opencv
import pyrealsense2 as rs # import pyrealsense

class ObjectDetector: # create class
    def __init__(self, calibration_matrix_path): # initialize
        self.T_cam_to_tcp = np.load(calibration_matrix_path)
        self.pipeline = rs.pipeline() # setting pipeline for realsense
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280,720,rs.format.bgr8,30)
        cfg.enable_stream(rs.stream.depth,1280,720,rs.format.z16,30)
        self.profile = self.pipeline.start(cfg)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)

    def get_detection(self): # function for detection
        frames = self.pipeline.wait_for_frames() # setting frames
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = max(contours, key=cv2.contourArea, default=None)

        if best_contour is None or cv2.contourArea(best_contour)<1000:
            return None

        rect = cv2.minAreaRect(best_contour)
        (center_x, center_y),_,angle=rect 
        center_x,center_y = int(center_x),int(center_y)

        depth_value = depth_frame.get_distance(center_x,center_y)
        if depth_value<=0:
            return None

        xyz_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics,[center_x,center_y],depth_value)
        xyz_camera = np.array([*xyz_camera,1])

        detection={
            "pixel":(center_x,center_y),
            "position_camera":xyz_camera[:3],
            "orientation_deg":angle,
            "color_image": color_image 
        }
        return detection

    def release(self):
        self.pipeline.stop() # stop pipeline