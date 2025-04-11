# import cv2
# import numpy as np
# import pyrealsense2 as rs

# class Dimension:
#     def __init__(self, intrinsics=None, cam_to_gripper_matrix=None):
#         self.intrinsics = intrinsics
#         self.cam_to_gripper_matrix = cam_to_gripper_matrix if cam_to_gripper_matrix is not None else np.eye(4)

#     def calculate_distance_3d(self, point1, point2):
#         return np.linalg.norm(point1 - point2)

#     def is_point_in_frame(self, x, y, width=640, height=480):
#         return 0 <= x < width and 0 <= y < height

#     def average_depth(self, x, y, depth_frame, window=3):
#         depths = []
#         for dx in range(-window//2, window//2 + 1):
#             for dy in range(-window//2, window//2 + 1):
#                 nx, ny = x + dx, y + dy
#                 if self.is_point_in_frame(nx, ny):
#                     d = depth_frame.get_distance(nx, ny)
#                     if d > 0:
#                         depths.append(d)
#         return np.mean(depths) if depths else 0

#     def pixel_to_camera(self, x, y, depth_frame):
#         depth = self.average_depth(x, y, depth_frame)
#         if depth == 0:
#             return None
#         if self.intrinsics is None:
#             self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
#         point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
#         return np.array(point_3d)

#     def transform_to_gripper(self, camera_coords):
#         """Apply transformation from camera to robot gripper frame"""
#         camera_coords_h = np.append(camera_coords, 1.0)
#         gripper_coords = np.dot(self.cam_to_gripper_matrix, camera_coords_h)
#         return gripper_coords[:3]

#     def calculate_object_dimensions(self, box_points, depth_frame):
#         """Calculate object width and height using 3D distances between bounding box corners"""
#         points_3d = []
#         for point in box_points:
#             x, y = int(point[0]), int(point[1])
#             if not self.is_point_in_frame(x, y):
#                 continue
#             camera_coords = self.pixel_to_camera(x, y, depth_frame)
#             if camera_coords is not None:
#                 points_3d.append(camera_coords[:3])

#         if len(points_3d) < 4:
#             return None, None

#         # Use distances between consecutive corners
#         edge1 = self.calculate_distance_3d(points_3d[0], points_3d[1])
#         edge2 = self.calculate_distance_3d(points_3d[1], points_3d[2])
#         width = min(edge1, edge2)
#         height = max(edge1, edge2)
#         return width, height

#     def process_frames(self, color_image, depth_image, depth_frame):
#         original = color_image.copy()
#         gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.imshow('Detected Binary', binary)

#         detected_objects = []

#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area > 1000:
#                 rect = cv2.minAreaRect(contour)
#                 box = cv2.boxPoints(rect)
#                 box = np.array(box, dtype=np.int32)
#                 center_x, center_y = np.array(rect[0], dtype=np.int32)

#                 if not self.is_point_in_frame(center_x, center_y, color_image.shape[1], color_image.shape[0]):
#                     continue

#                 camera_coords = self.pixel_to_camera(center_x, center_y, depth_frame)
#                 if camera_coords is None:
#                     continue

#                 gripper_coords = self.transform_to_gripper(camera_coords)
#                 width, height = self.calculate_object_dimensions(box, depth_frame)
#                 box_reshaped = box.reshape((-1, 1, 2))

#                 if width is None or height is None:
#                     cv2.polylines(original, [box_reshaped], True, (0, 255, 255), 2)
#                     continue

#                 detected_objects.append({
#                     'center_x': center_x,
#                     'center_y': center_y,
#                     'gripper_x': gripper_coords[0],
#                     'gripper_y': gripper_coords[1],
#                     'gripper_z': gripper_coords[2],
#                     'width': width * 100,
#                     'height': height * 100
#                 })

#                 cv2.polylines(original, [box_reshaped], True, (0, 255, 0), 2)
#                 cv2.circle(original, (center_x, center_y), 5, (0, 0, 255), -1)
#                 cv2.putText(original, f"Pos: ({gripper_coords[0]:.2f}, {gripper_coords[1]:.2f}, {gripper_coords[2]:.2f})",
#                             (center_x - 100, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                 cv2.putText(original, f"Size: {width*100:.1f} x {height*100:.1f} cm",
#                             (center_x - 100, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         return original, detected_objects

#     def release(self):
#         if hasattr(self, 'pipeline') and self.pipeline is not None:
#             self.pipeline.stop()


# object_detector.py
import numpy as np # importing numpy
import cv2 # importing cv2
import pyrealsense2 as rs # importing pyrealsense
from typing import Optional, Dict, Any, Tuple
import logging

class ObjectDetector:
    def __init__(
        self,
        calibration_matrix_path: str,
        min_contour_area: int = 1000,
        threshold_method: str = 'otsu',
        color_stream_config: Tuple[int, int, int] = (1280, 720, 30),
        depth_stream_config: Tuple[int, int, int] = (1280, 720, 30),
        contour_validation: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ObjectDetector with configuration parameters.
        
        Args:
            calibration_matrix_path: Path to camera-to-TCP calibration matrix
            min_contour_area: Minimum area for a contour to be considered valid
            threshold_method: Thresholding method ('otsu' or 'adaptive')
            color_stream_config: (width, height, fps) for color stream
            depth_stream_config: (width, height, fps) for depth stream
            contour_validation: Parameters for contour validation (aspect_ratio, solidity, etc.)
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load calibration matrix
        try:
            self.T_cam_to_tcp = np.load(calibration_matrix_path)
            self.logger.info("Successfully loaded calibration matrix")
        except Exception as e:
            self.logger.error(f"Failed to load calibration matrix: {str(e)}")
            raise

        # Store configuration parameters
        self.min_contour_area = min_contour_area
        self.threshold_method = threshold_method
        self.contour_validation = contour_validation or {
            'min_aspect_ratio': 0.2,
            'max_aspect_ratio': 5.0,
            'min_solidity': 0.8
        }

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        
        try:
            cfg.enable_stream(
                rs.stream.color, 
                color_stream_config[0], 
                color_stream_config[1], 
                rs.format.bgr8, 
                color_stream_config[2]
            )
            cfg.enable_stream(
                rs.stream.depth,
                depth_stream_config[0],
                depth_stream_config[1],
                rs.format.z16,
                depth_stream_config[2]
            )
            self.profile = self.pipeline.start(cfg)
            self.logger.info("Pipeline started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {str(e)}")
            raise

        # Get camera intrinsics and create align object
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)

    def is_valid_contour(self, contour: np.ndarray) -> bool:
        """
        Validate contour based on area, aspect ratio, and other geometric properties.
        
        Args:
            contour: Contour to validate
            
        Returns:
            bool: True if contour is valid, False otherwise
        """
        area = cv2.contourArea(contour)
        if area < self.min_contour_area:
            return False

        # Calculate contour properties
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        
        # Avoid division by zero
        longer = max(width, height)
        shorter = min(width, height)
        if shorter == 0:
            return False
            
        aspect_ratio = longer / shorter
        if not (self.contour_validation['min_aspect_ratio'] <= aspect_ratio <= self.contour_validation['max_aspect_ratio']):
            return False

        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
            
        solidity = float(area) / hull_area
        if solidity < self.contour_validation['min_solidity']:
            return False

        return True

    def apply_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Apply thresholding to grayscale image based on configured method.
        
        Args:
            gray_image: Input grayscale image
            
        Returns:
            Thresholded binary image
        """
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
            
        return binary

    def get_detections(self) -> Optional[Dict[str, Any]]:
        """
        Get all valid detections in the current frame.
        
        Returns:
            Dictionary containing detection information or None if no valid detections
        """
        try:
            # Get frames
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                self.logger.warning("Missing color or depth frame")
                return None

            # Process image
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Noise reduction
            
            # Thresholding
            binary = self.apply_threshold(gray)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process all valid contours
            detections = []
            for contour in contours:
                if not self.is_valid_contour(contour):
                    continue

                # Get bounding rectangle properties
                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect
                center_x, center_y = int(center_x), int(center_y)

                # Get depth at center point
                depth_value = depth_frame.get_distance(center_x, center_y)
                if depth_value <= 0:
                    self.logger.debug("Invalid depth value")
                    continue

                # Convert to 3D coordinates
                xyz_camera = rs.rs2_deproject_pixel_to_point(
                    self.intrinsics,
                    [center_x, center_y],
                    depth_value
                )
                xyz_camera = np.array([*xyz_camera, 1])
                
                # Transform to TCP coordinates
                xyz_tcp = self.T_cam_to_tcp @ xyz_camera

                detections.append({
                    "pixel": (center_x, center_y),
                    "size": (width, height),
                    "position_camera": xyz_camera[:3],
                    "position_tcp": xyz_tcp[:3],
                    "orientation_deg": angle,
                    "contour_area": cv2.contourArea(contour),
                    "color_image": color_image
                })

            if not detections:
                return None
                
            return {
                "detections": detections,
                "color_image": color_image,
                "binary_image": binary
            }

        except Exception as e:
            self.logger.error(f"Error in detection: {str(e)}")
            return None

    def release(self) -> None:
        """Release resources and stop pipeline."""
        try:
            self.pipeline.stop()
            self.logger.info("Pipeline stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {str(e)}")
            raise
