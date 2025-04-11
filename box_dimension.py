import cv2
import numpy as np
import pyrealsense2 as rs

class Dimension:
    def __init__(self, intrinsics=None, cam_to_gripper_matrix=None):
        self.intrinsics = intrinsics
        self.cam_to_gripper_matrix = cam_to_gripper_matrix if cam_to_gripper_matrix is not None else np.eye(4)

    def calculate_distance_3d(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def is_point_in_frame(self, x, y, width=640, height=480):
        return 0 <= x < width and 0 <= y < height

    def average_depth(self, x, y, depth_frame, window=3):
        depths = []
        for dx in range(-window//2, window//2 + 1):
            for dy in range(-window//2, window//2 + 1):
                nx, ny = x + dx, y + dy
                if self.is_point_in_frame(nx, ny):
                    d = depth_frame.get_distance(nx, ny)
                    if d > 0:
                        depths.append(d)
        return np.mean(depths) if depths else 0

    def pixel_to_camera(self, x, y, depth_frame):
        depth = self.average_depth(x, y, depth_frame)
        if depth == 0:
            return None
        if self.intrinsics is None:
            self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
        return np.array(point_3d)

    def transform_to_gripper(self, camera_coords):
        """Apply transformation from camera to robot gripper frame"""
        camera_coords_h = np.append(camera_coords, 1.0)
        gripper_coords = np.dot(self.cam_to_gripper_matrix, camera_coords_h)
        return gripper_coords[:3]

    def calculate_object_dimensions(self, box_points, depth_frame):
        """Calculate object width and height using 3D distances between bounding box corners"""
        points_3d = []
        for point in box_points:
            x, y = int(point[0]), int(point[1])
            if not self.is_point_in_frame(x, y):
                continue
            camera_coords = self.pixel_to_camera(x, y, depth_frame)
            if camera_coords is not None:
                points_3d.append(camera_coords[:3])

        if len(points_3d) < 4:
            return None, None

        # Use distances between consecutive corners
        edge1 = self.calculate_distance_3d(points_3d[0], points_3d[1])
        edge2 = self.calculate_distance_3d(points_3d[1], points_3d[2])
        width = min(edge1, edge2)
        height = max(edge1, edge2)
        return width, height

    def process_frames(self, color_image, depth_image, depth_frame):
        original = color_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('Detected Binary', binary)

        detected_objects = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                center_x, center_y = np.array(rect[0], dtype=np.int32)

                if not self.is_point_in_frame(center_x, center_y, color_image.shape[1], color_image.shape[0]):
                    continue

                camera_coords = self.pixel_to_camera(center_x, center_y, depth_frame)
                if camera_coords is None:
                    continue

                gripper_coords = self.transform_to_gripper(camera_coords)
                width, height = self.calculate_object_dimensions(box, depth_frame)
                box_reshaped = box.reshape((-1, 1, 2))

                if width is None or height is None:
                    cv2.polylines(original, [box_reshaped], True, (0, 255, 255), 2)
                    continue

                detected_objects.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'gripper_x': gripper_coords[0],
                    'gripper_y': gripper_coords[1],
                    'gripper_z': gripper_coords[2],
                    'width': width * 100,
                    'height': height * 100
                })

                cv2.polylines(original, [box_reshaped], True, (0, 255, 0), 2)
                cv2.circle(original, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(original, f"Pos: ({gripper_coords[0]:.2f}, {gripper_coords[1]:.2f}, {gripper_coords[2]:.2f})",
                            (center_x - 100, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(original, f"Size: {width*100:.1f} x {height*100:.1f} cm",
                            (center_x - 100, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return original, detected_objects

    def release(self):
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            self.pipeline.stop()
