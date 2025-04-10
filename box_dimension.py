import cv2
import numpy as np
import pyrealsense2 as rs


class Dimension():   
    def calculate_distance_3d(self, point1, point2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt(np.sum((point1 - point2)**2))
    
    def calculate_object_dimensions(self, box_points, depth_frame):
        """Calculate the real-world dimensions of an object from its bounding box points"""
        # Get 3D points for each corner of the bounding box
        points_3d = []
        for point in box_points:
            x, y = int(point[0]), int(point[1])
            # Skip points outside the frame
            if not self.is_point_in_frame(x, y):
                continue
                
            camera_coords = self.pixel_to_camera(x, y, depth_frame)
            if camera_coords is not None:
                points_3d.append(camera_coords[:3])  # Just take X, Y, Z
        
        if len(points_3d) < 4:
            return None, None  # Not enough valid points
            
        # Calculate the dimensions using the 3D points
        points_3d = np.array(points_3d)
        
        # Sort points to ensure consistent ordering
        # This is a simplified approach - might need adjustment based on object orientation
        sorted_indices = np.lexsort((points_3d[:, 0], points_3d[:, 1]))
        sorted_points = points_3d[sorted_indices]
        
        # Calculate width and height based on sorted points
        # This is simplified and may need adjustment for rotated objects
        if len(sorted_points) >= 4:
            # Calculate diagonal distances
            diag1 = self.calculate_distance_3d(sorted_points[0], sorted_points[3])
            diag2 = self.calculate_distance_3d(sorted_points[1], sorted_points[2])
            
            # Calculate side distances
            width = self.calculate_distance_3d(sorted_points[0], sorted_points[1])
            height = self.calculate_distance_3d(sorted_points[0], sorted_points[2])
            
            return width, height
        
        return None, None

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
                box = np.array(box, dtype=np.int32)  # Use np.int32 instead of np.int0
                center_x, center_y = np.array(rect[0], dtype=np.int32)  # Center in pixels

                # Check if center point is within frame
                if not self.is_point_in_frame(center_x, center_y):
                    continue
                
                # Convert to camera coordinates
                camera_coords = self.pixel_to_camera(center_x, center_y, depth_frame)
                if camera_coords is None:
                    continue  # Skip if invalid depth
                
                # Convert to gripper coordinates
                gripper_coords = self.transform_to_gripper(camera_coords)
                
                # Calculate object dimensions
                width, height = self.calculate_object_dimensions(box, depth_frame)
                if width is None or height is None:
                    # Draw bounding box in yellow if dimensions couldn't be calculated
                    box_reshaped = box.reshape((-1, 1, 2))
                    cv2.polylines(original, [box_reshaped], True, (0, 255, 255), 2)
                    continue
                    
                detected_objects.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'gripper_x': gripper_coords[0],
                    'gripper_y': gripper_coords[1],
                    'gripper_z': gripper_coords[2],
                    'width': width * 100,  # Convert to cm
                    'height': height * 100  # Convert to cm
                })
                
                # Draw bounding box using polylines
                box_reshaped = box.reshape((-1, 1, 2))
                cv2.polylines(original, [box_reshaped], True, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(original, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Display coordinates and dimensions
                cv2.putText(original, f"Position: ({gripper_coords[0]:.2f}, {gripper_coords[1]:.2f}, {gripper_coords[2]:.2f})",
                           (center_x - 100, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(original, f"Size: {width*100:.1f} x {height*100:.1f} cm",
                           (center_x - 100, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return original, detected_objects

    def release(self):
        self.pipeline.stop()
