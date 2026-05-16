#!/usr/bin/env python3
import os
import argparse
import rospy
import numpy as np
import yaml
import cv2
from duckietown.dtros import DTROS, NodeType # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage # pyright: ignore[reportMissingImports]
from cv_bridge import CvBridge # pyright: ignore[reportMissingImports]
from rl_package.debug_bot import send_live_camera

class VisualDebugNode(DTROS):
    def __init__(self, node_name):
        super(VisualDebugNode, self).__init__(node_name=node_name, node_type=NodeType.DIAGNOSTICS)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckie1nav')
        self.bridge = CvBridge()

        self.obs_shape = (160, 120)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.tilt_strength = 0.0006
        self.img_width = 640        # Defaults, will be updated by calib
        self.img_height = 480

        rospy.loginfo(f"Streaming Sim2Real Debug View from {self.veh}...")
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed", 
            CompressedImage, 
            self.callback, 
            queue_size=1, 
            buff_size=2**24
        )

        self.map_x, self.map_y = self._load_calibration()

    def _compute_tilt_homography(self):
        """
        Builds a homography that simulates tilting the camera downward.
        """
        W, H = self.img_width, self.img_height
        cx = W / 2
        shift = self.tilt_strength * W * H
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([
            [cx - (cx - 0)     * (1 - self.tilt_strength * H), shift],
            [cx + (W - cx)     * (1 - self.tilt_strength * H), shift],
            [W, H],
            [0, H]
        ])
        return cv2.getPerspectiveTransform(src, dst)
    
    def _load_calibration(self):
        """Loads intrinsic parameters and prepares cv2 maps."""
        calib_path = f"/data/config/calibrations/camera_intrinsic/{self.veh}.yaml"
        
        with open(calib_path, 'r') as f:
            calib_data = yaml.safe_load(f)
            
        intrinsics = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
        distortion = np.array(calib_data['distortion_coefficients']['data'])
        img_width = calib_data['image_width']
        img_height = calib_data['image_height']

        #zoom_factor = 2.2

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, (img_width, img_height), 0, (img_width, img_height)
        )
        # Manually scale the focal lengths (fx, fy) to zoom in
        #new_camera_matrix[0, 0] *= zoom_factor
        #new_camera_matrix[1, 1] *= zoom_factor
        # Move the "eye" down
        #new_camera_matrix[1, 2] += 40

        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics, distortion, None, new_camera_matrix, (img_width, img_height), cv2.CV_32FC1
        )
        self.H_tilt = self._compute_tilt_homography()
        

        return map_x, map_y
    
    def preprocess_debug(self, obs_bgr):
        rectified = cv2.remap(obs_bgr, self.map_x, self.map_y, cv2.INTER_LINEAR)
        warped = cv2.warpPerspective(rectified, self.H_tilt, (self.img_width, self.img_height))
        #yuv = cv2.cvtColor(warped, cv2.COLOR_BGR2YUV)
        #yuv[:, :, 0] = self.clahe.apply(yuv[:, :, 0])
        #warped = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        small = cv2.resize(warped, self.obs_shape, interpolation=cv2.INTER_LINEAR)
        
        h, w = small.shape[:2]
        
        # 4. Crop lines for visualization
        top = int(h * (5/12))
        left = int(w * 0.20)
        right = int(w * 0.80)
        
        cv2.line(small, (0, top),   (w, top),   (255, 255, 255), 2)
        cv2.line(small, (left, 0),  (left, h),  (0, 255, 0),     2)
        cv2.line(small, (right, 0), (right, h), (0, 255, 0),     2)
            
        img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return small.transpose(2, 0, 1)
        

    def callback(self, msg):
        obs = self.bridge.compressed_imgmsg_to_cv2(msg)
        
        debug_frame = self.preprocess_debug(obs)
        
        send_live_camera(observation=debug_frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Duckiebot Visual Debug Node")
    args, unknown = parser.parse_known_args()

    node = VisualDebugNode(node_name='visual_debug_node')
    rospy.spin()