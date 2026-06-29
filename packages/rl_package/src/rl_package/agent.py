import os
import numpy as np
import collections
import cv2
import yaml

import onnxruntime as ort # pyright: ignore[reportMissingImports]

class DuckiebotAgent:
    def __init__(self, model_path, algo_type="sac", grayscale=True, frame_stack=4, device=None):
        print("\n" + "="*50)
        print(" RUNNING ON: ONNX (NVIDIA EDGE EXECUTOR ENGINE) ")
        print("="*50)
            
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.prev_action = np.array([0.0, 0.0])
        self.obs_shape = (160, 120)
        self.alpha = 0.5  # Lower = smoother but more lag
        self.tilt_strength = 0.0006
        self.img_width = 640        # Defaults, will be updated by calib
        self.img_height = 480
        
        self.c = 1 if grayscale else 3
        self.frames = collections.deque(maxlen=frame_stack)

        if model_path.endswith(".engine") or model_path.endswith(".cleanrl_model"):
            model_path = model_path.replace(".engine", ".onnx").replace(".cleanrl_model", ".onnx")

        # Dynamic Algorithm Identification based on filename prefix
        filename = os.path.basename(model_path).lower()
        if filename.startswith("sac"):
            self.algo_type = "sac"
        elif filename.startswith("td3"):
            self.algo_type = "td3"
        else:
            self.algo_type = algo_type.lower()

        print(f"Loading ONNX Model Graph from {model_path}...")

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # Limit to 2GB max allocation safety boundaries
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider' # Fallback safety layer
        ]

        # Initialize the acceleration session handler
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Extract network layer entry keys dynamically
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.veh = os.environ.get("VEHICLE_NAME", "duckiebot98")
        self.map_x, self.map_y = self._load_calibration()
        print(f"ONNX Session successfully bound to GPU Provider: {self.session.get_providers()}")

    def _compute_tilt_homography(self):
        """
        Builds a homography that simulates tilting the camera downward.
        """
        W, H = self.img_width, self.img_height
        cx = W / 2
        shift = self.tilt_strength * W * H
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([
            [cx - (cx - 0) * (1 - self.tilt_strength * H), shift],
            [cx + (W - cx) * (1 - self.tilt_strength * H), shift],
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

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, (img_width, img_height), 0, (img_width, img_height)
        )

        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics, distortion, None, new_camera_matrix, (img_width, img_height), cv2.CV_32FC1
        )
        self.H_tilt = self._compute_tilt_homography()

        self.map_x = cv2.warpPerspective(map_x, self.H_tilt, (img_width, img_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.map_y = cv2.warpPerspective(map_y, self.H_tilt, (img_width, img_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return self.map_x, self.map_y
    
    def preprocess_cv(self, obs_bgr):
        """
        Replicates the Sim2Real vision pipeline.
        """
        img_org = cv2.remap(obs_bgr, self.map_x, self.map_y, cv2.INTER_LINEAR)
        
        h, w = img_org.shape[:2]
        v_crop_frac = 0.4
        top_third = int(h * ((1 - v_crop_frac) * (1/3) + v_crop_frac))
        h_crop_frac = 0.2
        left = int(w * h_crop_frac)
        right = int(w * (1.0 - h_crop_frac))
        img_org = img_org[top_third:h, left:right]

        img_org = cv2.GaussianBlur(img_org, (3, 3), 0)
        img_org = cv2.resize(img_org, (84, 84), interpolation=cv2.INTER_LINEAR)
        
        if self.grayscale:
            img_processed = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            img_processed = img_processed[np.newaxis, :, :]
        else:
            img_processed = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
            img_processed = img_processed.transpose(2, 0, 1)
            
        return img_processed

    def get_action(self):
        stacked_input = np.concatenate(list(self.frames), axis=0)
        input_tensor = np.expand_dims(stacked_input, axis=0)

        raw_outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        current_raw_action = raw_outputs[0].reshape(-1)
        smoothed_action = (self.alpha * current_raw_action) + ((1.0 - self.alpha) * self.prev_action)
        self.prev_action = smoothed_action.copy()
        return smoothed_action

    def postprocess_kinematics(self, action):
        """
        Translates [v, omega] to physical Wheel Commands [u_l, u_r].
        """
        v_scale = 0.8
        omega_scale = 3
        v, omega = action[0] * v_scale, action[1] * omega_scale

        # DB21J physical constants
        radius, wheel_dist, k, gain, trim, limit = 0.0318, 0.102, 27.0, 1.0, -0.05, 1.0

        # Kinematic equations
        u_r = ((v + 0.5 * omega * wheel_dist) / radius) * (gain + trim) / k
        u_l = ((v - 0.5 * omega * wheel_dist) / radius) * (gain - trim) / k

        if np.abs(u_r) > limit or np.abs(u_l) > limit:
            excess = max(np.abs(u_r), np.abs(u_l))
            scale = limit / excess          # uniform scale preserves steering ratio
            u_r *= scale
            u_l *= scale
        
        return np.array([u_l, u_r], dtype=np.float32)
    
    def update_buffer(self, processed_frame):
        """Appends the last frame to the stack"""
        if len(self.frames) == 0:
            for _ in range(self.frame_stack):
                self.frames.append(processed_frame)
        else:
            self.frames.append(processed_frame)