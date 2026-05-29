#!/usr/bin/env python3
import os
import sys
import time
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType  # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage  # pyright: ignore[reportMissingImports]

# Import the model & agent layout safely from your sibling package!
from rl_package.agent import DuckiebotAgent # pyright: ignore[reportMissingImports]

class HardwareBenchmarkNode(DTROS):
    def __init__(self, node_name):
        super(HardwareBenchmarkNode, self).__init__(node_name=node_name, node_type=NodeType.DIAGNOSTICS)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot98')

        device_arg = "cpu"
        if "--device" in sys.argv:
            idx = sys.argv.index("--device")
            device_arg = sys.argv[idx + 1]

        # Standard baseline configuration path
        repo_path = os.environ.get("DT_REPO_PATH", "/code/catkin_ws/src/duckie-rl-deploy")
        model_full_path = os.path.join(repo_path, "assets/models/sac_v10.cleanrl_model")
        
        self.agent = DuckiebotAgent(
            model_path=model_full_path, 
            algo_type="sac",
            device=device_arg
        )
        
        self.camera_delays = []
        self.compute_times = []
        self.total_pipeline_times = []
        self.msg_count = 0
        self.target_iterations = 100

        rospy.loginfo(f"Waiting for live camera feed from /{self.veh}...")
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed", 
            CompressedImage, 
            self.callback, 
            queue_size=1, 
            buff_size=2**24
        )

    def callback(self, msg):
        if self.msg_count >= self.target_iterations:
            self.print_stats()
            rospy.signal_shutdown("Benchmark Complete")
            return

        capture_time = msg.header.stamp.to_sec()
        receive_time = rospy.Time.now().to_sec()
        cam_delay = (receive_time - capture_time) * 1000 
        
        t_start = time.perf_counter()
        
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            proc = self.agent.preprocess_cv(obs)
            self.agent.update_buffer(proc)
            
            if len(self.agent.frames) == self.agent.frame_stack:
                action = self.agent.get_action()
                _ = self.agent.postprocess_kinematics(action)
                
                t_end = time.perf_counter()
                compute_time = (t_end - t_start) * 1000 
                
                self.camera_delays.append(cam_delay)
                self.compute_times.append(compute_time)
                self.total_pipeline_times.append(cam_delay + compute_time)
                self.msg_count += 1
                
                if self.msg_count % 20 == 0:
                    rospy.loginfo(f"Processed {self.msg_count}/{self.target_iterations} live frames...")
                    
        except Exception as e:
            rospy.logerr(f"Benchmark run exception: {e}")

    def print_stats(self):
        print("\n" + "="*45)
        print("🚀 END-TO-END ROS HARDWARE BENCHMARK 🚀")
        print("="*45)
        print(f"Avg Camera + ROS Network Delay: {np.mean(self.camera_delays):.2f} ms")
        print(f"Avg Decompression + Inference:  {np.mean(self.compute_times):.2f} ms")
        print("-" * 45)
        
        total_mean = np.mean(self.total_pipeline_times)
        print(f"TOTAL 'Glass-to-Action' Latency: {total_mean:.2f} ms")
        print(f"True Maximum Frequency:          {1000/total_mean:.2f} Hz")
        print("="*45 + "\n")

if __name__ == '__main__':
    node = HardwareBenchmarkNode(node_name='hw_benchmark_node')
    rospy.spin()