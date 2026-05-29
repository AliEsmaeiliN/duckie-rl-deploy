#!/usr/bin/env python3
import os
import sys
import time
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage # pyright: ignore[reportMissingImports]

from rl_package.agent import DuckiebotAgent # pyright: ignore[reportMissingImports]

class HardwareBenchmarkNode(DTROS):
    def __init__(self, node_name):
        super(HardwareBenchmarkNode, self).__init__(node_name=node_name, node_type=NodeType.DIAGNOSTICS)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot98')

        device_arg = "cpu"
        if "--device" in sys.argv:
            idx = sys.argv.index("--device")
            device_arg = sys.argv[idx + 1]

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

        self.run_isolated_dry_run(iterations=50)

        rospy.loginfo(f"Waiting for live camera feed from /{self.veh}...")
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed", 
            CompressedImage, 
            self.callback, 
            queue_size=1, 
            buff_size=2**24
        )

    def run_isolated_dry_run(self, iterations):
        """
        Runs an isolated synthetic test to catch timing bottlenecks 
        across specific pipeline components.
        """
        dummy_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print(f"\n" + "="*50)
        print(f"📦 RUNNING DRY-RUN BENCHMARK ON {str(self.agent.device).upper()} 📦")
        print("="*50)
        
        print("Warming up models and memory buffers...")
        for _ in range(15):
            proc = self.agent.preprocess_cv(dummy_obs)
            self.agent.update_buffer(proc)
            if len(self.agent.frames) == self.agent.frame_stack:
                act = self.agent.get_action()
                _ = self.agent.postprocess_kinematics(act)
                
        print(f"Running {iterations} synthetic pipeline iterations...\n")
        pre_times, nn_times, kinematics_times, total_times = [], [], [], []

        for _ in range(iterations):
            t_start_total = time.perf_counter()
            
            # Step A: Vision Pipeline (Remap -> Crop -> Resize -> Grayscale)
            t0 = time.perf_counter()
            processed_frame = self.agent.preprocess_cv(dummy_obs)
            self.agent.update_buffer(processed_frame)
            t_pre = time.perf_counter() - t0
            
            if len(self.agent.frames) == self.agent.frame_stack:
                # Step B: Neural Network Inference Pass
                t1 = time.perf_counter()
                action = self.agent.get_action()
                t_nn = time.perf_counter() - t1
                
                # Step C: Inverse Kinematics Conversion
                t2 = time.perf_counter()
                _ = self.agent.postprocess_kinematics(action)
                t_kin = time.perf_counter() - t2
                
                t_total = time.perf_counter() - t_start_total
                
                pre_times.append(t_pre * 1000)
                nn_times.append(t_nn * 1000)
                kinematics_times.append(t_kin * 1000)
                total_times.append(t_total * 1000)

        print("--- Isolated Profiler Component Breakdowns ---")
        print(f"Avg Pre-processing: {np.mean(pre_times):.2f} ms")
        print(f"Avg NN Inference:   {np.mean(nn_times):.2f} ms")
        print(f"Avg Kinematics:     {np.mean(kinematics_times):.2f} ms")
        print(f"Avg Total Pipeline: {np.mean(total_times):.2f} ms")
        
        max_freq = 1000 / np.mean(total_times)
        print(f"\nTheoretical Max Freq: {max_freq:.2f} Hz")
        
        budget = (1 / 7.5) * 1000
        print(f"Target Freq (rl_node): 7.50 Hz (Time Budget: {budget:.2f} ms)")
        
        if max_freq > 7.5:
            print("Status: ✅ HEALTHY (Pipeline is comfortably within budget)")
        else:
            print("Status: ❌ BOTTLENECK (Pipeline is too slow for 7.5Hz target)")
        print("="*50 + "\n")

    def callback(self, msg):
        if self.msg_count >= self.target_iterations:
            self.print_stats()
            rospy.signal_shutdown("Benchmark Complete")
            return

        capture_time = msg.header.stamp.to_sec()
        receive_time = rospy.Time.now().to_sec()
        cam_delay = (receive_time - capture_time) * 1000 # Convert to ms
        
        t_start = time.perf_counter()
        
        try:
            # Clean direct native numpy buffer decompressor bypasses CvBridge
            np_arr = np.frombuffer(msg.data, np.uint8)
            obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            proc = self.agent.preprocess_cv(obs)
            self.agent.update_buffer(proc)
            
            if len(self.agent.frames) == self.agent.frame_stack:
                action = self.agent.get_action()
                _ = self.agent.postprocess_kinematics(action)
                
                t_end = time.perf_counter()
                compute_time = (t_end - t_start) * 1000 # ms
                
                self.camera_delays.append(cam_delay)
                self.compute_times.append(compute_time)
                self.total_pipeline_times.append(cam_delay + compute_time)
                self.msg_count += 1
                
                if self.msg_count % 20 == 0:
                    rospy.loginfo(f"Processed {self.msg_count}/{self.target_iterations} live frames...")
                    
        except Exception as e:
            rospy.logerr(f"Live benchmark run exception: {e}")

    def print_stats(self):
        print("\n" + "="*45)
        print("🚀 LIVE GLASS-TO-ACTION HARDWARE TELEMETRY 🚀")
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