#!/usr/bin/env python3
import os
import sys
import time
import rospy
import cv2
import numpy as np
import torch # Imported directly for explicit memory tracking layers

from rl_package.agent import DuckiebotAgent # pyright: ignore[reportMissingImports]
from duckietown.dtros import DTROS, NodeType # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage # pyright: ignore[reportMissingImports]

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
        
        # Comprehensive Streaming Metrics Storage Arrays
        self.camera_delays = []
        self.compute_times = []
        self.total_pipeline_times = []
        self.pre_times = []
        self.nn_times = []
        self.kinematics_times = []
        
        self.msg_count = 0
        self.target_iterations = 150 # Increased sample size for better variance tracking
        self.deadline_breaches = 0
        self.time_budget_ms = (1.0 / 7.5) * 1000  # 133.33 ms

        # Run Detailed Stress Diagnostics
        self.run_isolated_dry_run(iterations=100)

        rospy.loginfo(f"Initialization Complete. Opening Live ROS Capture Subscriptions on /{self.veh}...")
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed", 
            CompressedImage, 
            self.callback, 
            queue_size=1, 
            buff_size=2**24
        )

    def get_hardware_memory_info(self):
        """Retrieves raw runtime memory allocation metrics from the active hardware device."""
        if self.agent.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.agent.device) / (1024 ** 2)
            cached = torch.cuda.memory_reserved(self.agent.device) / (1024 ** 2)
            return f"VRAM Allocated: {allocated:.2f}MB | VRAM Reserved: {cached:.2f}MB"
        else:
            # CPU memory reporting reads active process status fields
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if 'VmRSS:' in line:
                            return f"Host System RAM RSS: {line.split()[1]} KB"
            except IOError:
                pass
            return "Host System Memory: Profiling Unavailable"

    def run_isolated_dry_run(self, iterations):
        dummy_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\n" + "═"*60)
        print(f"🔬 ADVANCED SYNTHETIC PROFILER RUNNING ON: [{str(self.agent.device).upper()}]")
        print("═"*60)
        print(f"Initial Memory Footprint: {self.get_hardware_memory_info()}")
        
        print("Warming up compute layer tensor paths...")
        for _ in range(20):
            proc = self.agent.preprocess_cv(dummy_obs)
            self.agent.update_buffer(proc)
            if len(self.agent.frames) == self.agent.frame_stack:
                _ = self.agent.postprocess_kinematics(self.agent.get_action())
                
        print(f"Executing {iterations} continuous stress iterations...")
        pre_times, nn_times, kin_times, total_times = [], [], [], []

        for _ in range(iterations):
            t_start = time.perf_counter()
            
            # Sub-component 1: Computer Vision Step
            t0 = time.perf_counter()
            proc = self.agent.preprocess_cv(dummy_obs)
            self.agent.update_buffer(proc)
            dt_pre = (time.perf_counter() - t0) * 1000
            
            if len(self.agent.frames) == self.agent.frame_stack:
                # Sub-component 2: Model Inference Execution
                t1 = time.perf_counter()
                action = self.agent.get_action()
                dt_nn = (time.perf_counter() - t1) * 1000
                
                # Sub-component 3: Kinematics Mapping
                t2 = time.perf_counter()
                _ = self.agent.postprocess_kinematics(action)
                dt_kin = (time.perf_counter() - t2) * 1000
                
                dt_total = (time.perf_counter() - t_start) * 1000
                
                pre_times.append(dt_pre)
                nn_times.append(dt_nn)
                kin_times.append(dt_kin)
                total_times.append(dt_total)

        print("\n📊 ISOLATED COMPONENT BREAKDOWNS (Mean ± StdDev):")
        print(f" └─ Vision Pipeline:  {np.mean(pre_times):6.2f} ms  (± {np.std(pre_times):.2f} ms)")
        print(f" └─ NN Inference Pass: {np.mean(nn_times):6.2f} ms  (± {np.std(nn_times):.2f} ms)")
        print(f" └─ Kinematics Wrap:  {np.mean(kin_times):6.2f} ms  (± {np.std(kin_times):.2f} ms)")
        print(f" └─ Total Loop Time:  {np.mean(total_times):6.2f} ms  (± {np.std(total_times):.2f} ms)")
        
        # Tail-latency checks
        p99 = np.percentile(total_times, 99)
        max_f = 1000.0 / np.mean(total_times)
        
        print("\n📈 PERFORMANCE METRICS SUMMARY:")
        print(f" ├─ Theoretical Maximum Throughput: {max_f:.2f} Hz")
        print(f" ├─ Worst-Case Tail Latency (P99):  {p99:.2f} ms")
        print(f" ├─ Final Stable Memory Footprint:  {self.get_hardware_memory_info()}")
        
        if p99 > self.time_budget_ms:
            print(f" ⚠️  WARNING: P99 Tail Latency ({p99:.2f}ms) exceeds execution budget ({self.time_budget_ms:.2f}ms)!")
            print("    This system could suffer from periodic control drops on high OS loads.")
        else:
            print(" ✅ PROFILE HEALTHY: Core loop variance is safely bound inside target constraints.")
        print("═"*60 + "\n")

    def callback(self, msg):
        if self.msg_count >= self.target_iterations:
            self.print_stats()
            rospy.signal_shutdown("Benchmark Evaluation Matrix Filled Successfully.")
            return

        capture_time = msg.header.stamp.to_sec()
        receive_time = rospy.Time.now().to_sec()
        cam_delay = (receive_time - capture_time) * 1000 
        
        t_start = time.perf_counter()
        
        try:
            # Native high-performance decompression pipeline
            np_arr = np.frombuffer(msg.data, np.uint8)
            obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            t0 = time.perf_counter()
            proc = self.agent.preprocess_cv(obs)
            self.agent.update_buffer(proc)
            dt_pre = (time.perf_counter() - t0) * 1000
            
            if len(self.agent.frames) == self.agent.frame_stack:
                t1 = time.perf_counter()
                action = self.agent.get_action()
                dt_nn = (time.perf_counter() - t1) * 1000
                
                t2 = time.perf_counter()
                _ = self.agent.postprocess_kinematics(action)
                dt_kin = (time.perf_counter() - t2) * 1000
                
                dt_compute = (time.perf_counter() - t_start) * 1000
                dt_total_pipeline = cam_delay + dt_compute
                
                # Append telemetry matrices
                self.camera_delays.append(cam_delay)
                self.compute_times.append(dt_compute)
                self.total_pipeline_times.append(dt_total_pipeline)
                self.pre_times.append(dt_pre)
                self.nn_times.append(dt_nn)
                self.kinematics_times.append(dt_kin)
                
                # Check for runtime deadline breaches
                if dt_compute > self.time_budget_ms:
                    self.deadline_breaches += 1
                
                self.msg_count += 1
                if self.msg_count % 30 == 0:
                    rospy.loginfo(f"Telemetry Matrix Aggregation Status: [{self.msg_count}/{self.target_iterations}] captured...")
                    
        except Exception as e:
            rospy.logerr(f"Live telemetry parsing failure thread state: {e}")

    def print_stats(self):
        print("\n" + "═"*60)
        print("🚀 END-TO-END GLASS-TO-ACTION PRODUCTION DIAGNOSTICS 🚀")
        print("═"*60)
        print(f"Profiled Compute Backend Engine: {str(self.agent.device).upper()}")
        print(f"Aggregated Frame Sample Count:  {len(self.total_pipeline_times)}")
        print("-" * 60)
        
        print("⏱️  LATENCY BREAKDOWNS (Mean ± Standard Deviation):")
        print(f" ├─ Driver Network Transport Link: {np.mean(self.camera_delays):6.2f} ms  (± {np.std(self.camera_delays):.2f} ms)")
        print(f" ├─ CV Image Decoding + Prep:     {np.mean(self.pre_times):6.2f} ms  (± {np.std(self.pre_times):.2f} ms)")
        print(f" ├─ Torch Neural Net Inference:   {np.mean(self.nn_times):6.2f} ms  (± {np.std(self.nn_times):.2f} ms)")
        print(f" ├─ Kinematic Actuator Transform: {np.mean(self.kinematics_times):6.2f} ms  (± {np.std(self.kinematics_times):.2f} ms)")
        print("-" * 60)
        
        avg_total = np.mean(self.total_pipeline_times)
        p99_total = np.percentile(self.total_pipeline_times, 99)
        true_max_hz = 1000.0 / np.mean(self.compute_times)
        breach_percentage = (self.deadline_breaches / self.msg_count) * 100
        
        print("🏁 SYSTEM STABILITY VERDICT:")
        print(f" ├─ Mean Glass-to-Action Latency:  {avg_total:.2f} ms")
        print(f" ├─ 99th Percentile Max Latency:   {p99_total:.2f} ms")
        print(f" ├─ Isolated Pure Node Frequency:  {true_max_hz:.2f} Hz")
        print(f" ├─ Compute Frame Budget Breaches: {self.deadline_breaches} occurrences ({breach_percentage:.1f}%)")
        print("-" * 60)
        
        if breach_percentage > 5.0:
            print("🚨 CRITICAL TELEMETRY VERDICT: UNSTABLE DEPLOYMENT ENVIRONMENT")
            print("   The node is dropping its processing deadline too frequently. The robot")
            print("   will drift asynchronously or display jittery lane tracking maneuvers.")
        else:
            print("✨ CRITICAL TELEMETRY VERDICT: OPTIMIZED AND PRODUCTION SAFE")
            print("   The pipeline is operating within stable time-delta bounds for real-world loops.")
        print("═"*60 + "\n")

if __name__ == '__main__':
    node = HardwareBenchmarkNode(node_name='advanced_hw_benchmark_node')
    rospy.spin()