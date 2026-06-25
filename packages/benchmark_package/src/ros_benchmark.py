#!/usr/bin/env python3
import os
import sys
import time
import rospy
import cv2
import numpy as np

from rl_package.agent import DuckiebotAgent   # pyright: ignore[reportMissingImports]
from duckietown.dtros import DTROS, NodeType  # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage   # pyright: ignore[reportMissingImports]
from duckietown_msgs.msg import WheelsCmdStamped # pyright: ignore[reportMissingImports]

class HardwareBenchmarkNode(DTROS):
    def __init__(self, node_name):
        super(HardwareBenchmarkNode, self).__init__(node_name=node_name, node_type=NodeType.DIAGNOSTICS)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot98')
    
        self.shared_registry = "/mnt/shared_models"
        repo_path = os.environ.get("DT_REPO_PATH", "/code/catkin_ws/src/duckie-rl-deploy")
        model_ver = os.environ.get("MODEL", "vu1b")
        
        # Look for the optimized ONNX graph
        model_full_path = os.path.join(repo_path, f"assets/models/sac_{model_ver}.onnx")
        self.load_agent_model(model_full_path, model_ver)
        
        # Initialize the Physical Wheel Publisher (Glass-to-Rubber Benchmark)
        self.pub_wheels = rospy.Publisher(
            f"/{self.veh}/wheels_driver_node/wheels_cmd", 
            WheelsCmdStamped, 
            queue_size=1
        )
        
        # Comprehensive Streaming Metrics Storage Arrays
        self.camera_delays = []
        self.compute_times = []
        self.total_pipeline_times = []
        self.pre_times = []
        self.nn_times = []
        self.kinematics_times = []
        self.publish_times = [] # Tracks ROS serialization latency
        
        self.msg_count = 0
        self.target_iterations = 150 # Large sample size for high-fidelity variance tracking
        self.deadline_breaches = 0
        
        # 7.5 Hz Control loop budget (133.33 ms)
        self.time_budget_ms = (1.0 / 7.5) * 1000  

        # Run Detailed Stress Diagnostics (WARNING: WHEELS WILL SPIN)
        self.run_isolated_dry_run(iterations=100)

        rospy.loginfo(f"Initialization Complete. Opening Live ROS Capture Subscriptions on /{self.veh}...")
        self.sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed", 
            CompressedImage, 
            self.callback, 
            queue_size=1, 
            buff_size=2**24
        )

    def load_agent_model(self, model_full_path, model_ver):
        """Helper to load ONNX models safely."""
        if not os.path.exists(model_full_path):
            rospy.logwarn(f"Requested policy missing at target path: {model_full_path}")
            # Fallback check inside the shared registry
            fallback_path = os.path.join(self.shared_registry, f"sac_{model_ver}.onnx")
            if os.path.exists(fallback_path):
                model_full_path = fallback_path
            else:
                rospy.logwarn(f"Could not find ONNX model. Ensure the .onnx file exists.")
        
        rospy.loginfo(f"Loading ONNX graph from: {model_full_path}")
        self.agent = DuckiebotAgent(
            model_path=model_full_path, 
            algo_type='sac'
        )
        
        # Reset frame stacking buffer
        if hasattr(self.agent, 'clear_frame_history'):
            self.agent.clear_frame_history()
        
    def get_hardware_memory_info(self):
        """
        Retrieves raw runtime memory allocation natively from Linux.
        Because PyTorch is removed, we read the Resident Set Size (VmRSS). 
        On Jetson Nano's Unified Memory, this captures both CPU and GPU workspace RAM!
        """
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if 'VmRSS:' in line:
                        kb = int(line.split()[1])
                        mb = kb / 1024.0
                        return f"Unified RAM Allocated: {mb:.2f} MB"
        except IOError:
            pass
        return "Hardware Memory: Profiling Unavailable"

    def run_isolated_dry_run(self, iterations):
        dummy_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        
        backend_info = "ONNX Runtime Execution"
        if hasattr(self.agent, 'session'):
            backend_info = str(self.agent.session.get_providers())

        print("\n" + "═"*60)
        print(f"🔬 ADVANCED SYNTHETIC PROFILER RUNNING ON:\n   {backend_info}")
        print("═"*60)
        print(f"Initial Memory Footprint: {self.get_hardware_memory_info()}")
        print("⚠️ WARNING: SYNTHETIC DRY RUN WILL SPIN THE PHYSICAL WHEELS! KEEP ROBOT ELEVATED.")
        
        print("Warming up compute layer tensor paths...")
        for _ in range(20):
            proc = self.agent.preprocess_cv(dummy_obs)
            self.agent.update_buffer(proc)
            if len(self.agent.frames) == getattr(self.agent, 'frame_stack', 4):
                _ = self.agent.postprocess_kinematics(self.agent.get_action())
                
        print(f"Executing {iterations} continuous stress iterations...")
        pre_times, nn_times, kin_times, pub_times, total_times = [], [], [], [], []

        for _ in range(iterations):
            t_start = time.perf_counter()
            
            # Sub-component 1: Computer Vision Step
            t0 = time.perf_counter()
            proc = self.agent.preprocess_cv(dummy_obs)
            self.agent.update_buffer(proc)
            dt_pre = (time.perf_counter() - t0) * 1000
            
            if len(self.agent.frames) == getattr(self.agent, 'frame_stack', 4):
                # Sub-component 2: ONNX Model Inference Execution
                t1 = time.perf_counter()
                action = self.agent.get_action()
                dt_nn = (time.perf_counter() - t1) * 1000
                
                # Sub-component 3: Kinematics Mapping
                t2 = time.perf_counter()
                u_l, u_r = self.agent.postprocess_kinematics(action)
                dt_kin = (time.perf_counter() - t2) * 1000
                
                # Sub-component 4: ROS Network Serialization
                t3 = time.perf_counter()
                msg_wheels = WheelsCmdStamped()
                msg_wheels.header.stamp = rospy.Time.now()
                msg_wheels.vel_left = float(u_l)
                msg_wheels.vel_right = float(u_r)
                self.pub_wheels.publish(msg_wheels)
                dt_pub = (time.perf_counter() - t3) * 1000
                
                dt_total = (time.perf_counter() - t_start) * 1000
                
                pre_times.append(dt_pre)
                nn_times.append(dt_nn)
                kin_times.append(dt_kin)
                pub_times.append(dt_pub)
                total_times.append(dt_total)

        print("\n📊 ISOLATED COMPONENT BREAKDOWNS (Mean ± StdDev):")
        print(f" └─ Vision Pipeline:   {np.mean(pre_times):6.2f} ms  (± {np.std(pre_times):.2f} ms)")
        print(f" └─ ONNX Inference:    {np.mean(nn_times):6.2f} ms  (± {np.std(nn_times):.2f} ms)")
        print(f" └─ Kinematics Wrap:   {np.mean(kin_times):6.2f} ms  (± {np.std(kin_times):.2f} ms)")
        print(f" └─ ROS Async Pub:     {np.mean(pub_times):6.2f} ms  (± {np.std(pub_times):.2f} ms)")
        print(f" └─ Total Node Loop:   {np.mean(total_times):6.2f} ms  (± {np.std(total_times):.2f} ms)")
        
        p99 = np.percentile(total_times, 99)
        max_f = 1000.0 / np.mean(total_times)
        
        print("\n📈 PERFORMANCE METRICS SUMMARY:")
        print(f" ├─ Theoretical Maximum Throughput: {max_f:.2f} Hz")
        print(f" ├─ Worst-Case Tail Latency (P99):  {p99:.2f} ms")
        print(f" ├─ Final Stable Memory Footprint:  {self.get_hardware_memory_info()}")
        print("═"*60 + "\n")

    def callback(self, msg):
        if self.msg_count >= self.target_iterations:
            self.print_stats()
            # Issue an emergency stop to the wheels before shutting down
            stop_msg = WheelsCmdStamped()
            stop_msg.header.stamp = rospy.Time.now()
            stop_msg.vel_left = 0.0
            stop_msg.vel_right = 0.0
            self.pub_wheels.publish(stop_msg)
            
            rospy.signal_shutdown("Benchmark Evaluation Matrix Filled Successfully.")
            return

        capture_time = msg.header.stamp.to_sec()
        receive_time = rospy.Time.now().to_sec()
        cam_delay = (receive_time - capture_time) * 1000 
        
        t_start = time.perf_counter()
        
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            t0 = time.perf_counter()
            proc = self.agent.preprocess_cv(obs)
            self.agent.update_buffer(proc)
            dt_pre = (time.perf_counter() - t0) * 1000
            
            if len(self.agent.frames) == getattr(self.agent, 'frame_stack', 4):
                t1 = time.perf_counter()
                action = self.agent.get_action()
                dt_nn = (time.perf_counter() - t1) * 1000
                
                t2 = time.perf_counter()
                u_l, u_r = self.agent.postprocess_kinematics(action)
                dt_kin = (time.perf_counter() - t2) * 1000
                
                t3 = time.perf_counter()
                msg_wheels = WheelsCmdStamped()
                msg_wheels.header.stamp = rospy.Time.now()
                msg_wheels.vel_left = float(u_l)
                msg_wheels.vel_right = float(u_r)
                self.pub_wheels.publish(msg_wheels)
                dt_pub = (time.perf_counter() - t3) * 1000
                
                dt_compute = (time.perf_counter() - t_start) * 1000
                dt_total_pipeline = cam_delay + dt_compute
                
                # Append telemetry matrices
                self.camera_delays.append(cam_delay)
                self.compute_times.append(dt_compute)
                self.total_pipeline_times.append(dt_total_pipeline)
                self.pre_times.append(dt_pre)
                self.nn_times.append(dt_nn)
                self.kinematics_times.append(dt_kin)
                self.publish_times.append(dt_pub)
                
                if dt_compute > self.time_budget_ms:
                    self.deadline_breaches += 1
                
                self.msg_count += 1
                if self.msg_count % 30 == 0:
                    rospy.loginfo(f"Telemetry Matrix Aggregation Status: [{self.msg_count}/{self.target_iterations}] captured...")
                    
        except Exception as e:
            rospy.logerr(f"Live telemetry parsing failure thread state: {e}")

    def print_stats(self):
        backend_info = str(self.agent.session.get_providers()) if hasattr(self.agent, 'session') else "ONNX Engine"
        
        print("\n" + "═"*60)
        print("🚀 END-TO-END GLASS-TO-RUBBER PRODUCTION DIAGNOSTICS 🚀")
        print("═"*60)
        print(f"Profiled Compute Backend Engine: {backend_info}")
        print(f"Aggregated Frame Sample Count:  {len(self.total_pipeline_times)}")
        print("-" * 60)
        
        print("⏱️  LATENCY BREAKDOWNS (Mean ± Standard Deviation):")
        print(f" ├─ Driver Network Transport Link: {np.mean(self.camera_delays):6.2f} ms  (± {np.std(self.camera_delays):.2f} ms)")
        print(f" ├─ CV Image Decoding + Prep:      {np.mean(self.pre_times):6.2f} ms  (± {np.std(self.pre_times):.2f} ms)")
        print(f" ├─ ONNX C++ Graph Inference:      {np.mean(self.nn_times):6.2f} ms  (± {np.std(self.nn_times):.2f} ms)")
        print(f" ├─ Kinematic Actuator Transform:  {np.mean(self.kinematics_times):6.2f} ms  (± {np.std(self.kinematics_times):.2f} ms)")
        print(f" ├─ ROS Asynchronous Wheel Pub:    {np.mean(self.publish_times):6.2f} ms  (± {np.std(self.publish_times):.2f} ms)")
        print("-" * 60)
        
        avg_total = np.mean(self.total_pipeline_times)
        p99_total = np.percentile(self.total_pipeline_times, 99)
        true_max_hz = 1000.0 / np.mean(self.compute_times)
        breach_percentage = (self.deadline_breaches / self.msg_count) * 100
        
        print("🏁 SYSTEM STABILITY VERDICT:")
        print(f" ├─ Mean Glass-to-Rubber Latency:  {avg_total:.2f} ms")
        print(f" ├─ 99th Percentile Max Latency:   {p99_total:.2f} ms")
        print(f" ├─ Isolated Pure Node Frequency:  {true_max_hz:.2f} Hz")
        print(f" ├─ Compute Frame Budget Breaches: {self.deadline_breaches} occurrences ({breach_percentage:.1f}%)")
        print("-" * 60)
        
        if breach_percentage > 5.0:
            print("🚨 CRITICAL TELEMETRY VERDICT: UNSTABLE DEPLOYMENT ENVIRONMENT")
            print("   The node is dropping its processing deadline too frequently.")
        else:
            print("✨ CRITICAL TELEMETRY VERDICT: OPTIMIZED AND PRODUCTION SAFE")
            print("   The pipeline is operating within stable time-delta bounds for real-world loops.")
        print("═"*60 + "\n")

if __name__ == '__main__':
    node = HardwareBenchmarkNode(node_name='onnx_hw_benchmark_node')
    rospy.spin()