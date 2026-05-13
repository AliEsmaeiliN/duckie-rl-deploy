#!/usr/bin/env python3
import os
import time
import numpy as np
from rl_package.agent import DuckiebotAgent

def mock_pipeline_benchmark(algo="sac", iterations=200):
    veh = os.environ.get('VEHICLE_NAME', 'duckie1nav')
    repo_path = os.environ.get("DT_REPO_PATH", "/code/duckie-rl-deploy")
    model_full_path = os.path.join(repo_path, f"assets/models/{algo}_v10.cleanrl_model")
    
    print(f"Initializing Agent for {veh} using {algo.upper()}...")
    
    agent = DuckiebotAgent(model_path=model_full_path, algo_type=algo)
    
    dummy_obs = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print(f"\n--- Benchmarking Full Pipeline on {agent.device} ---")
    
    print("Warming up models and memory buffers...")
    for _ in range(15):
        proc = agent.preprocess_cv(dummy_obs)
        agent.update_buffer(proc)
        if len(agent.frames) == agent.frame_stack:
            act = agent.get_action()
            _ = agent.postprocess_kinematics(act)
            
    # Benchmark phase
    print(f"Running {iterations} pipeline iterations...\n")
    pre_times, nn_times, kinematics_times, total_times = [], [], [], []

    for _ in range(iterations):
        t_start_total = time.perf_counter()
        
        # Step A: Vision Pipeline (Remap -> Crop -> Resize -> Grayscale)
        t0 = time.perf_counter()
        processed_frame = agent.preprocess_cv(dummy_obs)
        agent.update_buffer(processed_frame)
        t_pre = time.perf_counter() - t0
        
        if len(agent.frames) == agent.frame_stack:
            
            t1 = time.perf_counter()
            action = agent.get_action()
            t_nn = time.perf_counter() - t1
            
            t2 = time.perf_counter()
            wheel_cmds = agent.postprocess_kinematics(action)
            t_kin = time.perf_counter() - t2
            
            t_total = time.perf_counter() - t_start_total
            
            pre_times.append(t_pre * 1000)
            nn_times.append(t_nn * 1000)
            kinematics_times.append(t_kin * 1000)
            total_times.append(t_total * 1000)

    print("--- Benchmark Results ---")
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

if __name__ == "__main__":
    mock_pipeline_benchmark()