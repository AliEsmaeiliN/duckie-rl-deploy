#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import rospy # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage # pyright: ignore[reportMissingImports]
from duckietown_msgs.msg import WheelsCmdStamped # pyright: ignore[reportMissingImports]
from packages.agent import DuckiebotAgent
from packages.debug_bot import run_remote_debug

class RLNode:
    def __init__(self, algo="sac"):
        rospy.init_node('rl_agent_node')
        self.veh = os.environ.get('VEHICLE_NAME', 'duckie1nav')

        repo_path = os.environ.get("DT_REPO_PATH", "/solution")
        model_full_path = os.path.join(repo_path, f"assets/models/{algo}_Final.cleanrl_model")
        
        self.agent = DuckiebotAgent(
            model_path=model_full_path, 
            algo_type=algo
        )
        
        self.wheel_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("Node started. Listening for camera frames...")
        rospy.on_shutdown(self.emergency_stop)

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        run_remote_debug(self.agent, self, obs)

    def write(self, topic, data):
        if topic == 'wheels':
            pass
            
    def emergency_stop(self):
        rospy.loginfo("Shutting down... sending stop command to wheels.")
        stop_msg = WheelsCmdStamped()
        stop_msg.vel_left = 0.0
        stop_msg.vel_right = 0.0
        self.wheel_pub.publish(stop_msg)

if __name__ == '__main__':
    algo_arg = "sac"
    if "--algo" in sys.argv:
        idx = sys.argv.index("--algo")
        algo_arg = sys.argv[idx + 1]

    node = RLNode(algo=algo_arg)
    rospy.spin()