#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage # pyright: ignore[reportMissingImports]
from duckietown_msgs.msg import WheelsCmdStamped # pyright: ignore[reportMissingImports]
from packages.agent import DuckiebotAgent
from packages.debug_bot import run_remote_debug

class RLNode(DTROS):
    def __init__(self, node_name, algo="sac"):
        super(RLNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        self.veh = os.environ.get('VEHICLE_NAME', 'duckie1nav')

        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"

        repo_path = os.environ.get("DT_REPO_PATH", "/code/duckie-rl-deploy")
        model_full_path = os.path.join(repo_path, f"assets/models/{algo}_Final.cleanrl_model")
        
        self.agent = DuckiebotAgent(
            model_path=model_full_path, 
            algo_type=algo
        )
        self.last_obs = None
        self.wheel_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo(f"Node started for {self.veh}. Mode: {'DEBUG' if self.debug_mode else 'INFERENCE'}")

    def callback(self, msg):
        
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_obs = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def run(self):
        rate = rospy.Rate(7.5)
        while not rospy.is_shutdown():
            if self.last_obs is not None:
                if self.debug_mode:
                    run_remote_debug(self.agent, self, self.last_obs)
                else:
                    action = self.agent.get_action(self.last_obs)
                    wheel_cmds = self.agent.postprocess_kinematics(action)
                    self.write("wheels", wheel_cmds)

            rate.sleep()
    
    def write(self, topic, data):
        if topic == 'wheels':
            try:
                cmd_msg = WheelsCmdStamped()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.vel_left = data[0]
                cmd_msg.vel_right = data[1]
                self.wheel_pub.publish(cmd_msg)
            except (rospy.ROSException, rospy.ROSInterruptException):
                pass
            
    def on_shutdown(self):
        rospy.loginfo("Emergency Stop triggered by DTROS shutdown.... sending stop command to wheels.")
        self.write("wheels", [0.0, 0.0])
        super(RLNode, self).on_shutdown()

if __name__ == '__main__':
    algo_arg = "sac"
    if "--algo" in sys.argv:
        idx = sys.argv.index("--algo")
        algo_arg = sys.argv[idx + 1]

    node = RLNode(node_name='rl_agent_node', algo=algo_arg)
    node.run()
    rospy.spin()