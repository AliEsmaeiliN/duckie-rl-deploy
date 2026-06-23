#!/usr/bin/env python3
import os
import sys
import json
import threading
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType # pyright: ignore[reportMissingImports]
from sensor_msgs.msg import CompressedImage # pyright: ignore[reportMissingImports]
from std_msgs.msg import String
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped # pyright: ignore[reportMissingImports]
from cv_bridge import CvBridge # pyright: ignore[reportMissingImports]
from rl_package.agent import DuckiebotAgent

class RLNode(DTROS):
    def __init__(self, node_name, algo="sac"):
        super(RLNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        self.bridge = CvBridge()
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot98')
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"
        self.algo = algo.lower()

        if self.debug_mode:
            from rl_package.debug_bot import run_remote_debug
            self.remote_debug = run_remote_debug

        self.state_lock = threading.Lock()
        self.active = False  
        self.shared_registry = "/mnt/shared_models"
        
        # Initial Model Setup
        model_ver = os.environ.get("MODEL", "v10")
        self.load_agent_model(model_ver)

        self.last_obs = None
        self.wheel_pub_wlwr = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)

        self.cmd_sub = rospy.Subscriber(
            f"/{self.veh}/{node_name}/commands", 
            String, 
            self.unified_command_callback, 
            queue_size=10
        )
        
        # Camera Subscriber
        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1, buff_size=2**24)

        self.frame_rate = 30
        self.action_freq = 15
        
        rospy.loginfo(f"Node started for {self.veh}. Mode: {'DEBUG' if self.debug_mode else 'INFERENCE'}")

    def load_agent_model(self, model_ver):
        """Helper to load or reload models safely."""
        model_full_path = os.path.join(self.shared_registry, f"{self.algo}_{model_ver}.cleanrl_model")

        if not os.path.exists(model_full_path):
            rospy.logwarn(f"Requested policy missing at target path: {model_full_path}")
            repo_path = os.environ.get("DT_REPO_PATH", "/code/catkin_ws/src/duckie-rl-deploy")
            model_full_path = os.path.join(repo_path, "assets/models/sac_v10.cleanrl_model")
            rospy.logwarn(f"Hard-locking execution structure to baseline container recovery policy: {model_full_path}")
        
        rospy.loginfo(f"Loading weights from: {model_full_path}")
        self.agent = DuckiebotAgent(
            model_path=model_full_path, 
            algo_type=self.algo
        )
        self.clear_frame_history()

    def clear_frame_history(self):
        """Flushes the temporal history buffer to prevent structural jerks."""
        if hasattr(self.agent, 'frames') and self.agent.frames is not None:
            self.agent.frames.clear()
            rospy.logdebug("Agent observation history cleared.")

    def unified_command_callback(self, msg):
        """Handles all state changes and hot-swaps via incoming JSON payloads."""
        try:
            payload = json.loads(msg.data)
            cmd = payload.get("cmd")
        except json.JSONDecodeError:
            rospy.logerr("Received malformed JSON on command topic.")
            return

        with self.state_lock:
            if cmd == "pause":
                self.active = False
                self.write("wheels", [0.0, 0.0])
                self.clear_frame_history()
                rospy.loginfo("RL Policy Execution: PAUSED. Emergency brakes applied.")
            
            elif cmd == "resume":
                self.clear_frame_history() # Clean slate for the sudden acceleration sequence
                self.active = True
                rospy.loginfo("RL Policy Execution: RESUMED.")
            
            elif cmd == "swap":
                new_model_ver = payload.get("model", "").strip()
                rospy.logwarn(f"Hot-swapping model weights to version: {new_model_ver}")
                
                # Force halt during transition sequence
                self.active = False
                self.write("wheels", [0.0, 0.0])
                
                try:
                    self.load_agent_model(new_model_ver)
                    rospy.loginfo(f"Successfully transitioned network weights to version {new_model_ver}")
                    # Keep it paused after swap for safety; requires an explicit 'resume' command from the CLI
                    rospy.loginfo("Agent is currently PAUSED with new weights. Send 'resume' to drive.")
                except Exception as e:
                    rospy.logerr(f"Failed to hot-swap model cleanly: {e}")
                    self.active = False
                    self.write("wheels", [0.0, 0.0])

    def callback(self, msg):
        with self.state_lock:
            if self.active:
                self.last_obs = self.bridge.compressed_imgmsg_to_cv2(msg)
        

    def run(self):
        rate = rospy.Rate(self.action_freq)
        while not rospy.is_shutdown():
            with self.state_lock:
                is_active = self.active
                current_obs = self.last_obs

            if is_active and current_obs is not None:
                processed_frame = self.agent.preprocess_cv(current_obs)
                self.agent.update_buffer(processed_frame)
                
                if len(self.agent.frames) == self.agent.frame_stack:
                    if self.debug_mode:
                        self.remote_debug(self.agent, self, processed_frame)
                    else:
                        action = self.agent.get_action()
                        wheel_cmds = self.agent.postprocess_kinematics(action)
                        self.write("wheels", wheel_cmds)
            else:
                self.write("wheels", [0.0, 0.0])

            rate.sleep()
    
    def write(self, topic, data):
        if topic == 'wheels':
            try:
                cmd_msg = WheelsCmdStamped()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.vel_left = data[0]
                cmd_msg.vel_right = data[1]
                self.wheel_pub_wlwr.publish(cmd_msg)
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

    node = RLNode(node_name='rl_node', algo=algo_arg)
    node.run()
    rospy.spin()