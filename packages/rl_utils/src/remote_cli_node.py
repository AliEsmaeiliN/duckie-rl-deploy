#!/usr/bin/env python3
import rospy
import json
import os
import sys
import time
from std_msgs.msg import String

class RLRemoteCLI:
    def __init__(self):
        rospy.init_node('rl_remote_cli', anonymous=True)
        
        self.vehicle_name = os.environ.get("VEHICLE_NAME", "duckiebot98")
        topic_name = f"/{self.vehicle_name}/rl_node/commands"
        
        self.pub_cmds = rospy.Publisher(topic_name, String, queue_size=10)
        
        rospy.sleep(0.5)
        print("\n" + "="*40)
        print("=== Duckiebot RL Remote Control Center ===")
        print(f"Target Vehicle : {self.vehicle_name}")
        print(f"Target Topic   : {topic_name}")
        print("="*40)
        print("Available Commands: pause, resume, swap [version], exit\n")

    def send_command(self, payload_dict):
        msg = String()
        msg.data = json.dumps(payload_dict)
        self.pub_cmds.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            try:
                # If stdin is detached/non-interactive, input() throws EOFError immediately
                user_input = input(f"\033[1;36m({self.vehicle_name}-remote) #\033[0m ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == "exit":
                    print("Exiting Remote Control...")
                    break
                elif cmd == "pause":
                    self.send_command({"cmd": "pause"})
                    print(">> Sent: PAUSE")
                elif cmd == "resume":
                    self.send_command({"cmd": "resume"})
                    print(">> Sent: RESUME")
                elif cmd == "swap":
                    if len(parts) < 2:
                        print(">> Error: Please specify a model version (e.g., swap v10)")
                        continue
                    model_ver = parts[1]
                    self.send_command({"cmd": "swap", "model": model_ver})
                    print(f">> Sent: SWAP to model '{model_ver}'")
                else:
                    print(f">> Unknown command: '{cmd}'")

            except EOFError:
                rospy.sleep(1.0)
            except KeyboardInterrupt:
                print("\nExiting Remote Control gracefully...")
                break

if __name__ == '__main__':
    try:
        cli = RLRemoteCLI()
        cli.run()
    except rospy.ROSInterruptException:
        pass