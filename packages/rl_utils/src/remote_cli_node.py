#!/usr/bin/env python3
import rospy
import json
import cmd
import os
import sys
from std_msgs.msg import String

class RLRemoteCLI(cmd.Cmd):
    intro = '\n=== Duckiebot RL Remote Control ===\nType help or ? to list commands.\n'
    prompt = '\033[1;36m(rl-remote)\033[0m '  # Cyan colored prompt

    def __init__(self):
        super().__init__()
        # anonymous=True allows multiple CLI instances if needed
        rospy.init_node('rl_remote_cli', anonymous=True)
        
        self.vehicle_name = os.environ.get("VEHICLE_NAME", "duckiebot98")
        topic_name = f"/{self.vehicle_name}/rl_node/commands"
        
        self.pub_cmds = rospy.Publisher(topic_name, String, queue_size=10)
        
        # Give the publisher a moment to register with the ROS Master
        rospy.sleep(0.5)
        print(f"Connected to ROS Master. Targeting vehicle: {self.vehicle_name}")
        print(f"Publishing to: {topic_name}")

    def send_command(self, payload_dict):
        msg = String()
        msg.data = json.dumps(payload_dict)
        self.pub_cmds.publish(msg)

    def do_pause(self, arg):
        """Pause the RL agent's control loop. Sends emergency zero-velocity commands."""
        self.send_command({"cmd": "pause"})
        print(">> Sent: PAUSE")

    def do_resume(self, arg):
        """Resume the RL agent's control loop and clears the frame history."""
        self.send_command({"cmd": "resume"})
        print(">> Sent: RESUME")

    def do_swap(self, arg):
        """Swap the running model dynamically. Usage: swap [version_name] (e.g., swap v10)"""
        arg = arg.strip()
        if not arg:
            print(">> Error: Please specify a model version (e.g., swap v10)")
            return
        
        print(f">> Sent: SWAP to model '{arg}'. Agent will PAUSE and await a 'resume' command.")
        self.send_command({"cmd": "swap", "model": arg})

    def do_status(self, arg):
        """Pings the current status (Customizable based on your needs)"""
        print(f">> Targeting vehicle: {self.vehicle_name}")
        print(">> Note: To get live telemetry, consider adding a telemetry subscriber here in the future.")

    def do_exit(self, arg):
        """Exit the remote CLI."""
        print("Exiting Remote CLI...")
        rospy.signal_shutdown("User exited CLI.")
        return True
    
    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        return self.do_exit(arg)

if __name__ == '__main__':
    try:
        cli = RLRemoteCLI()
        cli.cmdloop()
    except rospy.ROSInterruptException:
        print("\nROS Interrupted. Exiting...")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)