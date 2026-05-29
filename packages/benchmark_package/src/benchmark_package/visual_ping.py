#!/usr/bin/env python3
import os
import sys
import rospy
from duckietown.dtros import DTROS, NodeType # pyright: ignore[reportMissingImports]
from duckietown_msgs.msg import WheelsCmdStamped # pyright: ignore[reportMissingImports]

class ScreenPingNode(DTROS):
    def __init__(self, node_name):
        super(ScreenPingNode, self).__init__(node_name=node_name, node_type=NodeType.DIAGNOSTICS)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckie1nav')
        
        self.wheel_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1)
        
    def run_multi_ping(self):
        print("\n" * 10)
        rospy.loginfo("Start your Slo-Mo camera.")
        rospy.loginfo("Point the camera so it can see BOTH this laptop screen AND the Duckiebot wheel!")
        rospy.sleep(5.0)

        for i in range(1, 4):
            rospy.loginfo(f"Ping {i}/3 firing in 3 seconds...")
            rospy.sleep(3.0)

            # 1. Prepare Wheel Message
            wheel_msg = WheelsCmdStamped()
            wheel_msg.header.stamp = rospy.Time.now()
            wheel_msg.vel_left = 0.4
            wheel_msg.vel_right = 0.4

            # 2. FIRE THE COMMAND
            self.wheel_pub.publish(wheel_msg)
            
            # 3. INSTANTLY FLASH THE LAPTOP SCREEN
            print("\n" * 10)
            print("████████████████████████████████████████████████████████████")
            print("█████████████████████   🚨 GO! 🚨  █████████████████████████")
            print("████████████████████████████████████████████████████████████")
            print("\n" * 10)
            sys.stdout.flush()

            # 4. Spin for 1 second
            rospy.sleep(1.0)
            
            # 5. Stop
            stop_msg = WheelsCmdStamped()
            stop_msg.header.stamp = rospy.Time.now()
            stop_msg.vel_left = 0.0
            stop_msg.vel_right = 0.0
            self.wheel_pub.publish(stop_msg)
            
            print("\n" * 40 + "MOTORS STOPPED. Waiting for next ping..." + "\n" * 10)

        rospy.loginfo("✅ All 3 tests complete. You can stop recording.")
        rospy.signal_shutdown("Ping Complete")

if __name__ == '__main__':
    node = ScreenPingNode(node_name='screen_ping_node')
    node.run_multi_ping()