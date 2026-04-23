import socket
import pickle
import struct
import cv2
import numpy as np

LAPTOP_IP = "192.168.0.51" 
PORT = 8089

# Global socket to keep connection alive across steps
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((LAPTOP_IP, PORT))
except Exception as e:
    print(f"Could not connect to laptop: {e}")

def run_remote_debug(agent, context, observation):
    """
    Called by solution.py solve() function.
    """

    stacked_frames = np.concatenate(list(agent.frames), axis=0)

    action = agent.get_action()
    action = [action[0], action[1]]
    wheel_cmds = agent.postprocess_kinematics(action)

    data = {
        "image": stacked_frames, 
        "action": action,
        "motors": wheel_cmds
    }
    
    # Send via socket...
    try:
        msg = pickle.dumps(data)
        client_socket.sendall(struct.pack("Q", len(msg)) + msg)
    except Exception as e:
        print(f"Send failed: {e}")