#!/usr/bin/env python3
import rospy
import tf
import socket
import threading
import numpy as np
import msgpack
import msgpack_numpy
import argparse
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import tkinter as tk

msgpack_numpy.patch()

# ==== Globals ====
latest_depth = None
latest_rgb = None
latest_joint = None
bridge = CvBridge()
tf_listener = None

# ==== Socket Communication ====
def send_msg(sock, data):
    packed = msgpack.packb(data, default=msgpack_numpy.encode)
    sock.sendall(len(packed).to_bytes(4, 'big') + packed)

def recv_response(sock):
    header = sock.recv(4)
    if not header:
        return None
    msg_len = int.from_bytes(header, 'big')
    chunks = []
    while msg_len > 0:
        chunk = sock.recv(min(4096, msg_len))
        if not chunk:
            raise ConnectionError("Connection closed while receiving response")
        chunks.append(chunk)
        msg_len -= len(chunk)
    return msgpack.unpackb(b''.join(chunks), object_hook=msgpack_numpy.decode)

# ==== ROS Callbacks ====
def depth_callback(msg):
    global latest_depth
    latest_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def rgb_callback(msg):
    global latest_rgb
    latest_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def joint_callback(msg):
    global latest_joint
    latest_joint = np.array(msg.position, dtype=np.float32)

# ==== TF Gripper Pose ====
def get_gripper_pose(listener, source_frame="table", target_frame="fr3_hand_tcp"):
    try:
        listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
        gripper_pose = np.array(list(trans) + list(rot), dtype=np.float32)
        return gripper_pose
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn(f"TF lookup failed: {e}")
        return None

# ==== Utilities ====
def get_cam_to_table(listener, source_frame="table", camera_frame="camera"):
    try:
        listener.waitForTransform(source_frame, camera_frame, rospy.Time(0), rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform(source_frame, camera_frame, rospy.Time(0))
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]
        tf_matrix[:3, 3] = trans
        return tf_matrix
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn(f"TF lookup failed: {e}")
        return np.eye(4)

def depth_to_xyz(depth, K, cam_to_table_tf):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    xyz_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid_mask = (depth > 0).reshape(-1)
    xyz_cam = xyz_cam[valid_mask]

    xyz_cam_homo = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1))], axis=-1)
    xyz_table_homo = (cam_to_table_tf @ xyz_cam_homo.T).T
    xyz_table = xyz_table_homo[:, :3]

    return xyz_table, valid_mask

# ==== GUI ====
def create_gui(trigger_step, trigger_done):
    win = tk.Tk()
    win.title("Data Collector")

    tk.Button(win, text="Collect Step", width=25, command=trigger_step).pack(pady=10)
    tk.Button(win, text="Finish Episode", width=25, command=trigger_done).pack(pady=10)

    win.mainloop()

# ==== Main Loop ====
def main():
    global tf_listener

    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_K', nargs=9, type=float, default=[
        543.15, 0.0, 320.0,
        0.0, 543.15, 240.0,
        0.0, 0.0, 1.0
    ], help='Camera intrinsic matrix flattened')
    args = parser.parse_args()
    K = np.array(args.cam_K).reshape(3, 3)

    rospy.init_node("data_collection_socket_server")

    rospy.Subscriber('/stereo/depth', Image, depth_callback)
    rospy.Subscriber('/stereo/left/image_rect_color', Image, rgb_callback)
    rospy.Subscriber('/joint_states', JointState, joint_callback)
    tf_listener = tf.TransformListener()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', 5005))
    sock.listen(1)
    print("[Server] Waiting for client...")
    conn, addr = sock.accept()
    print(f"[Server] Client connected from {addr}")

    episode = []

    def trigger_step():
        cam_to_table_tf = get_cam_to_table(tf_listener)
        gripper_pose = get_gripper_pose(tf_listener)
        if gripper_pose is None or latest_depth is None or latest_rgb is None or latest_joint is None:
            print("[Server] Missing data, skipping step.")
            return

        xyz_table, valid_mask = depth_to_xyz(latest_depth, K, cam_to_table_tf)
        rgb_resized = cv2.resize(latest_rgb, (latest_depth.shape[1], latest_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        rgb_flat = rgb_resized.reshape(-1, 3)
        rgb_valid = rgb_flat[valid_mask]

        obs = {
            'xyz': xyz_table,
            'rgb': rgb_valid,
            'joint_states': latest_joint,
            'gripper': gripper_pose
        }
        print("[Server] Collected Step.")
        episode.append(obs)
        send_msg(conn, {'type': 'step', 'data': obs})
        try:
            result = recv_response(conn)
            print("[Server] Client response:", result)
        except Exception as e:
            print("[Server] Error receiving client response:", e)

    def trigger_done():
        print(f"[Server] Episode length: {len(episode)}")
        send_msg(conn, {'type': 'done'})
        episode.clear()
        print("[Server] Episode cleared.")

    threading.Thread(target=create_gui, args=(trigger_step, trigger_done), daemon=True).start()

    rospy.spin()
    conn.close()
    sock.close()
    print("[Server] Closed.")

if __name__ == '__main__':
    main()
