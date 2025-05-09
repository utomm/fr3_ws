#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf
import socket
import msgpack
import msgpack_numpy
import numpy as np
import sys
from sensor_msgs.msg import PointCloud2, JointState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import struct
import cv2

msgpack_numpy.patch()

# Globals
bridge = CvBridge()
latest_depth = None
latest_rgb = None
latest_joint_states = None

# ROS Callbacks
def depth_callback(msg):
    global latest_depth
    latest_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

def rgb_callback(msg):
    global latest_rgb
    latest_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # Match collection_server

def joint_states_callback(msg):
    global latest_joint_states
    latest_joint_states = np.array(msg.position, dtype=np.float32)

# Socket Communication
def socket_recv(sock):
    header = sock.recv(4)
    if not header:
        return None
    msg_len = int.from_bytes(header, 'big')
    chunks = []
    while msg_len > 0:
        chunk = sock.recv(min(4096, msg_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        chunks.append(chunk)
        msg_len -= len(chunk)
    return msgpack.unpackb(b''.join(chunks), object_hook=msgpack_numpy.decode)

def socket_send(sock, data):
    packed = msgpack.packb(data, default=msgpack_numpy.encode)
    sock.sendall(len(packed).to_bytes(4, 'big') + packed)

# Transform Pose
def transform_pose(listener, pose_table, source_frame='table', target_frame='fr3_link0'):
    br = tf.TransformListener()
    br.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
    (trans, rot) = br.lookupTransform(target_frame, source_frame, rospy.Time(0))
    matrix = tf.transformations.quaternion_matrix(rot)
    matrix[0:3, 3] = trans
    pose_homo = np.hstack((pose_table[:3], 1.0))
    transformed = matrix @ pose_homo
    pos_new = transformed[:3]

    q_orig = pose_table[3:]
    q_rot = rot
    q_new = tf.transformations.quaternion_multiply(q_rot, q_orig)

    return pos_new, q_new

def get_gripper_pose(listener, source_frame="table", target_frame="fr3_hand_tcp"):
    try:
        listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(1.0))
        (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
        gripper_pose = np.array(list(trans) + list(rot), dtype=np.float32)
        return gripper_pose
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn(f"TF lookup failed: {e}")
        return None

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
    # Match depth_to_xyz logic from collection_server
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

def main():
    global latest_depth, latest_rgb, latest_joint_states

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("dataset_executor_server", anonymous=True)

    # Camera intrinsic matrix (same as collection_server)
    K = np.array([
        [543.15, 0.0, 320.0],
        [0.0, 543.15, 240.0],
        [0.0, 0.0, 1.0]
    ])

    # ROS Subscribers
    rospy.Subscriber('/stereo/depth', Image, depth_callback)  # Match collection_server
    rospy.Subscriber('/stereo/left/image_rect_color', Image, rgb_callback)  # Match collection_server
    rospy.Subscriber('/joint_states', JointState, joint_states_callback)

    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander("fr3_manipulator")
    listener = tf.TransformListener()

    print("planning base frame:")
    print(move_group.get_planning_frame())
    print("planning ee frame:")
    print(move_group.get_end_effector_link())
    
    gripper_group_name = "fr3_hand"  # Replace with your gripper group name
    gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name)


    # Action Socket
    action_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    action_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    action_server.bind(('127.0.0.1', 5006))
    action_server.listen(1)
    print("[Server] Waiting for action client...")
    action_conn, action_addr = action_server.accept()
    print(f"[Server] Action client connected from {action_addr}")

    # Observation Socket
    obs_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    obs_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    obs_server.bind(('127.0.0.1', 5007))
    obs_server.listen(1)
    print("[Server] Waiting for observation client...")
    obs_conn, obs_addr = obs_server.accept()
    print(f"[Server] Observation client connected from {obs_addr}")

    while not rospy.is_shutdown():
        try:
            # Receive action
            action_msg = socket_recv(action_conn)
            if action_msg is None:
                break
            print("[Server] Received action to execute")
            
            socket_send(action_conn, {'type': 'confirmation', 'status': 'received'})
            print("[Server] Sent action confirmation to client.")

            gripper_pose_table = action_msg['gripper']
            pos, quat = transform_pose(listener, gripper_pose_table[:7])
            openness = gripper_pose_table[7]

            # Execute action
            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x, target_pose.position.y, target_pose.position.z = pos
            target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = quat

            waypoints = []
            start_pose = move_group.get_current_pose().pose

            # Add the target pose as a waypoint
            waypoints.append(start_pose)  # Start pose
            waypoints.append(target_pose)  # Target pose

            # Try Cartesian path planning
            (cartesian_plan, fraction) = move_group.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step (meters)
                0.0          # jump_threshold
            )

            if False and fraction == 1.0:  # Cartesian planning succeeded
                print("[Server] Cartesian path planning succeeded.")
                
                    # Fix trajectory timestamps
                for i, point in enumerate(cartesian_plan.joint_trajectory.points):
                    point.time_from_start = rospy.Duration(i * 0.1)  # Assign time in 0.1-second intervals
                
                move_group.execute(cartesian_plan, wait=True)
            else:
                print("[Server] Cartesian path planning failed. Falling back to default planning.")
                move_group.set_pose_target(target_pose)
                move_group.go(wait=True)

            # Stop and clear targets
            move_group.stop()
            move_group.clear_pose_targets()
            print("[Server] Movement complete.")
            
            # Plan and execute gripper movement
            print("[Server] Moving gripper to target joint values.")
            gripper_group.set_joint_value_target([openness * 0.04, openness * 0.04])
            gripper_group.go(wait=True)
            gripper_group.stop()
            gripper_group.clear_pose_targets()
            print("[Server] Gripper movement complete.")


            # Collect observation
            rospy.sleep(2)  # Wait for sensors to stabilize
            if latest_depth is None or latest_rgb is None or latest_joint_states is None:
                print("[Server] Missing sensor data, skipping observation.")
                # print which one is missing
                if latest_depth is None:
                    print("[Server] Depth data is missing.")
                if latest_rgb is None:
                    print("[Server] RGB data is missing.")
                if latest_joint_states is None:
                    print("[Server] Joint states data is missing.")
                continue

            # Get gripper pose and camera-to-table transformation
            gripper_pose = get_gripper_pose(listener)
            cam_to_table_tf = get_cam_to_table(listener)

            # Convert point cloud to numpy array
            xyz_table, valid_mask = depth_to_xyz(latest_depth, K, cam_to_table_tf)
            rgb_resized = cv2.resize(latest_rgb, (latest_depth.shape[1], latest_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgb_flat = rgb_resized.reshape(-1, 3)
            rgb_valid = rgb_flat[valid_mask]

            # Prepare observation
            obs = {
                'xyz': xyz_table,
                'rgb': rgb_valid,
                'joint_states': latest_joint_states,
                'gripper': gripper_pose
            }

            # Send observation
            socket_send(obs_conn, {'type': 'observation', 'data': obs})
            print("[Server] Sent observation to client.")
            # Wait for client response
            response = socket_recv(obs_conn)
            if response is None:
                break
            print("[Server] Received response from client:", response)

        except Exception as e:
            print("[Server] Error:", e)
            break

    action_conn.close()
    obs_conn.close()
    moveit_commander.roscpp_shutdown()
    print("[Server] Closed")

if __name__ == "__main__":
    main()
