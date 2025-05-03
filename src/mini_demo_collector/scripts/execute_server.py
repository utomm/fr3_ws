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
import sys  # Add this import

msgpack_numpy.patch()

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

def transform_pose(listener, pose_table, source_frame='table', target_frame='fr3_link0'):  # Updated source_frame
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

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("dataset_executor_server", anonymous=True)

    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander("fr3_arm")
    listener = tf.TransformListener()

    # --- Socket setup ---
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 5006))
    server.listen(1)
    print("[Server] Waiting for dataset client...")
    conn, addr = server.accept()
    print(f"[Server] Client connected from {addr}")

    while not rospy.is_shutdown():
        try:
            msg = socket_recv(conn)
            if msg is None:
                break
            print("[Server] Received pose to execute")

            gripper_pose_table = msg['gripper']  # (7,) float32
            pos, quat = transform_pose(listener, gripper_pose_table)

            target_pose = geometry_msgs.msg.Pose()
            target_pose.position.x, target_pose.position.y, target_pose.position.z = pos
            target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w = quat

            move_group.set_pose_target(target_pose)
            move_group.go(wait=True)
            move_group.stop()
            move_group.clear_pose_targets()

            print("[Server] Movement complete.")

        except Exception as e:
            print("[Server] Error:", e)
            break

    conn.close()
    moveit_commander.roscpp_shutdown()
    print("[Server] Closed")

if __name__ == "__main__":
    main()
