# socket_demo_server.py
import socket
import msgpack
import msgpack_numpy
import numpy as np
import time
msgpack_numpy.patch()

def create_fake_obs():
    return {
        'xyz': np.random.rand(1280*960, 3).astype(np.float32),
        'gripper': np.random.rand(7).astype(np.float32),
        'joint_states': np.random.rand(6).astype(np.float32),
        'target_pose': np.random.rand(7).astype(np.float32),
    }

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

def main(host='127.0.0.1', port=5005):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[Server] Listening on {host}:{port}")
    conn, addr = server.accept()
    print(f"[Server] Client connected from {addr}")

    for i in range(5):
        obs = create_fake_obs()
        send_msg(conn, obs)
        print(f"[Server] Sent observation {i+1}")

        try:
            response = recv_response(conn)
            print(f"[Server] Got response: {response}")
        except Exception as e:
            print(f"[Server] Error receiving response: {e}")
            break

        time.sleep(1)

    conn.close()
    print("[Server] Closed connection")

if __name__ == '__main__':
    main()
