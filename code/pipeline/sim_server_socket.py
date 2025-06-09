import multiprocessing
import subprocess
import socket
import json
import numpy as np
import signal

class SimServerSocket():
    
    def __init__(self, host_ip, port, docker_cmd_str, settings) -> None:
        signal.signal(signal.SIGINT, self.shutdown)
        self.host_ip = host_ip
        self.port = port
        self.docker_cmd_str = docker_cmd_str
        self.settings = settings

    def start_simulation(self):
        # multiprocessing.set_start_method("spawn")
        print('server: mp start method', multiprocessing.get_start_method())
        self.sim_process = multiprocessing.Process(target=self.sim_process_caller, args=[self.docker_cmd_str, self.port])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.host_ip, self.port))
            sock.listen()
            self.sim_process.start()
            conn, addr = sock.accept()            
            with conn:
                print('Connected by', addr)
                settings_bytes = json.dumps(self.settings).encode('utf-8')
                settings_bytes_len = len(settings_bytes).to_bytes(5, 'big')
                conn.send(settings_bytes_len)
                conn.send(settings_bytes)
                while True:
                    conn.send(b'NEXT')
                    results = []
                    first_msg = conn.recv(5)
                    if first_msg == b'END':
                        print('close server socket')
                        sock.close()
                        break
                    data_shape_0 = conn.recv(5)
                    data_shape_1 = conn.recv(5)
                    data_shape_2 = conn.recv(5)
                    frame_shape = (int.from_bytes(data_shape_0, "big"), int.from_bytes(data_shape_1, "big"), int.from_bytes(data_shape_2, "big"))
                    
                    incomming_msg_len = int.from_bytes(first_msg, "big")
                    chunks = []
                    bytes_recd = 0
                    while bytes_recd < incomming_msg_len:
                        chunk = conn.recv(min(incomming_msg_len - bytes_recd, 2048))
                        if chunk == b'':
                            raise RuntimeError("socket connection broken")
                        chunks.append(chunk)
                        bytes_recd = bytes_recd + len(chunk)
                    rgb_data_bytes =  b''.join(chunks)
                    rgb_data = np.frombuffer(rgb_data_bytes, dtype=np.uint8)
                    rgb_data = np.reshape(rgb_data, frame_shape)
                    results.append(rgb_data)
                    for elem in range(5):
                        msg_len_bytes = conn.recv(5)
                        msg_len = int.from_bytes(msg_len_bytes, "big")
                        msg_dtype_len_bytes = conn.recv(5)
                        msg_dtype_len = int.from_bytes(msg_dtype_len_bytes, "big")
                        msg_dtype_bytes = conn.recv(msg_dtype_len)
                        msg_dtype = msg_dtype_bytes.decode('utf-8')
                        chunks = []
                        bytes_recd = 0
                        while bytes_recd < msg_len:
                            chunk = conn.recv(min(msg_len - bytes_recd, 2048))
                            if chunk == b'':
                                raise RuntimeError("socket connection broken")
                            chunks.append(chunk)
                            bytes_recd = bytes_recd + len(chunk)
                        msg_bytes =  b''.join(chunks)
                        msg = np.frombuffer(msg_bytes, dtype=msg_dtype)
                        results.append(msg)
                    yield results
        self.sim_process.terminate()
        self.sim_process.join()
        # self.shutdown('a', 'b')
        print('success: isaac_sim shutdown')
            
    def sim_process_caller(self, docker_cmd_str, port):
        print('start_sim_process')
        subprocess.call(docker_cmd_str + f" --port {port}", shell=True)
        
    def shutdown(self, sig, frame):
        self.sim_process.terminate()
        self.sim_process.join()
    
