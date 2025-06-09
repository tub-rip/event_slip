    
import socket
import json
import argparse
import numpy as np


# print('ACCESSED sim_client_socket.py')


parser = argparse.ArgumentParser()
parser.add_argument('--ip', default='127.0.0.1')
parser.add_argument('--port', default=9898)
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((args.ip, int(args.port)))
print('CONNECTION ESTABLISHED', args.port)
settings_len_bytes = sock.recv(5)
settings_len = int.from_bytes(settings_len_bytes, "big")

chunks = []
bytes_recd = 0
while bytes_recd < settings_len:
    chunk = sock.recv(min(settings_len - bytes_recd, 2048))
    if chunk == b'':
        raise RuntimeError("socket connection broken")
    chunks.append(chunk)
    bytes_recd = bytes_recd + len(chunk)
settings_bytes =  b''.join(chunks)
settings = json.loads(settings_bytes)


def start_isaac_sim():
    import simulator_docker as simulator
    # isaac_sim = simulator.IsaacSim()
    isaac_sim = simulator.IsaacSim(socket=sock, settings=settings)
    isaac_sim.run_simulation()
    # isaac_sim.turn_off()

print('before start_isaac_sim', flush=True) 
start_isaac_sim()

