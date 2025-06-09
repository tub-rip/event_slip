import json, csv
import os
from subprocess import Popen
import multiprocessing
from multiprocessing import Pipe
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as pp

# from sam_for_frames_module import Mask
from sim_server_socket import SimServerSocket
from v2e import simulate_events
from frame_and_event_video import EventVidGenerator


class SimDataSet():
    
    frames = []
    frame_names = []
    ts = []
    gripper_pos = []
    gripper_ori = []
    cube_pos = []
    cube_ori = []
    ang_diff = []
    ang_diff2 = []
    events = None # [timestamp (float s), x, y, polarity (0,1)] (v2e)
    
    
    #settings
    settings = {}
    header_csv = ["ts", "end_effector_pos", "end_effector_ori", "cube_pos", "cube_ori", "angular_difference", "frames"]
    host_ip = "127.0.0.1"
    csv_file_name = "isaac_stats.csv"
    plot_file_name = 'ang_diff_plot.png'
    mask_file_name = '' # is set with name of used frame like framename_mask.png
    frames_vid_name = 'vid_frames.mp4'
    frames_folder_name = "frames"
    data_folder_name = "data"
    frame_ext = ".png"
    frame_number_for_mask = 0
    frame_for_mask = None
    
    ffmpeg_path = "/home/thilo/workspace/ffmpeg-6.1-amd64-static/"
    
        
    def __init__(self, settings, output_folder, port) -> None:
        self.settings = settings
        self.port = port
        self.gen_folder_structure(output_folder)        
        
        self.docker_cmd_str = "docker run --name isaac_sim_pipeline" + str(port) + " --entrypoint ./python.sh --gpus '\"'device=${CUDA_VISIBLE_DEVICES}'\"' -e \"ACCEPT_EULA=Y\" --rm --network=host     -e \"PRIVACY_CONSENT=Y\"     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw     -v ~/workspace/:/isaac-sim/workspace:rw -v ~/textures/:/isaac-sim/textures:rw isaac-sim_pipeline /isaac-sim/workspace/ma_slip_detection/isaac_sim/standalone/ma.simulations.standalone/pipeline/sim_client_socket.py"
        self.start_simulation()
        
        # self.tasks = [self.task_v2e, self.task_sam, self.task_frames_vid]
        # self.tasks = [self.task_v2e, self.task_frames_vid]
        self.tasks = [self.task_v2e]
        self.calculate()
        
        event_vid_generator = EventVidGenerator(events_input_path=f"{self.event_sim_path}/events_v2e.txt", output_folder_path = self.plot_path, path_frames_file=f'{self.csv_file_path}/{self.csv_file_name}', path_frames_folder=self.frames_folder_path)
        # event_vid_generator = EventVidGenerator(events_input_path=f"{self.event_sim_path}/events_v2e.txt", output_folder_path = self.plot_path, mask_input_path = f"{self.mask_path}/frame_00083333337_mask.png", ang_diff = f'{self.csv_file_path}/{self.csv_file_name}', path_frames_file=f'{self.csv_file_path}/{self.csv_file_name}', path_frames_folder=self.frames_folder_path)
        # estimator = Estimator(events_input_path=f'{self.event_sim_path}/events_v2e.txt', output_folder_path = self.plot_path, mask_input_path = f'{self.mask_path}/{self.mask_file_name}', ang_diff = (self.ang_diff, self.ts))
        self._save_settings()
        
    def gen_folder_structure(self, base_folder):
        self.base_folder_path = f"{base_folder}"
        os.makedirs(self.base_folder_path, exist_ok=True)
        self.data_output_path = f"{self.base_folder_path}/{self.data_folder_name}"
        os.makedirs(self.data_output_path, exist_ok=True)
        self.frames_folder_path = f"{self.data_output_path}/{self.frames_folder_name}"
        os.makedirs(self.frames_folder_path, exist_ok=True)
        
        self.csv_file_path = f"{self.data_output_path}"
        self.mask_path = f"{self.data_output_path}"
        self.plot_path = f"{self.data_output_path}"
        self.event_sim_path = f"{self.data_output_path}"
        self.frames_vid_path = f"{self.data_output_path}"
        
        print('base_folder_path', self.base_folder_path)
        print('data_output_path', self.data_output_path)
        print('frames_folder_path', self.frames_folder_path)
        print('csv_file_path', self.csv_file_path)
        print('mask_path', self.mask_path)
        print('plot_path', self.plot_path)
        print('event_sim_path', self.event_sim_path)
        
    def start_simulation(self):
        sim_server_socket = SimServerSocket(self.host_ip, self.port, self.docker_cmd_str, self.settings["isaac_sim"])
        gen = sim_server_socket.start_simulation()
        for results in gen:
            self.frames.append(results[0])
            if not hasattr(self,'first_frame_ts'):
                self.first_frame_ts = results[1][0].item()
            frame_ts = results[1][0].item() - self.first_frame_ts + 16666668 #framerate is 60Hz, v2e creates also events before first frame.
            self.ts.append(frame_ts)
            self.gripper_pos.append(results[2])
            self.gripper_ori.append(results[3]) # quat with x,y,z,w
            self.cube_pos.append(results[4])
            self.cube_ori.append(results[5])  # quat with x,y,z,w
            self.ang_diff.append(self._angular_difference(results[3], results[5]))
            self._save_frame(results[0], frame_ts)
            self._write_to_csv([[frame_ts]] + results[2:] + [[self.ang_diff[-1]], [self.frame_names[-1]]])
        # self.csv_file
        print('self.frames len', len(self.frames))
        self.plot_ang_diff_distribution()
    
    def _write_to_csv(self, row):
        # print('ROW:', row)
        if not hasattr(self, "csv_writer"):
            self.csv_file = open(f'{self.csv_file_path}/{self.csv_file_name}', 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(self.header_csv)
        row_list = []
        for elem in row:
            elem_str = ''
            for entry in elem:
                elem_str += str(entry) if len(elem_str) == 0 else f" {entry}"
            row_list.append(elem_str)
        self.csv_writer.writerow(row_list)
            
    def _save_frame(self, frame, ts):
        frame_name = f"frame_{int(ts):011}{self.frame_ext}"
        self.frame_names.append(frame_name)
        if not hasattr(self, "frames_count"):
            self.frames_count = 0
        if self.frames_count == self.frame_number_for_mask:
            self.frame_for_mask = cv2.resize(frame, (346, 260))
            filename, mask_ext = os.path.splitext(frame_name)
            self.mask_file_name = f"{filename}_mask{mask_ext}"
        self.frames_count += 1
        cv2.imwrite(f"{self.frames_folder_path}/{frame_name}", frame)
    
    def _save_settings(self):
        with open(f"{self.data_output_path}/settings.txt", "w") as file:
            file.write(json.dumps(self.settings))
            
    def task_frames_vid(self):
        cmd = [f'{self.ffmpeg_path}/ffmpeg', '-y', '-framerate', '60', '-pattern_type', 'glob', '-i', f'{self.frames_folder_path}/frame_*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{self.frames_vid_path}/{self.frames_vid_name}']
        def frames_vid(cmd):
            p = Popen(cmd)
            p.wait()
        vid_process = multiprocessing.Process(target=frames_vid ,args=([cmd]))
        vid_process.start()
        return vid_process
        
        # /home/thilo/workspace/ffmpeg-6.1-amd64-static/ffmpeg -framerate 60 -pattern_type glob -i './frames/*.png'   -c:v libx264 -pix_fmt yuv420p ./sim_vid.mp4
        
    def task_v2e(self):
        print('V2E start')
        # manager = multiprocessing.Manager()
        # self.events = manager.list()
        v2e_settings = self.settings["event_sim"][self.settings["event_sim"]["simulator"]]
        v2e_process = multiprocessing.Process(target=simulate_events,args=(v2e_settings, self.frames_folder_path, self.event_sim_path))
        v2e_process.start()
        return v2e_process

    def task_sam(self):
        print('SAM start')
        
        # frame_for_mask_path = f"{self.frames_folder_path}/frame_00083333337.png"
        # frame = cv2.imread(frame_for_mask_path)
        # frame = cv2.resize(frame, (346, 260))
        # self.frame_for_mask = frame
        # self.mask_file_name = "frame_00083333337_mask.png"
        
        # self.frame_for_mask = f"{self.frames_folder_path}/{self.frame_for_mask['frame_name']}"
        mask = Mask(self.frame_for_mask, self.mask_file_name, self.mask_path, show=False, camera_mount='sideView', checkpoint='/home/thilo/segment-anything/checkpoints/sam_vit_h_4b8939.pth')
        sam_process = multiprocessing.Process(target=mask.generate_mask, args=())
        sam_process.start()
        return sam_process
    
    def calculate(self):
        self.processes = []
        for task in self.tasks:
            self.processes.append(task())
            # self.processes.append(task(manager))
        for process in self.processes:
            process.join()
        
    def _angular_difference(self, ori_1, ori_2):
        ori_1_r = Rotation.from_quat([ori_1[0], ori_1[1], ori_1[2], ori_1[3]])
        ori_2_r = Rotation.from_quat([ori_2[0], ori_2[1], ori_2[2], ori_2[3]])
        ori_1_M = Rotation.as_matrix(ori_1_r)
        ori_2_M = Rotation.as_matrix(ori_2_r)
        # Set angular diff relative to init pose of cube and gripper
        if not hasattr(self, "R1"):
            self.R1 = ori_1_M
            self.R2 = ori_2_M
            return 0.0
        ori_1_M = ori_1_M @ self.R1.T
        ori_2_M = ori_2_M @ self.R2.T
        ang = np.zeros((ori_1_M.shape[0]))
        delR = ori_1_M[:, :].T @ ori_2_M[:, :]
        ang = abs(np.arccos((np.trace(delR) - 1) / 2) * (180.0 / np.pi))
        return ang
    
    def plot_ang_diff_distribution(self):
        pp.plot(self.ts, self.ang_diff, marker=".", linestyle="", markersize="0.5")
        pp.grid(axis="y", color="0.95")
        pp.savefig(f"{self.plot_path}/{self.plot_file_name}", dpi=1200)
        pp.clf()
    
if __name__ == "__main__":
    settings = json.load(open("/home/thilo/workspace/ma_slip_detection/isaac_sim/standalone/ma.simulations.standalone/pipeline/settings.json"))
    test_output = '/home/thilo/workspace/data/pipeline/test82'
    SimDataSet(settings, test_output, 9897)