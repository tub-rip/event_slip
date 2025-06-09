import os
import numpy as np
import scipy.io
from subprocess import Popen
import multiprocessing
import csv
import cv2
import math
import random
import time


generate_vis_frames_and_vids = True
create_frame_vid = True
create_frame_event_vid = True

slip_threshold_degree = 1.0 # 1.0
non_slip_threshold_degree = slip_threshold_degree
non_slip_threshold_degree = 0.1
max_data_size = math.inf

set_path = "/home/thilo/workspace/data/evaluation_sets"
set_name = "data001_train_randComb_huge1000_v2e_thres15_final" #"data001_test_randComb_huge200_v2e_thres15/" #"mix_01_2textures" #"data064_basic_medium_v2e_thres15" #"data050_v2e_thres12" # "data051_v2e_thres15"
data_collection_path = f"{set_path}/{set_name}"
output_path = f"/home/thilo/workspace/data/vt_snn/raw_data/raw_{set_name}_adClearSep{non_slip_threshold_degree}-{slip_threshold_degree}_size2x{max_data_size}_test"
os.makedirs(output_path, exist_ok=True)
os.makedirs(f"{output_path}/prophesee_recordings", exist_ok=True)
vid_path = f"{output_path}/vids"

def create_frames_and_vid(source_path, output_path, file_name, dir, event_dict, ts_list, create_frame_vid = False, create_frame_event_vid = False):  
    
    def frames_vid(source_path, output_path, file_name, dir, event_dict, ts_list):
        ffmpeg_path = "/home/thilo/workspace/ffmpeg-6.1-amd64-static/"
        frames_source_path = f"{source_path}/data/frames"
        
        frames_output_path = f"{output_path}/frames"
        vid_frames_output_path = f"{output_path}/vid_frames"
        vid_output_path = f"{output_path}/vids"
        os.makedirs(frames_output_path, exist_ok=True)
        os.makedirs(vid_frames_output_path, exist_ok=True)
        os.makedirs(vid_output_path, exist_ok=True)
        frames_output_path_with_events = f"{output_path}/frames_with_events"
        vid_frames_output_path_with_events = f"{output_path}/vid_frames_with_events"
        vid_output_path_with_events = f"{output_path}/vids_with_events"
        os.makedirs(frames_output_path_with_events, exist_ok=True)
        os.makedirs(vid_frames_output_path_with_events, exist_ok=True)
        os.makedirs(vid_output_path_with_events, exist_ok=True)
        
        vid_list_path = f"{output_path}/vids/{file_name}_{dir}.txt"
        vid_list_path_with_events = f"{output_path}/vids_with_events/{file_name}_{dir}.txt"
        vid_name = f"{file_name}_{dir}.mp4"
        frame_name = f"{file_name}_{dir}.png"
        frames_shape = (346, 260) # (250, 200)
        events_index = 0
        last_frame = cv2.imread(f'{frames_source_path}/frame_{int(ts_list[-1]):011}.png', cv2.IMREAD_GRAYSCALE)
        last_frame = cv2.resize(last_frame, frames_shape)
        last_frame = cv2.cvtColor(last_frame,cv2.COLOR_GRAY2RGB)
        last_frame = last_frame[10:, 73:273]
        cv2.imwrite(f'{frames_output_path}/{frame_name}', last_frame)
        with open(vid_list_path, "w") as vid_list:
            with open(vid_list_path_with_events, "w") as vid_list_with_events:
                # counter = 0
                for ts in ts_list:
                    frame_img = cv2.imread(f'{frames_source_path}/frame_{int(ts):011}.png', cv2.IMREAD_GRAYSCALE)
                    frame_img = cv2.resize(frame_img, frames_shape)
                    frame_img = cv2.cvtColor(frame_img,cv2.COLOR_GRAY2RGB)
                    frame_img = frame_img[10:, 73:273] # ROI size (200,250)
                    cv2.imwrite(f'{vid_frames_output_path}/{file_name}_{dir}_ts{int(ts):011}.png', frame_img)
                    vid_list.write(f"file '{vid_frames_output_path}/{file_name}_{dir}_ts{int(ts):011}.png'\n")
                    for event in zip(event_dict["ts"][events_index:], event_dict["x"][events_index:], event_dict["y"][events_index:], event_dict["p"][events_index:]):
                        if event[0] > ts/1000000000:
                            break
                        if event[2] >= 10 and event[1] >= 73 and event[1] < 273:
                            pol_value = [255, 0, 0] if event[3] == 1 else [0,0, 255]
                            frame_img[event[2] - 10, event[1] - 73] = pol_value
                            last_frame[event[2] - 10, event[1] - 73] = pol_value
                        events_index += 1
                    cv2.imwrite(f'{vid_frames_output_path_with_events}/{file_name}_{dir}_ts{int(ts):011}.png', frame_img)
                    vid_list_with_events.write(f"file '{vid_frames_output_path_with_events}/{file_name}_{dir}_ts{int(ts):011}.png'\n")
                    # counter += 1
                vid_list_with_events.close()
            vid_list.close()
                
        cv2.imwrite(f'{frames_output_path_with_events}/{frame_name}', last_frame)
        cmd = [f'{ffmpeg_path}/ffmpeg', '-f', 'concat', '-safe', '0', '-i', f'{vid_list_path}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{vid_output_path}/{vid_name}']
        cmd_with_events = [f'{ffmpeg_path}/ffmpeg', '-f', 'concat', '-safe', '0', '-i', f'{vid_list_path_with_events}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{vid_output_path_with_events}/{vid_name}']
        if create_frame_vid:
            p01 = Popen(cmd)
        if create_frame_event_vid:
            p02 = Popen(cmd_with_events)
        if create_frame_vid:
            p01.wait()
        if create_frame_event_vid:
            p02.wait()
    vid_process = multiprocessing.Process(target=frames_vid ,args=([source_path, output_path, file_name, dir, event_dict, ts_list]))
    return vid_process
    # vid_process.start()

processes = []
count_slip = 1
count_no_slip = 1
count_unclear = 1
for dir in os.listdir(data_collection_path):
    try:
        with open(f"{data_collection_path}/{dir}/data/isaac_stats.csv") as stats_file:
            events_file = open(f"{data_collection_path}/{dir}/data/events_v2e.txt")
            events_lines = events_file.readlines()[6:]
            position_index = 0
            init_ang_diff = 0
            frame_counter = 0
            frames_ts_list = []
            local_ang_diff = 0.0
            current_stats_line = None
            chunk_size = 10 # collecting data over 10 frames aka chunks with length of appr. 0.16sec
            for stats_line in stats_file.readlines()[2:]: # skipping header row and also first row of data.
                # Framerate of 60/1 aka 0.016
                if len(events_lines) == 0:
                    break
                stats_line = stats_line.split(",")
                current_stats_line = stats_line
                frames_ts_list.append(float(stats_line[0]))
                if frame_counter < chunk_size - 1:
                    
                    current_ang_diff = float(stats_line[5]) - init_ang_diff
                    local_ang_diff = current_ang_diff if current_ang_diff > local_ang_diff else local_ang_diff
                    frame_counter += 1
                else:
                    frame_counter = 0
                    init_ang_diff = float(stats_line[5])
                    global_ang_diff = float(stats_line[5])
                    is_slip = None
                    if abs(local_ang_diff) < non_slip_threshold_degree:
                        is_slip = False
                    if abs(local_ang_diff) >= slip_threshold_degree:
                        is_slip = True
                    ts = []
                    x = []
                    y = []
                    polarity = []
                    for events_line in events_lines[position_index:]:
                        events_line_list = events_line.split(" ")
                        if float(events_line_list[0]) > float(stats_line[0])/1000000000:
                            break
                        position_index += 1
                        ts.append(float(events_line_list[0])) # ts
                        x.append(int(events_line_list[1])) # x
                        y.append(int(events_line_list[2])) # y
                        polarity.append(1 if int(events_line_list[3]) == 1 else -1) # polarity (0,1) to (-1,1)
                    if len(ts) == 0:
                        break
                    first_timestamp = ts[0]
                    ts =  [timestamp - first_timestamp for timestamp in ts]
                    mat_name = f'rotate_{count_slip:02}_td' if is_slip else f'stable_{count_no_slip:02}_td'
                    frame_src = f'{data_collection_path}/{dir}/data/frames_contours_events/frame_contours_events_{int(stats_line[0]):012}.png'
                    event_dict = {
                        "ts": ts,
                        "x": x,
                        "y": y,
                        "p": polarity
                    }
                    with open(f'{output_path}/prophesee_recordings/{mat_name}.csv', 'w') as event_csv:
                        writer = csv.writer(event_csv)
                        writer.writerows(zip(ts,x,y,polarity))
                    if not is_slip is None:
                        if is_slip:
                            if count_slip <= max_data_size:
                                new_process = create_frames_and_vid(f"{data_collection_path}/{dir}", output_path, f"rotate_{count_slip:02}_ts{float(stats_line[0])/1000000000:.11f}_localAD{abs(local_ang_diff):.5f}", dir, event_dict, frames_ts_list, create_frame_vid, create_frame_event_vid)
                                processes.append(new_process)
                                scipy.io.savemat(f'{output_path}/prophesee_recordings/{mat_name}.mat', mdict={'td_data': event_dict})
                            count_slip += 1
                        else:
                            if count_no_slip <= max_data_size:
                                new_process = create_frames_and_vid(f"{data_collection_path}/{dir}", output_path, f"stable_{count_no_slip:02}_ts{float(stats_line[0])/1000000000:.11f}_localAD{abs(local_ang_diff):.5f}", dir, event_dict, frames_ts_list, create_frame_vid, create_frame_event_vid)
                                processes.append(new_process)
                                scipy.io.savemat(f'{output_path}/prophesee_recordings/{mat_name}.mat', mdict={'td_data': event_dict})
                            count_no_slip += 1
                    else:
                        new_process = create_frames_and_vid(f"{data_collection_path}/{dir}", output_path, f"unclear_{count_unclear:02}_ts{float(stats_line[0])/1000000000:.11f}_localAD{abs(local_ang_diff):.5f}", dir, event_dict, frames_ts_list, create_frame_vid, create_frame_event_vid)
                        processes.append(new_process)
                        count_unclear += 1
                    frames_ts_list = []
                    local_ang_diff = 0.0
            if count_slip > max_data_size and count_no_slip > max_data_size:
                break
            
    except Exception as err:
        print('EXCEPTION: could not open file - ', err)

if generate_vis_frames_and_vids:
    batch_size = 1
    processes_in_execution = []
    for i, p in enumerate(processes):
        processes_in_execution.append(p)
        p.start()
        if len(processes_in_execution) == batch_size or i == (len(processes) - 1):
            keep_running = True
            while keep_running:
                counter_x = 0
                for process in processes_in_execution:
                    print('checking process', counter_x)
                    counter_x += 1
                    if not process.exitcode is None:
                        print('process finished, starting new one. processes_in_execution: ', len(processes_in_execution))
                        processes_in_execution.remove(process)
                        process.terminate()
                        process.join()
                        keep_running = False
                        break
                    else:
                        time.sleep(1)
        time.sleep(1)


stable_selection = random.sample(range(1, count_no_slip), count_slip-1)
counter = 1
# print("stable_selection", stable_selection)
for dir in os.listdir(f"{output_path}/prophesee_recordings/"):
    dir_list = dir.split("_")
    if dir_list[0] == "stable":
        if not int(dir_list[1]) in stable_selection:
            os.remove(f"{output_path}/prophesee_recordings/{dir}")
        else:
            os.rename(f"{output_path}/prophesee_recordings/{dir}", f"{output_path}/prophesee_recordings/tmp_stable_{counter:02}_td.mat")
            counter += 1
            
for dir in os.listdir(f"{output_path}/prophesee_recordings/"):
    dir_list = dir.split("_")
    if dir_list[0] == "tmp":
        os.rename(f"{output_path}/prophesee_recordings/{dir}", f"{output_path}/prophesee_recordings/{dir_list[1]}_{dir_list[2]}_{dir_list[3]}")