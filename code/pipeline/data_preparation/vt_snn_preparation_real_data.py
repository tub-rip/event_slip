import os
import numpy as np
import scipy.io
from subprocess import Popen
import multiprocessing
import copy
import cv2
import math
import random
import csv


generate_vis_frames_and_vids = True
create_frame_vid = False
create_frame_event_vid = False

slip_threshold_degree = 1.0 # 1.0
non_slip_threshold_degree = slip_threshold_degree
non_slip_threshold_degree = 0.1
max_data_size = math.inf

# set_path = "/home/thilo/workspace/data/evaluation_sets"
# set_name = "data001_test_randComb_huge200_v2e_thres15" #"data001_train_randComb_huge1000_v2e_thres15_final" #"data001_test_randComb_huge200_v2e_thres15/" #"mix_01_2textures" #"data064_basic_medium_v2e_thres15" #"data050_v2e_thres12" # "data051_v2e_thres15"
# set_path = "/home/thilo/workspace/data/real_data/"
set_path = "/home/thilo/workspace/data/real_data/extracted/"
set_name = "ready_to_preprocess" #"data001_train_randComb_huge1000_v2e_thres15_final" #"data001_test_randComb_huge200_v2e_thres15/" #"mix_01_2textures" #"data064_basic_medium_v2e_thres15" #"data050_v2e_thres12" # "data051_v2e_thres15"
data_collection_path = f"{set_path}/{set_name}"
output_path = f"/home/thilo/workspace/data/vt_snn/raw_data/raw_{set_name}_adClearSep{non_slip_threshold_degree}-{slip_threshold_degree}_size2x{max_data_size}"
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
count_intervals = 1
counter_slips = 0
counter_stable = 0
for dir in os.listdir(data_collection_path):
    try:
        # events_file = open(f"{data_collection_path}/{dir}/data/events_labeled.csv")
        # events_lines = events_file.readlines()[6:]
        with open(f"{data_collection_path}/{dir}/data/events_labeled.csv") as f:
            # line = f.readlines()
            rows = [list(map(float, line.strip().split(','))) for line in f.readlines()[1:]]
            # events_lines = line[1:]
            events_lines = [row for row in rows if row[-1] != 0]
            first_ts = events_lines[0]
            # print('first_ts', first_ts[0])
            # print('events_lines', events_lines[0][0])
            events_lines = [(x - first_ts[0],y,z,a,b,c) for x,y,z,a,b,c in events_lines]
            # print('events_lines', type(events_lines), type(events_lines[0]), events_lines[0])
            position_index = 0
            frame_counter = 0
            frames_ts_list = []
            begin_events_interval = 0
            end_events_interval = 0.16 # this is 0.16 seconds in nanosecons #160000000 #160000000 #160000000
            events_interval_size = 0.16 # this is 0.16 seconds in nanosecons # 160000000 #160000000 #160000000
            
            # for x in events_lines:
            #     print('x', type(x), x )
            #     print('x[0]', type(x[0]), x[0])
            #     print('x[0] - first_ts[0]', type(x[0] - first_ts[0]), x[0] - first_ts[0])
                
            # for stats_line in stats_file.readlines()[1:]: # skipping header row
            #     # Framerate of 60/1 aka 0.016
            #     if len(events_lines) == 0:
            #         break
            #     stats_line = stats_line.split(",")
            #     current_stats_line = stats_line
            #     frames_ts_list.append(float(stats_line[0]))
            #     if frame_counter < chunk_size - 1:
            #         frame_counter += 1
            #     else:
            frame_counter = 0
            # counter_slips = 0
            # counter_stable = 0
            while not position_index > len(events_lines):
                # print('events_lines len vs position_index', position_index, len(events_lines))
                ts = []
                x = []
                y = []
                polarity = []
                slice_events = 0
                slice_slip_events = 0
                # print('events_lines[position_index:]', events_lines[0][0])
                # print('events_lines[position_index:]', events_lines[position_index].split(",")[0])
                # print('(event_lines)', (events_lines))
                for events_line in events_lines[position_index:]:
                    events_line_list = events_line
                    # print("events_line_list", events_line_list[0])
                    if events_line_list[0] > end_events_interval:
                        print('###########')
                        print('event_line_list[0]', events_line_list[0]) 
                        print("end_events_interval", end_events_interval)
                        # print('break because next slice')
                        begin_events_interval += events_interval_size
                        end_events_interval += events_interval_size
                        # print('begin_events_interval', float(begin_events_interval/10000000000))
                        # print('end_events_interval', float(end_events_interval/10000000000))
                        break
                    position_index += 1
                    ts.append(float(events_line_list[0])) # ts
                    x.append(int(events_line_list[1])) # x
                    y.append(int(events_line_list[2])) # y
                    polarity.append(1 if int(events_line_list[3]) == 1 else -1) # polarity (0,1) to (-1,1)
                    slice_events += 1
                    slice_slip_events += int(events_line_list[4])
                if not len(ts) == 0:
                    # first_timestamp = ts[0]
                    slice_slip_percentage = (slice_slip_events/slice_events)*100
                    #print('slice_slip_percentage', slice_slip_percentage)
                    slip = False
                    if slice_slip_percentage > 20:
                        # print('SLIP')
                        counter_slips += 1
                        slip = True
                    else:
                        # print('STABLE')
                        counter_stable += 1
                    #print('counter_slips', counter_slips)
                    #print('counter_stable', counter_stable)
                    # ts =  [timestamp - first_timestamp for timestamp in ts]
                    mat_name = f'rotate_{counter_slips:02}_td' if slip else f'stable_{counter_stable:02}_td'
                    count_intervals += 1

                    event_dict = {
                        "ts": ts,
                        "x": x,
                        "y": [259-elem for elem in y],
                        "p": polarity
                    }
                    # print('last ts', ts[-1])
                    # with open(f'{output_path}/prophesee_recordings/{mat_name}.csv', 'w') as event_csv:
                    #     writer = csv.writer(event_csv)
                    #     writer.writerows(zip(ts,x,y,polarity))
                    # new_process = create_frames_and_vid(f"{data_collection_path}/{dir}", output_path, f"real_{count_intervals:02}_ts{float(stats_line[0])/1000000000:.11f}_localAD{abs(local_ang_diff):.5f}", dir, event_dict, frames_ts_list, create_frame_vid, create_frame_event_vid)
                    # processes.append(new_process)
                    scipy.io.savemat(f'{output_path}/prophesee_recordings/{mat_name}.mat', mdict={'td_data': event_dict})
                    frames_ts_list = []
                else:
                    position_index += 1
            
    except Exception as err:
        print('EXCEPTION: could not open file - ', err)

# if generate_vis_frames_and_vids:
#     batch_size = 1
#     processes_in_execution = []
#     for i, p in enumerate(processes):
#         processes_in_execution.append(p)
#         p.start()
#         if len(processes_in_execution) == batch_size or i == (len(processes) - 1):
#             keep_running = True
#             while keep_running:
#                 counter_x = 0
#                 for process in processes_in_execution:
#                     print('checking process', counter_x)
#                     counter_x += 1
#                     if not process.exitcode is None:
#                         print('process finished, starting new one. processes_in_execution: ', len(processes_in_execution))
#                         processes_in_execution.remove(process)
#                         process.terminate()
#                         process.join()
#                         keep_running = False
#                         break
#                     else:
#                         time.sleep(1)
#         time.sleep(1)

print('Counter SLIPS:', counter_slips)
print('Counter STABLE:', counter_stable)
stable_selection = random.sample(range(1, counter_stable), counter_slips)
counter = 1
print("stable_selection  len", len(stable_selection))
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