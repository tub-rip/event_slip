import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from subprocess import Popen
import os

class EventVidGenerator():
    
    bin_size=0.01 * 1000000000 # default
    path_ffmpeg = "/home/thilo/workspace/ffmpeg-6.1-amd64-static/"
    frames_and_events_folder_name = "frames_and_events"
    vid_name = 'vid_frames_and_events.mp4'
    
    def __init__(self, events = None, output_folder_path = None, events_input_path = None, path_frames_file = None, path_frames_folder = None, frames_shape = (346, 260)) -> None:
        self.events = events if not events is None else self.load_events(events_input_path)
        self.events_ts = (self.events[:,0].astype(float)*1000000000).astype(int)
        self.events_coords_x = self.events[:,1].astype(int)
        self.events_coords_y = self.events[:,2].astype(int)
        self.events_pol = self.events[:,3].astype(int)
        self.frames, self.frames_ts = self.load_frames(path_frames_file, path_frames_folder, frames_shape)
        self.output_folder_path = output_folder_path
        self.create_video()
        
    def load_events(self, events_input_path):
        events = []
        # print('events')
        with open(events_input_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                events.append((row))
        return np.array(events[7:])
    
    
    def load_frames(self, path_frames_file, path_frames_folder, frames_shape):
        if path_frames_file is None or path_frames_folder is None:
            return None, None
        frames = []
        ts = []
        with open(path_frames_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            i = header.index('frames')
            j = header.index('ts')
            for row in reader:
                frame_img = cv2.imread(f'{path_frames_folder}/{row[i]}', cv2.IMREAD_GRAYSCALE)
                frame_img = cv2.resize(frame_img, frames_shape)
                frame_img = cv2.cvtColor(frame_img,cv2.COLOR_GRAY2RGB)
                frames.append(frame_img)
                ts.append(row[j])
        return frames, list((np.array(ts).astype(float)).astype(int))
        
    # def create_video(self):
    #     self.frames, self.frames_ts
    #     if self.frames is None or self.frames_ts is None:
    #         print('Missing frames or frames_ts to create video')
    #         return
    #     os.makedirs(f'{self.output_folder_path}/{self.frames_and_events_folder_name}', exist_ok=True)
    #     counter_events_ts = 0
    #     event_ts_current = self.events_ts[counter_events_ts]
    #     frames_ts_next = self.frames_ts[1:]
    #     frames_ts_next.append(float('inf'))
    #     print('first event ts', event_ts_current)
    #     print('first frames ts', self.frames_ts[0])
    #     print('first event earlier then first event ts?', event_ts_current <= self.frames_ts[0])
    #     # print('first frames_next ts', frames_ts_next[0])
    #     print('last frames ts', self.frames_ts[-1])
    #     print('last event ts', self.events_ts[-1])
    #     print('last event earlier then last event ts?', self.events_ts[-1] <= frames_ts_next[-1])
    #     for frame, frame_ts, frame_ts_next in zip(self.frames, self.frames_ts, frames_ts_next):
    #         frame_shape = frame.shape
    #         # if frame_ts == 400000020:
    #         while event_ts_current < frame_ts-3*16666668 and counter_events_ts < len(self.events_ts):
    #         # while event_ts_current < frame_ts_next and counter_events_ts < len(self.events_ts):
    #             if frame_shape[1] > self.events_coords_x[counter_events_ts] and frame_shape[0] > self.events_coords_y[counter_events_ts]:
    #                 # if event_ts_current > 333333350 and event_ts_current <= 350000018: # prev ts: 1:383333353 2:366666685 3:350000018 4:333333350
    #                 value = [255, 0, 0] if self.events_pol[counter_events_ts] else [0,0, 255]
    #                 a = self.events_coords_x[counter_events_ts]
    #                 b = self.events_coords_y[counter_events_ts]
    #                 frame[b][a] = value 
    #             else:
    #                 print('frame_shape', frame_shape)
    #                 print('ELSE', self.events_coords_x[counter_events_ts], self.events_coords_y[counter_events_ts])
    #             counter_events_ts += 1
    #             event_ts_current = self.events_ts[counter_events_ts]
    #             cv2.imwrite(f'{self.output_folder_path}/{self.frames_and_events_folder_name}/frame_and_events_{int(frame_ts):012}.png', frame)
            
    #     cmd = [f'{self.path_ffmpeg}/ffmpeg', '-y', '-framerate', '60', '-pattern_type', 'glob', '-i', f'{self.output_folder_path}/{self.frames_and_events_folder_name}/frame_and_events_*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{self.output_folder_path}/{self.vid_name}']
    #     p = Popen(cmd)
    #     p.wait()
        
    def create_video(self):
        self.frames, self.frames_ts
        if self.frames is None or self.frames_ts is None:
            print('Missing frames or frames_ts to create video')
            return
        os.makedirs(f'{self.output_folder_path}/{self.frames_and_events_folder_name}', exist_ok=True)
        y = self.frames_ts[0]
        # self.frames_ts = [((x - y) + 16666668) for x in self.frames_ts]
        # self.frames_ts = self.frames_ts - self.frames_ts[0] + 16666668
        counter_events_ts = 0
        event_ts_current = self.events_ts[counter_events_ts]
        frames_ts_next = self.frames_ts[1:]
        frames_ts_next.append(float('inf'))
        print('first event ts', event_ts_current)
        print('first frames ts', self.frames_ts[0])
        print('first event earlier then first event ts?', event_ts_current <= self.frames_ts[0])
        # print('first frames_next ts', frames_ts_next[0])
        print('last frames ts', self.frames_ts[-1])
        print('last event ts', self.events_ts[-1])
        print('last event earlier then last event ts?', self.events_ts[-1] <= frames_ts_next[-1])
        for frame, frame_ts, frame_ts_next in zip(self.frames, self.frames_ts, frames_ts_next):
            frame_shape = frame.shape
            while event_ts_current < frame_ts and counter_events_ts < len(self.events_ts):
            # while event_ts_current < frame_ts_next and counter_events_ts < len(self.events_ts):
                if frame_shape[1] > self.events_coords_x[counter_events_ts] and frame_shape[0] > self.events_coords_y[counter_events_ts]:
                    value = [255, 0, 0] if self.events_pol[counter_events_ts] else [0,0, 255]
                    a = self.events_coords_x[counter_events_ts]
                    b = self.events_coords_y[counter_events_ts]
                    frame[b][a] = value 
                else:
                    print('frame_shape', frame_shape)
                    print('ELSE', self.events_coords_x[counter_events_ts], self.events_coords_y[counter_events_ts])
                counter_events_ts += 1
                event_ts_current = self.events_ts[counter_events_ts]
            # print('frame ts:', frame_ts)
            # print('last event smaller then frame ts?', frame_ts >= self.events_ts[counter_events_ts - 1])
            # print('last_event ts:', self.events_ts[counter_events_ts - 1])
            # print('last_event ts:', event_ts_current)
            # print('last_event ts same?:', event_ts_current == self.events_ts[counter_events_ts - 1])
            # print('current event ts:', self.events_ts[counter_events_ts])
            # print('########')
            cv2.imwrite(f'{self.output_folder_path}/{self.frames_and_events_folder_name}/frame_and_events_{int(frame_ts):012}.png', frame)
            
        cmd = [f'{self.path_ffmpeg}/ffmpeg', '-y', '-framerate', '60', '-pattern_type', 'glob', '-i', f'{self.output_folder_path}/{self.frames_and_events_folder_name}/frame_and_events_*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{self.output_folder_path}/{self.vid_name}']
        p = Popen(cmd)
        p.wait()
        
if __name__ == "__main__":
    output_folder = "/home/thilo/workspace/data/evaluation_sets/data000_train_randComb_tiny1_v2e_thres15/set000_20241110-173808/data/test"
    event_vid_generator = EventVidGenerator(events_input_path=f"/home/thilo/workspace/data/evaluation_sets/data000_train_randComb_tiny1_v2e_thres15/set000_20241110-173808/data/events_v2e.txt", output_folder_path = output_folder, path_frames_file=f'/home/thilo/workspace/data/evaluation_sets/data000_train_randComb_tiny1_v2e_thres15/set000_20241110-173808/data/isaac_stats.csv', path_frames_folder="/home/thilo/workspace/data/evaluation_sets/data000_train_randComb_tiny1_v2e_thres15/set000_20241110-173808/data/frames")