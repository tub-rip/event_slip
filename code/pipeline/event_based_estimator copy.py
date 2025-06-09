import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from subprocess import Popen
import os

class Estimator():
    
    bin_size=0.01 * 1000000000 # default
    path_ffmpeg = "/home/thilo/workspace/ffmpeg-6.1-amd64-static/"
    frames_contours_events_folder_name = "frames_contours_events"
    vid_name = 'vid_frames_contours_events.mp4'
    
    def __init__(self, events = None, mask = None, output_folder_path = None, events_input_path = None, mask_input_path = None, ang_diff = None, path_frames_file = None, path_frames_folder = None, frames_shape = (346, 260)) -> None:
        self.events = events if not events is None else self.load_events(events_input_path)
        self.mask_img = mask if not mask is None else self.load_mask(mask_input_path)
        self.frames, self.frames_ts = self.load_frames(path_frames_file, path_frames_folder, frames_shape)
        if not ang_diff is None:
            self.ang_diff, self.ang_diff_ts = ang_diff if not type(ang_diff) is str else self.load_ang_diff(ang_diff)
        self.output_folder_path = output_folder_path
        print('mask_input_path', type(self.mask_img), self.mask_img.shape)
        
        self.mask_generator()
        self.event_to_events_sorter()
        self.create_plot()
        self.create_video()
        
    def load_events(self, events_input_path):
        events = []
        # print('events')
        with open(events_input_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                events.append((row))
        return np.array(events[7:])
    
    def load_ang_diff(self, ang_diff_path):
        ang_diff = []
        ts = []
        with open(ang_diff_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            i = header.index('angular_difference')
            j = header.index('ts')
            print('header', i, type(header), header)
            for row in reader:
                ang_diff.append(row[i])
                ts.append(row[j])
        return np.array(ang_diff).astype(float), (np.array(ts).astype(float)*1000000000).astype(int)
    
    def load_mask(self, mask_input_path):
        return cv2.imread(mask_input_path, cv2.IMREAD_GRAYSCALE)
    
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

    def mask_generator(self, mask_contour_thickness=20):
        ### Generate mask contours ###
        # contours_mask, _ = cv2.findContours(image=self.mask_img, mode=cv2.RETR_FLOODFIL, method=cv2.CHAIN_APPROX_NONE)
        contours_mask, _ = cv2.findContours(image=self.mask_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        contours_mask_img = np.zeros((self.mask_img.shape[0],self.mask_img.shape[1]), dtype=np.uint8)
        contours_mask_img = cv2.drawContours(image=contours_mask_img, contours=contours_mask, contourIdx=-1, color=255, thickness=mask_contour_thickness, lineType=cv2.LINE_AA)
        # cv2.imwrite(os.path.join(path_interval, frame_name + '_mask_base_contours.png'), contours_mask_img)

        ### Generate base mask inner and outer border ###
        self.mask_inner_border_img = contours_mask_img & self.mask_img
        self.mask_outer_border_img = contours_mask_img & np.invert(self.mask_img)
        # cv2.imwrite(os.path.join(image_folder_path, frame_name + '_mask_base_contours_inner_border.png'), contours_mask_base_inner_border_img)
        
    def event_to_events_sorter(self):
        print('mask_img', type(self.mask_img), len(self.mask_img), self.mask_img.shape)
        self.events_inside = []
        self.events_outside = []
        self.events_inner_border = []
        self.events_outer_border = []
        # events_mask_extended_inside = []
        # events_mask_extended_inner_border = []
        self.events_ts = (self.events[:,0].astype(float)*1000000000).astype(int)
        self.events_coords_x = self.events[:,1].astype(int)
        self.events_coords_y = self.events[:,2].astype(int)
        print('max x', max(self.events_coords_x))
        print('min x', min(self.events_coords_x))
        print('max y', max(self.events_coords_y))
        print('min y', min(self.events_coords_y))
        self.events_pol = self.events[:,3].astype(int)
        
        # test_len = int(len(self.events_ts)/128)
        # self.events_ts = self.events_ts[:test_len]
        # self.events_coords_x = self.events_coords_x[:test_len]
        # self.events_coords_y = self.events_coords_y[:test_len]
        
        for x,y,t in zip(self.events_coords_x, self.events_coords_y, self.events_ts):
            if self.mask_img[y][x]:
                self.events_inside.append(t)
            else:
                self.events_outside.append(t)
            if self.mask_inner_border_img[y][x]:
                self.events_inner_border.append(t)
            if self.mask_outer_border_img[y][x]:
                self.events_outer_border.append(t)
            # if self.mask_extended_img[y][x]:
            #     events_mask_extended_inside.append(t)
            # if self.mask_extended_inner_border_img[y][x]:
            #     events_mask_extended_inner_border.append(t)
            
    def setup_axis(self, axis, bins, title = None, xlabel = None, ylabel = None, legend = False):
        if not xlabel is None: axis.set_xlabel(xlabel)
        if not ylabel is None: axis.set_ylabel(ylabel)
        if not title is None: axis.title.set_text(title)
        if legend: axis.legend()
        axis.grid(axis="y", color="0.95")
        print('A05')
        # axis.set_xticks(np.arange(bins[0], bins[-1]+.05, 0.1))
        # print('A06')
        # axis.set_xticks(np.arange(bins[0], bins[-1]+.05, .01), minor=True)
        # print('A07')
        # axis.grid(which='major')
        # print('A08')
        # axis.grid(which='minor', linestyle='--', alpha=0.5)
        # print('A09')
        
    def create_plot(self):
        fig, axis = plt.subplots(nrows=2, ncols=1, figsize=(20,20))
        xlabel = 'Time in seconds'
        ylabel = 'Events per second'
        
    
        self.num_bins = math.floor(self.events_ts[-1]/self.bin_size if self.bin_size != 0 else self.events_ts[-1])
        print('num_bins', self.num_bins)
        print('self.events_ts', type(self.events_ts), len(self.events_ts))
        (counts_all, bins_all) = np.histogram(self.events_ts, bins=self.num_bins)
        print('counts_all', len(counts_all), 'bins_all', len(bins_all))
        ### Total events inner and outer border ###
        (counts_inner_border, bins_inner_border) = np.histogram(self.events_inner_border, bins=self.num_bins)
        axis[0].plot(bins_all[:-1], (counts_inner_border/self.bin_size/1000), label="Inner border events")
        (counts_outer_border, bins_outer_border) = np.histogram(self.events_outer_border, bins=self.num_bins)
        axis[0].plot(bins_all[:-1], (counts_outer_border/self.bin_size/1000), label="Outer border events")
        self.setup_axis(axis[0], bins_all, 'Total events inner and outer border', xlabel, ylabel, True)
        
        # Angular difference plot is available
        if not self.ang_diff is None:
            print('plotting ang_diff', len(self.ang_diff), len(self.ang_diff_ts))
            axis[1].plot(self.ang_diff_ts, self.ang_diff, marker=".", linestyle="", markersize="1")
            self.setup_axis(axis[1], self.ang_diff_ts, 'Angular Difference', 'Time in seconds', 'degree', False)
        
        fig.savefig(f"{self.output_folder_path}/plot_borders_total.png")
        
        
    def create_video(self):
        self.frames, self.frames_ts
        if self.frames is None or self.frames_ts is None:
            print('Missing frames or frames_ts to create video')
            return
        os.makedirs(f'{self.output_folder_path}/{self.frames_contours_events_folder_name}', exist_ok=True)
        self.frame_with_mask_and_events = []
        counter_events_ts = 0
        event_ts_current = self.events_ts[counter_events_ts]
        frames_ts_next = self.frames_ts[1:]
        frames_ts_next.append(float('inf'))
        counter = 0
        mask_contour_outer_border, _ = cv2.findContours(image=self.mask_outer_border_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        mask_contour_inner_border, _ = cv2.findContours(image=self.mask_inner_border_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        for frame, frame_ts, frame_ts_next in zip(self.frames, self.frames_ts, frames_ts_next):
            
            frame = cv2.drawContours(image=frame, contours=mask_contour_outer_border, contourIdx=-1, color=(127, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            frame = cv2.drawContours(image=frame, contours=mask_contour_inner_border, contourIdx=-1, color=(255, 127, 255), thickness=1, lineType=cv2.LINE_AA)
            frame_shape = frame.shape
            while event_ts_current < frame_ts_next and counter_events_ts < len(self.events_ts):
                event_ts_current = self.events_ts[counter_events_ts]
                if frame_shape[1] > self.events_coords_x[counter_events_ts] and frame_shape[0] > self.events_coords_y[counter_events_ts]:
                    value = [255, 0, 0] if self.events_pol[counter_events_ts] else [0,0, 255]
                    a = self.events_coords_x[counter_events_ts]
                    b = self.events_coords_y[counter_events_ts]
                    frame[b][a] = value 
                else:
                    print('frame_shape', frame_shape)
                    print('ELSE', self.events_coords_x[counter_events_ts], self.events_coords_y[counter_events_ts])
                counter_events_ts += 1
            # self.frame_with_mask_and_events.append(frame)
            # counter += 1
            cv2.imwrite(f'{self.output_folder_path}/{self.frames_contours_events_folder_name}/frame_contours_events_{int(frame_ts):012}.png', frame)
            
        cmd = [f'{self.path_ffmpeg}/ffmpeg', '-y', '-framerate', '60', '-pattern_type', 'glob', '-i', f'{self.output_folder_path}/{self.frames_contours_events_folder_name}/frame_contours_events_*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{self.output_folder_path}/{self.vid_name}']
        p = Popen(cmd)
        p.wait()