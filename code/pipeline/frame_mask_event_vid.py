import cv2
import csv
import numpy as np
from subprocess import Popen
import multiprocessing

class Vid:
    
    vid = None
    frame_with_mask_and_events = None
    
    
    
    def __init__(self, frames, frames_ts, events_ts, events_coord_y, events_coords_x, events_pol, mask) -> None:
        
        self.frame_with_mask_and_events = []
        counter_events_ts = 0
        event_ts_current = events_ts[counter_events_ts]
        frames_ts_next = frames_ts[1:]
        frames_ts_next.append(float('inf'))
        counter = 0
        mask_contour, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        for frame, frame_ts, frame_ts_next in zip(frames, frames_ts, frames_ts_next):
            
            frame = cv2.drawContours(image=frame, contours=mask_contour, contourIdx=-1, color=(127, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            frame_shape = frame.shape
            while event_ts_current < frame_ts_next and counter_events_ts < len(events_ts):
                event_ts_current = events_ts[counter_events_ts]
                if frame_shape[0] > events_coord_y[counter_events_ts] and frame_shape[1] > events_coords_x[counter_events_ts]:
                    value = [255, 0, 0] if events_pol[counter_events_ts] else [0,0, 255]
                    a = events_coords_x[counter_events_ts]
                    b = events_coord_y[counter_events_ts]
                    frame[a][b] = value 
                counter_events_ts += 1
            self.frame_with_mask_and_events.append(frame)
            counter += 1
            
    
    def save_vid(self):
        pass
    
    def save_imgs(self):
        pass


def load_frames(path_frames_file, path_frames_folder, frames_shape):
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
    
def load_events(path_events_file):
    events = []
    with open(path_events_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            events.append((row))
    events = np.array(events[7:])
    return (events[:,0].astype(float)*1000000000).astype(int), list(events[:,1].astype(int)), events[:,2].astype(int), events[:,3].astype(int)


if __name__ == "__main__":
    path_base = '/home/thilo/workspace/data/pipeline/test99/data/'
    # path_base = '/home/thilo/workspace/data/first_evaluation_sets/set010_20240401-134640/data/'
    path_frames_file = f'{path_base}/isaac_stats.csv'
    path_frames_folder = f'{path_base}/frames'
    path_events_file = f'{path_base}/events_v2e.txt'
    path_mask = f'{path_base}/frame_00083333337_mask.png'
    path_frames_mask_events_folder = f'{path_base}/frames_masks_events'
    path_ffmpeg = "/home/thilo/workspace/ffmpeg-6.1-amd64-static/"
    path_vid = f'{path_base}'
    vid_name = 'vid_frames_mask_events.mp4'
    frames_shape = (346, 260)
    
    frames, frames_ts = load_frames(path_frames_file, path_frames_folder, frames_shape)
    # for frame_ts in frames_ts:
    #     print(f'01: {int(frame_ts):025}')
    #     print(f'02: {int(frame_ts)}')
    #     print(f'03: {frame_ts}')
    events_ts, events_coord_y, events_coords_x, events_pol = load_events(path_events_file)
    mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)

    new_vid = Vid(frames, frames_ts, events_ts, events_coord_y, events_coords_x, events_pol, mask)
    
    for img, frame_ts in zip(new_vid.frame_with_mask_and_events, frames_ts):
        cv2.imwrite(f'{path_frames_mask_events_folder}/frame_mask_event_{int(frame_ts):025}.png', img)
        
    cmd = [f'{path_ffmpeg}/ffmpeg', '-y', '-framerate', '60', '-pattern_type', 'glob', '-i', f'{path_frames_mask_events_folder}/frame_mask_event_*.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', f'{path_vid}/{vid_name}']
    def frames_vid(cmd):
        p = Popen(cmd)
        p.wait()
    vid_process = multiprocessing.Process(target=frames_vid ,args=([cmd]))
    vid_process.start()
    vid_process.join()
