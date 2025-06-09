from segment_anything import SamPredictor, sam_model_registry
# import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# import time
# from segment_anything.utils.transforms import ResizeLongestSide

class Mask():

    def __init__(self, frame, mask_file_name, output_path, show=False, camera_mount='sideView', checkpoint='../sam_checkpoints/sam_vit_h_4b8939.pth'):
        print("SAM 03")
        self.mask_file_name = mask_file_name
        self.output_path = output_path
        self.frame = frame
        self.camera_mount = camera_mount
        self.checkpoint = checkpoint
        # print('#################### start mask generation ####################')
        # self.mask_image = self.generate_mask()
        # print('####################### Mask generated ########################')
        # if show:
        #     self.show()
        # self.save_mask()
    
    def generate_mask(self):
        sam = sam_model_registry["default"](checkpoint=self.checkpoint)
        sam.to(device='cpu')
        predictor = SamPredictor(sam)
        predictor.set_image(self.frame)
        self.input_points = None
        self.input_label = None
        # if self.camera_mount == "frontView":
        #     self.input_points = np.array([[170, 120],[170, 145]])
        #     self.input_label = np.array([1,1])
        # else:
            # self.input_points = np.array([[156, 140],[184, 140]])
            # self.input_label = np.array([1,1])
        
        
        # x_07, y_07 = (int(frame_shape[0]/2), int(frame_shape[1]/2))
        # x_08, y_08 = (int(frame_shape[0]/2), int(frame_shape[1]/2)+5)
        self.input_points = self.generate_points()
        print('SAM prompt points', self.input_points)
        self.input_label = np.array([1,1])
        # self.input_label = np.array([1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
        self.masks, self.scores, self.logits = predictor.predict(
            point_coords=self.input_points,
            point_labels=self.input_label,
            multimask_output=False,
        )
        h, w = self.masks[0].shape[-2:]
        mask_image = self.masks[0].reshape(h, w)
        self.mask_image =  mask_image*255
        # self.mask_image[int(self.frame.shape[0] - self.input_points[0][0])][int(self.frame.shape[1] - self.input_points[0][1])] = 120
        try:
            self.mask_image[self.input_points[0][1]-1][self.input_points[0][0]-1] = 120
        except:
            print('SAM: could not add gray prompt pixel')
        self.save_mask()
        
    def generate_points(self):
        y_gripper_00 = 100
        # no_go_area = [158, 188], [103]
        # size_min_hight = 120
        # size_max_hight = 240
        # size_min_width = 30
        # size_max_width = 150
        # max_side_dist = 120
        # min_side_dist = 0
        # max_gripper_depth = 40
        frame_shape = self.frame.shape # height = 260, width = 346, channels = 3
        print("FRAME_SHAPE", frame_shape)
        # y_00, x_00 = (int(frame_shape[0]), int(frame_shape[1]))
        y_00, x_00 = (105, int(frame_shape[1]/2))
        y_01, x_01 = (150, int(frame_shape[1]/2))
        
        # x_00, y_00 = (int(frame_shape[0]), int(frame_shape[1]))
        # x_00, y_00 = (int(frame_shape[0]/2+20), int(frame_shape[1]/2))
        # x_01, y_01 = (int(frame_shape[0]/2)+70, int(frame_shape[1]/2))
        # x_02, y_02 = (int(frame_shape[0]/2), int(frame_shape[1]/2)-35)
        # x_03, y_03 = (int(frame_shape[0]/2)+75, int(frame_shape[1]/2)+35)
        # x_04, y_04 = (int(frame_shape[0]/2)+75, int(frame_shape[1]/2)-35)
        # x_05, y_05 = (int(frame_shape[0]/2)-75, int(frame_shape[1]/2)+35)
        # x_06, y_06 = (int(frame_shape[0]/2)-75, int(frame_shape[1]/2)-35)
        # x_gripper_00, y_gripper_00 = (int(frame_shape[0]/2), int(frame_shape[1]/2)-35)
        # x_gripper_01, y_gripper_01 = (int(frame_shape[0]/2)+8, int(frame_shape[1]/2)-35)
        # x_gripper_02, y_gripper_02 = (int(frame_shape[0]/2)-8, int(frame_shape[1]/2)-35)
        # x_gripper_03, y_gripper_03 = (int(frame_shape[0]/2), int(frame_shape[1]/2)-45)
        # x_gripper_04, y_gripper_04 = (int(frame_shape[0]/2)+10, int(frame_shape[1]/2)-45)
        # x_gripper_05, y_gripper_05 = (int(frame_shape[0]/2)-10, int(frame_shape[1]/2)-45)
        # x_gripper_06, y_gripper_06 = (int(frame_shape[0]/2), int(frame_shape[1]/2)-55)
        # x_gripper_07, y_gripper_07 = (int(frame_shape[0]/2)+10, int(frame_shape[1]/2)-55)
        # x_gripper_08, y_gripper_08 = (int(frame_shape[0]/2)-10, int(frame_shape[1]/2)-55)
        # x_gripper_09, y_gripper_09 = (int(frame_shape[0]/2), 20)
        return np.array([[x_00, y_00],[x_01, y_01]])
        # return np.array([[y_00, x_00], [y_01, x_01],[y_02, x_02],[y_03, x_03],[y_04, x_04],[y_05, x_05],[y_06, x_06], [y_gripper_00, x_gripper_00], [y_gripper_01, x_gripper_01], [y_gripper_02, x_gripper_02], [y_gripper_03, x_gripper_03], [y_gripper_04, x_gripper_04], [y_gripper_05, x_gripper_05], [y_gripper_06, x_gripper_06], [y_gripper_07, x_gripper_07], [y_gripper_08, x_gripper_08], [y_gripper_09, x_gripper_09]])
    
    def save_mask(self, random_color=False):
        # filename, img_ext = os.path.splitext(self.mask_file_name)
        # filepath = os.path.join(dest_folder, os.path.basename(filename) +'_mask' + img_ext)
        # filepath = f"{self.output_path}/{os.path.basename(filename)}_mask{img_ext}"
        save_path = f"{self.output_path}/{self.mask_file_name}"
        print('filepath: ', save_path)
        cv2.imwrite(save_path, self.mask_image)
    
    def show(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.frame)
        self._show_mask(self.masks[0], plt.gca())
        self._show_points(self.input_points, self.input_label, plt.gca())
        plt.title(f"Mask, Score: {self.scores[0]:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    def _show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def _show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)