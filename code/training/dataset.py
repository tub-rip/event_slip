"""Datasets definitions for VT-SNN.

Each dataset should return an output of shape (data, target, label).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class ViTacDataset(Dataset):
    def __init__(
        self,
        path,
        sample_file,
        output_size,
        # mode,
        spiking,
        rectangular=False,
        loihi=False,
        size=None,
    ):
        self.path = path
        self.size = size
        self.output_size = output_size
        # self.mode = mode
        self.rectangular = rectangular
        self.spiking = spiking
        self.samples = np.loadtxt(Path(path) / sample_file).astype("int")
       
        # if mode != "tact":  # includes vision
        if spiking:  # Load the correct downsampled data
            self.vis = torch.load(Path(path) / "ds_vis.pt")
        else:
            self.vis = torch.load(Path(path) / "ds_vis_non_spike.pt")
        if loihi and self.vis is not None:
            # combine positive/negative polarities
            self.vis = torch.from_numpy(
                np.clip(torch.sum(self.vis, dim=1).numpy(), 0, 1)
            )
            self.vis = self.vis.reshape(
                self.vis.shape[0], -1, 1, 1, self.vis.shape[-1]
            )

    def __getitem__(self, index):
        if self.size is not None:
            index = index % len(self.samples)
        input_index = self.samples[index, 0]
        class_label = self.samples[index, 1]
        
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1
        
        inputs = [self.vis[input_index]]

        return (
            *inputs,
            target_class,
            class_label,
        )

    def __len__(self):
        if self.size is not None:
            return self.size
        else:
            return self.samples.shape[0]
