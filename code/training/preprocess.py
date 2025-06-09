""" VT-SNN data preprocessor.

Usage (from root directory):

1. With guild:

guild run preprocess save_path=/path/to/save \
  data_path=/path/to/data \

2. With plain Python:

python vtsnn/preprocess.py \
  --save_path /path/to/save \
  --data_path /path/to/data \
  --threshold 1 \
  --selection grasp_lift_hold \
  --bin_duration 0.02 \
  --n_sample_per_object 20 \
  --slip 0
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import slayerSNN as snn
import torch.nn.functional as F
import glob

import pandas as pd
import numpy as np
import os
import logging
import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from pathlib import Path

from collections import namedtuple

parser = argparse.ArgumentParser(description="VT-SNN data preprocessor.")

parser.add_argument(
    "--save_path", type=str, help="Location to save data to.", required=True
)
parser.add_argument(
    "--data_path", type=str, help="Path to dataset.", required=True
)
parser.add_argument("--blacklist_path", type=str, help="Path to blacklist.")
# parser.add_argument(
#     "--tact_threshold", type=int, help="Threshold for tactile.", required=True
# )
parser.add_argument(
    "--vis_threshold", type=int, help="Threshold for vision.", required=True
)
parser.add_argument(
    "--n_sample_per_object",
    type=int,
    help="Number of samples per class.",
    required=True,
)
parser.add_argument(
    "--network_config", type=str, help="Configuration to use.", required=True
)
parser.add_argument("--seed", type=int, help="Random seed to use", default=100)

parser.add_argument(
    "--bin_duration", type=float, help="Binning duration.", required=True
)
parser.add_argument(
    "--num_splits",
    type=int,
    help="Number of splits for stratified K-folds.",
    default=5,
)

parser.add_argument(
    "--cnn3d", type=bool, help="Use settings for cnn3d net", default=False
)

args = parser.parse_args()

selections_slip = {"full": [1, 0.0, 0.15]}

SELECTION = [1, 0.0, 8.5]
Trajectory = namedtuple(
    "Trajectory", ["start", "reaching", "reached", "grasping"],
)



class CameraData:
    def __init__(self, obj_name, selection):
        # propophesee hyperparameters
        self.c = 2
        self.w = 200 #346
        self.h = 250 #260
        prophesee_cam = False
        x0 = 180
        y0 = 0

        file_path = (
            Path(args.data_path) / "prophesee_recordings" / f"{obj_name}"
        )

        self.T = 0.15
        print('###################### CameraData ######################')
        print('object_name', obj_name)
        self.start_t = 0.0
        # self.start_t = self.trajectory[traj_start] + offset
        # self.start_t = self.trajectory[traj_start] + offset

        td_data = loadmat(str(file_path) + "_td.mat")["td_data"]
        df = pd.DataFrame(columns=["x", "y", "polarity", "timestamp"])
        if prophesee_cam:
            traj_start, offset, self.T = selection # {"full": [1, 0.0, 0.15]}
            # self.start_t = 1.0
            self.start_t = self.trajectory[traj_start] + offset
            a = td_data["x"][0][0]
            b = td_data["y"][0][0]
            mask_x = (a >= 230) & (a < 430)
            mask_y = b >= 100
            a1 = a[mask_x & mask_y] - 230
            b1 = b[mask_x & mask_y] - 100
            df.x = a1.flatten()
            df.y = b1.flatten()
            df.polarity = td_data["p"][0][0][mask_x & mask_y].flatten()
            df.timestamp = (
                td_data["ts"][0][0][mask_x & mask_y].flatten() / 1000000.0
            )
        else:
            self.start_t = 0.0
            a = td_data["x"][0][0]
            b = td_data["y"][0][0]
            # 'horizontally'
            # a = td_data["y"][0][0]
            # b = td_data["x"][0][0]
            # mask_x = a >= 60
            # mask_y = (b >= 48) & (b < 298)
            # a1 = a[mask_x & mask_y] - 60
            # b1 = b[mask_x & mask_y] - 48
            # 'vertically'
            mask_x = (a >= 73) & (a < 273)
            mask_y = b >= 10
            a1 = a[mask_x & mask_y] - 73
            b1 = b[mask_x & mask_y] - 10
            df.x = a1.flatten()
            df.y = b1.flatten()
            df.polarity = td_data["p"][0][0][mask_x & mask_y].flatten()
            df.timestamp = (
                td_data["ts"][0][0][mask_x & mask_y].flatten()
            )
            # print('df.timestamp XX',df.timestamp)
            # print('df.timestamp[0]',df.timestamp[0])
            # print('df.timestamp len', len(df.timestamp))
            # print('df.timestamp[-1]',df.timestamp[len(df.timestamp)-1])
            df_ts_len = len(df.timestamp)
            # print('len df', len(df))
            # if len(df.timestamp) == 0:
            #     print('ALARM at', obj_name, df)
            #     print('td_data', td_data)
            # print('df.timestamp[0]', df.timestamp[0])
            # print('df.timestamp[df_ts_len -1]', df.timestamp[df_ts_len -1])
            # print('len df.timestamp', len(df.timestamp))
            # print('len df.timestamp[df_ts_len -1]', len(df.timestamp[df_ts_len -1]))
            if len(df.timestamp) == 0:
                self.T = 0.15
            else:
                self.T = df.timestamp[df_ts_len -1]

        self.df = df
        self.threshold = args.vis_threshold

    def binarize(self, bin_duration):
        # print('args.cnn3d', args.cnn3d)
        bin_number = 350 if args.cnn3d else 150
        bin_duration = self.T/bin_number
        # bin_number = int(np.floor(self.T / bin_duration))
        # print('bin_number', bin_number)
        data_matrix = np.zeros([self.c, self.w, self.h, bin_number], dtype=int)

        pos_df = self.df[self.df.polarity == 1]
        neg_df = self.df[self.df.polarity == -1]

        end_t = self.start_t + bin_duration
        count = 0

        init_t = self.start_t
        while end_t <= self.T + init_t:
            _pos_count = pos_df[
                (
                    (pos_df.timestamp >= self.start_t)
                    & (pos_df.timestamp < end_t)
                )
            ]
            b = pd.DataFrame(index=_pos_count.index)
            b = b.assign(
                xy=_pos_count["x"].astype(str)
                + "_"
                + _pos_count["y"].astype(str)
            )
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split("_"), some_array))).astype(
                int
            )
            if xy.shape[0] > 0:
                data_matrix[0, xy[:, 0], xy[:, 1], count] = 1

            _neg_count = neg_df[
                (
                    (neg_df.timestamp >= self.start_t)
                    & (neg_df.timestamp < end_t)
                )
            ]
            b = pd.DataFrame(index=_neg_count.index)
            b = b.assign(
                xy=_neg_count["x"].astype(str)
                + "_"
                + _neg_count["y"].astype(str)
            )
            mask = b.xy.value_counts() >= self.threshold
            some_array = mask[mask].index.values.astype(str)
            xy = np.array(list(map(lambda x: x.split("_"), some_array))).astype(
                int
            )
            if xy.shape[0] > 0:
                data_matrix[1, xy[:, 0], xy[:, 1], count] = 1

            self.start_t = end_t
            end_t += bin_duration
            test_end = end_t
            count += 1
        data_matrix = np.swapaxes(data_matrix, 1, 2)
        return data_matrix


def vis_bin_save(file_name, overall_count, bin_duration, selection, save_dir):
    cam_data = CameraData(file_name, selection)
    # print(f"before ", file_name)
    visData = cam_data.binarize(bin_duration)
    # print(f"saving ...")
    f = save_dir / f"{overall_count}_vis.npy"
    print(f"Writing {f}...")
    np.save(f, visData.astype(np.uint8))
    print(f"Written {f}...")


class ViTacData:
    def __init__(self, save_dir, list_of_objects, selection="full"):
        self.list_of_objects = list_of_objects
        self.iters = args.n_sample_per_object
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.selection = selection

    def binarize_save(self, bin_duration):
        "saves binned tactile and prophesee data"
        print('START BINARIZE')
        overall_count = 0
        # big_list_tact = []
        big_list_vis = []
        for obj in self.list_of_objects:
            for i in range(1, self.iters + 1):
                file_name = f"{obj}_{i:02}"
                # big_list_tact.append(
                #     [
                #         file_name,
                #         overall_count,
                #         bin_duration,
                #         self.selection,
                #         self.save_dir,
                #     ]
                # )
                big_list_vis.append(
                    [
                        file_name,
                        overall_count,
                        bin_duration,
                        self.selection,
                        self.save_dir,
                    ]
                )
                overall_count += 1

        # Parallel(n_jobs=18)(delayed(tact_bin_save)(*zz) for zz in big_list_tact)
        # Parallel(n_jobs=18)(delayed(vis_bin_save)(*zz) for zz in big_list_vis)
        print('Before LOOP')
        Parallel(n_jobs=18, max_nbytes=None)(delayed(vis_bin_save)(*zz) for zz in big_list_vis)
        print('END BINARIZE')

list_of_objects2 = ["stable", "rotate"]

ViTac = ViTacData(Path(args.save_path), list_of_objects2, selection=SELECTION)

ViTac.binarize_save(bin_duration=args.bin_duration)
print('after binarize')

# create labels
labels = []
current_label = -1
overall_count = -1
for obj in list_of_objects2:
    current_label += 1
    for i in range(0, args.n_sample_per_object):
        overall_count += 1
        labels.append([overall_count, current_label])
labels = np.array(labels)

# stratified k fold
skf = StratifiedKFold(n_splits=args.num_splits, random_state=100, shuffle=True)
train_indices = []
test_indices = []


for train_index, test_index in skf.split(np.zeros(len(labels)), labels[:, 1]):
    train_indices.append(train_index)
    test_indices.append(test_index)

print(
    "Training size:",
    len(train_indices[0]),
    ", Testing size:",
    len(test_indices[0]),
)

for split in range(args.num_splits):
    np.savetxt(
        Path(args.save_path) / f"train_80_20_{split+1}.txt",
        np.array(labels[train_indices[split], :], dtype=int),
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        Path(args.save_path) / f"test_80_20_{split+1}.txt",
        np.array(labels[test_indices[split], :], dtype=int),
        fmt="%d",
        delimiter="\t",
    )

# Reprocess into compact .pt format


class SumPool(torch.nn.Module):  # outputs spike trains
    def __init__(self, params):
        super(SumPool, self).__init__()
        self.slayer = snn.layer(params["neuron"], params["simulation"])
        self.pool = self.slayer.pool(4)

    def forward(self, input_data):
        slayer_psp_output = self.slayer.psp(input_data)
        slayer_pool_output = self.pool(slayer_psp_output)
        spike_out = self.slayer.spike(slayer_pool_output)
        return spike_out


class AvgPool(torch.nn.Module):
    def __init__(self):
        super(AvgPool, self).__init__()
        self.pool = torch.nn.AvgPool3d(
            (1, 4, 4), padding=[0, 1, 1], stride=(1, 4, 4)
        )

    def forward(self, input_data):
        out_data = F.relu(self.pool(input_data))
        return out_data


device = torch.device("cuda")

net_params = snn.params(args.network_config)
net = SumPool(net_params).to(device)
net2 = AvgPool().to(device)

ds_vis_spike_arr = []
ds_vis_non_spike_arr = []

vis_count = len(glob.glob(str(Path(args.save_path) / "*_vis.npy")))
print(f"Processing {vis_count} vision files...")

for i in range(vis_count):

    print(f"Processing vision {i}...")
    vis_npy = Path(args.save_path) / f"{i}_vis.npy"
    vis = torch.FloatTensor(np.load(vis_npy)).unsqueeze(0)
    vis = vis.to(device)

    with torch.no_grad():
        vis_pooled_spike = net.forward(vis)
        vis_pooled_non_spike = net2.forward(vis.permute(0, 1, 4, 2, 3))

    ds_vis_spike_arr.append(vis_pooled_spike.detach().cpu().squeeze(0))
    ds_vis_non_spike_arr.append(
        vis_pooled_non_spike.squeeze(0).permute(0, 2, 3, 1).detach().cpu()
    )
    print('vis_pooled_non_spike', vis_pooled_non_spike.shape)

print('SAVE ds_bis.pt', (Path(args.save_path) / "ds_vis.pt"))
ds_vis_spike = torch.stack(ds_vis_spike_arr)
print(f"ds_vis: {ds_vis_spike.shape}")
torch.save(ds_vis_spike, Path(args.save_path) / "ds_vis.pt")
del ds_vis_spike, ds_vis_spike_arr

ds_vis_non_spike = torch.stack(ds_vis_non_spike_arr)
print(f"ds_vis_non_spike: {ds_vis_non_spike.shape}")
torch.save(ds_vis_non_spike, Path(args.save_path) / "ds_vis_non_spike.pt")
