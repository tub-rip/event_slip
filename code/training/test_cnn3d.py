"""Test the CNN3D models.

Usage (from project root directory):

2. With vanilla Python:

python vtsnn/train_cnn3d.py \
 --epochs 1000 \
 --lr 0.00001 \
 --sample_file 1 \
 --batch_size 8 \
 --data_dir /path/to/preprocessed \
 --mode tact \
 --task cw

where mode is one of {tact, vis, mm} and task is {cw, slip}.
"""

# from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
# import numpy as np
# import copy
# from pathlib import Path
import logging
import argparse
# import os
from torch.utils.tensorboard import SummaryWriter

from dataset import ViTacDataset
from models.cnn3d import VisCNN3D

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()


class TestCNN3D:
    
    def __init__(self, data_dir, sample_file, checkpoint_dir, checkpoint_number, batch_size) -> None:
        self.output_size = 2
        self.model = VisCNN3D
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_number = checkpoint_number
        self.batch_size = batch_size

        self.test_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"test_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=False,
            rectangular=True,
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=True
        )
        self.device = torch.device("cuda")
        self.net = self.model(self.output_size).to(self.device)
        self.net.load_state_dict(torch.load(f"{args.checkpoint_dir}weights-{self.checkpoint_number:03d}.pt"))
        self.criterion = nn.CrossEntropyLoss()

    def run_test(self):
        self._test()

    def _test(self):
        self.net.eval()
        counter = 0
        correct = 0
        loss_batch = 0
        counter_used_samples = 0
        loss_total = 0
        batch_loss = 0
        test_acc = 0
        with torch.no_grad():
            for *inputs, _, label in self.test_loader:
                counter += 1
                inputs = [i.to(self.device) for i in inputs]
                label = label.to(self.device)
                output = self.net.forward(*inputs)
                _, predicted = torch.max(output.data, 1)
                # print('correct in batch', (predicted == label).sum().item())
                correct += (predicted == label).sum().item()
                loss = self.criterion(output, label)
                batch01 = loss.cpu().data.item()
                batch02 = loss.item()
                batch_loss += batch01
                loss_total += batch02
                counter_used_samples += len(label)
                # print('batch01', batch01)
                # print('label', label)
                # print('predicted', predicted)
                # print('size batch', len(label))


        # print('correct', correct)
        # print('batch_loss', batch_loss)
        # print('loss_total', loss_total)
        test_acc = correct / counter_used_samples
        # test_loss = batch_loss / counter_used_samples
        test_loss = batch_loss / (counter_used_samples/self.batch_size)
        print("loss/test", test_loss)
        print("acc/test", test_acc)
        return test_acc, test_loss
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Train MLP-GRU model.")

    parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path for saving checkpoints.",
        default=".",
    )
    parser.add_argument(
        "--sample_file",
        type=int,
        help="Sample number to train from.",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)
    parser.add_argument("--checkpoint_number", type=int, help="The checkpoint to load.", required=True)

    args = parser.parse_args()
    
    test_cnn3d = TestCNN3D(args.data_dir, args.sample_file, args.checkpoint_dir, args.checkpoint_number, args.batch_size)   
    test_cnn3d.run_test()