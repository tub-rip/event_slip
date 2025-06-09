"""Test the MLP models.

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
from models.snn import SlayerMLP
import slayerSNN as snn
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()


class TestCNN3D:
    
    def __init__(self, data_dir, sample_file, checkpoint_dir, checkpoint_number, batch_size, hidden_size, network_config, loss) -> None:
        self.output_size = 2
        self.model = SlayerMLP
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_number = checkpoint_number
        self.hidden_size = hidden_size
        self.params = snn.params(network_config)
        self.loss = loss
        self.batch_size = batch_size

        self.test_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"train_80_20_{sample_file}.txt",
            # sample_file=f"test_80_20_{sample_file}.txt",
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
        self.model_args = {
            "params": self.params,
            "input_size": (50, 63, 2),
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        self.net = self.model(**self.model_args).to(self.device)
        self.net.load_state_dict(torch.load(f"{self.checkpoint_dir}weights-{self.checkpoint_number:03d}.pt"))        
        if self.loss == "NumSpikes":
            self.params["training"]["error"]["type"] = "NumSpikes"
            error = snn.loss(self.params).to(self.device)
            self.criteria = error.numSpikes
        elif self.loss == "WeightedNumSpikes":
            self.params["training"]["error"]["type"] = "WeightedNumSpikes"
            error = snn.loss(self.params).to(self.device)
            self.criteria = error.weightedNumSpikes



    def run_test(self):
        self._test()

    def _test(self):
        self.net.eval()
        correct = 0
        counter_used_samples = 0
        batch_loss = 0
        with torch.no_grad():
            for *inputs, target, label in self.test_loader:
                inputs = [i.to(self.device) for i in inputs]
                target = target.to(self.device)
                # label = label.to(self.device)
                output = self.net.forward(*inputs)
                _, predicted = torch.max(output.data, 1)
                # correct += (predicted == label).sum().item()
                correct += torch.sum(snn.predict.getClass(output) == label).data.item()
                counter_used_samples += len(label)
                spike_loss = self.criteria(output, target)  # numSpikes
                batch_loss += spike_loss.cpu().data.item()
                # batch_loss += batch01
                # batch01 = loss.cpu().data.item()
                # batch02 = loss.item()
                # batch_loss += batch01
                # loss_total += batch02
                # print('batch01', batch01)
                # print('size batch', len(label))
                
                
        test_acc = correct / len(self.test_loader.dataset)
        # test_loss = batch_loss / len(self.test_loader.dataset)
        # print("loss/test", test_loss)
        # print("acc/test", test_acc)
        # test_acc = correct / counter_used_samples
        test_loss = batch_loss / counter_used_samples
        print("loss/test", test_loss)
        print("acc/test", test_acc)
        return test_acc, test_loss
    

# correct += torch.sum(snn.predict.getClass(output) == label).data.item()
# num_samples += len(label)
# spike_loss = self.criteria(output, target)  # numSpikes

# test_acc = correct / num_samples
# test_loss = spike_loss / len(self.test_loader)
# self.writer.add_scalar("acc/test", test_acc, epoch)
# self.writer.add_scalar("loss/test", test_loss, epoch)
# return test_acc, test_loss



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Test MLP model.")

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
    parser.add_argument("--hidden_size", type=int, help="Size of hidden layer.", required=True)
    parser.add_argument(
        "--network_config",
        type=str,
        help="Path SNN network configuration.",
        required=True,
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to use.",
        choices=["NumSpikes", "WeightedNumSpikes"],
        required=True,
    )

    args = parser.parse_args()
    
    test_cnn3d = TestCNN3D(args.data_dir, args.sample_file, args.checkpoint_dir, args.checkpoint_number, args.batch_size, args.hidden_size, args.network_config, args.loss)   
    test_cnn3d.run_test()