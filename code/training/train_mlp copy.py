"""Train the MLP-GRU models.

Usage (from project root directory):

1. With guild:
guild run mlp-gru:train-{cw,sd} mode={tact,vis,mm} data_dir=/path/to/preprocessed

2. With vanilla Python:

python vtsnn/train_mlp.py \
 --epochs 500 \
 --lr 0.0001 \
 --sample_file 1 \
 --batch_size 8 \
 --data_dir /path/to/preprocessed \
 --hidden_size 32 \
 --mode tact \
 --task cw

where mode is one of {tact, vis, mm} and task is {cw, slip}.
"""

from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import numpy as np
import copy
from pathlib import Path
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from dataset import ViTacDataset
from models.ann import TactMlpGru, VisMlpGru, MultiMlpGru
import wandb

wandb.login()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()


parser = argparse.ArgumentParser("Train MLP-GRU model.")
parser.add_argument(
    "--epochs", type=int, help="Number of epochs.", required=True
)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default=".",
)
parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument(
    "--sample_file",
    type=int,
    help="Sample number to train from.",
    required=True,
)
parser.add_argument(
    "--hidden_size", type=int, help="Size of hidden layer.", required=True
)

parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)

args = parser.parse_args()





class TrainMLP:
    
    
    def __init__(self, data_dir, sample_file, checkpoint_dir, batch_size, hidden_size, lr) -> None:
        self.output_size = 2
        self.model = VisMlpGru
        self.checkpoint_dir = checkpoint_dir
        self.lr = lr
        self.train_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"train_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=False,
            rectangular=False,
        )   

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
        )

        self.test_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"test_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=False,
            rectangular=False,
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
        )
        self.device = torch.device("cuda")
        self.writer = SummaryWriter("/home/thilo/workspace/data/tensorboard/mlp/data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574_mlp_optiAdam_batch256_lr0.007_03")    
        self.net = self.model(hidden_size, self.output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.5)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            
    def run_training(self):
        for epoch in range(1, args.epochs + 1):
            self._train(epoch)
            if epoch % 10 == 0:
                self._test(epoch)
            if epoch % 100 == 0:
                self._save_model(epoch)
            
    def _save_model(self, epoch):
        log.info(f"Writing model at epoch {epoch}...")
        checkpoint_path = Path(self.checkpoint_dir) / f"weights-{epoch:03d}.pt"
        torch.save(self.net.state_dict(), checkpoint_path)


    def _train(self, epoch):
        self.net.train()
        correct = 0
        batch_loss = 0
        train_acc = 0
        for *inputs, _, label in self.train_loader:
            inputs = [i.to(self.device) for i in inputs]
            label = label.to(self.device)
            output = self.net.forward(*inputs)
            loss = self.criterion(output, label)

            batch_loss += loss.cpu().data.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()

        train_acc = correct / len(self.train_loader.dataset)
        train_loss = batch_loss / len(self.train_loader.dataset)
        self.writer.add_scalar(
            "loss/train", train_loss, epoch
        )
        self.writer.add_scalar("acc/train", train_acc, epoch)
        return train_acc, train_loss


    def _test(self, epoch):
        self.net.eval()
        correct = 0
        batch_loss = 0
        test_acc = 0
        with torch.no_grad():
            for *inputs, _, label in self.test_loader:
                inputs = [i.to(self.device) for i in inputs]
                label = label.to(self.device)
                output = self.net.forward(*inputs)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).sum().item()
                loss = self.criterion(output, label)
                batch_loss += loss.cpu().data.item()

        test_acc = correct / len(self.test_loader.dataset)
        test_loss = batch_loss / len(self.test_loader.dataset)
        self.writer.add_scalar("loss/test", test_loss, epoch)
        self.writer.add_scalar("acc/test", test_acc, epoch)
        return test_acc, test_loss
        
        
train_mlp = TrainMLP(args.data_dir, args.sample_file, args.checkpoint_dir, args.batch_size,  args.hidden_size, args.lr)   
train_mlp.run_training()