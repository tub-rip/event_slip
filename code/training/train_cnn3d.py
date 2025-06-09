"""Train the CNN3D models.

Usage (from project root directory):

1. With guild:
guild run cnn3d:train-{cw,sd} mode={tact,vis,mm} data_dir=/path/to/preprocessed

2. With vanilla Python:

vt_snn example:
python vtsnn/train_cnn3d.py \
 --epochs 1000 \
 --lr 0.00001 \
 --sample_file 1 \
 --batch_size 8 \
 --data_dir /path/to/preprocessed \
 --mode tact \
 --task cw
 
ma example: (params for best result of sweep on data064)
python3 train_cnn3d.py \
--epoch 600 \
--lr 0.001 \
--sample_file 1 \
--batch_size 16 \
--data_dir /home/thilo/workspace/data/vt_snn/preprocessed/pre_data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574_cnn3d \
--checkpoint_dir /home/thilo/workspace/data/vt_snn/models/final/data064/ \
--optimizer RMSprop \
--repetition 01

where mode is one of {tact, vis, mm} and task is {cw, slip}.
"""

# from datetime import datetime
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
# import numpy as np
# import copy
from pathlib import Path
import logging
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

from dataset import ViTacDataset
from models.cnn3d import VisCNN3D
# from vtsnn.dataset import ViTacDataset
# from vtsnn.models.cnn3d import TactCNN3D, VisCNN3D, MmCNN3D

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()




class TrainCNN3D:
    
    def __init__(self, data_dir, sample_file, checkpoint_dir, batch_size, lr, optimizer_name, epochs, repetition, retrain_model = None) -> None:
        self.output_size = 2
        self.model = VisCNN3D
        self.checkpoint_dir = checkpoint_dir
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.epochs = epochs
        self.repetition = repetition
        self.train_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"train_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=False,
            rectangular=True,
        )

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True,
        )

        self.test_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"test_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=False,
            rectangular=True,
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=True,
        )
        self.device = torch.device("cuda")
        self.net = self.model(self.output_size).to(self.device)
        if not retrain_model is None:
            self.net.load_state_dict(torch.load(retrain_model))
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer_name == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr)
        set_name = data_dir.split("/")[-1] if not len(data_dir.split("/")[-1]) == 0 else data_dir.split("/")[len(data_dir.split("/"))-2]
        retrain = "" if retrain_model is None else "retrained_"
        self.result_name = f"{set_name[len('pre_'):]}_cnn3d_opti{optimizer_name}_lr{lr}_batchS{batch_size}_maxEpoch{self.epochs}_{retrain}{self.repetition}"
        self.writer = SummaryWriter(f"/home/thilo/workspace/data/tensorboard/final/models/{self.result_name}")  

    def run_training(self):
        for epoch in range(1, self.epochs + 1):
            self._train(epoch)
            if epoch % 10 == 0:
                self._test(epoch)
            if epoch % 10 == 0:
                self._save_model(epoch)




    def _save_model(self, epoch):
        log.info(f"Writing model at epoch {epoch}...")
        save_dir =  Path(self.checkpoint_dir) / f"model_{self.result_name}"
        os.makedirs(save_dir, exist_ok=True)
        # checkpoint_path = f"/home/thilo/workspace/data/vt_snn/models/final/test/weights-{epoch:03d}.pt"
        checkpoint_path = f"{save_dir}/weights-{epoch:03d}.pt"
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



if __name__ == "__main__":
    
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
    parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)
    parser.add_argument("--optimizer", type=str, help="Adam or RMSprop", required=True)
    parser.add_argument("--repetition", type=str, help="repetition of this training", required=True)
    parser.add_argument("--retrain_model", type=str, default=None, help="Use pre trained model", required=False)

    args = parser.parse_args()
    
    train_cnn3d = TrainCNN3D(args.data_dir, args.sample_file, args.checkpoint_dir, args.batch_size, args.lr, args.optimizer, args.epochs, args.repetition, args.retrain_model)   
    train_cnn3d.run_training()