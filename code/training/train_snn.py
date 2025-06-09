"""Train the VT-SNN models

1. With guild:
guild run vtsnn:train-{cw,sd} mode={tact,vis,mm} data_dir=/path/to/preprocessed/

2. With vanilla Python:

vt_snn example:
python vtsnn/train_snn.py \
 --epochs 500 \
 --lr 0.001 \
 --sample_file 1 \
 --batch_size 8 \
 --network_config ./network_config/correct_config.yml \
 --data_dir /path/to/preprocessed \
 --hidden_size 32 \
 --loss NumSpikes \
 --mode tact \
 --task cw
 
 
MA example: (params for best result of sweep on data064)
python3 train_snn.py \
--epoch 600 \
--lr 0.013 \
--sample_file 1 \
--batch_size 96 \
--data_dir /home/thilo/workspace/data/vt_snn/preprocessed/pre_data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574 \
--checkpoint_dir /home/thilo/workspace/data/vt_snn/models/final/snn/data064/ --optimizer RMSprop \
--repetition 01 \
--hidden_size 32 \
--network_config \
./network_config/slip_detection.yml \
--loss NumSpikes

where mode is one of {tact, vis, mm} and task is {cw, slip, ycb}.
"""
from pathlib import Path
import logging
import argparse
import pickle
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import slayerSNN as snn

from models.snn import SlayerMLP, SlayerMM
from dataset import ViTacDataset
# from vtsnn.models.snn import SlayerMLP, SlayerMM
# from vtsnn.dataset import ViTacDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger()

class TrainSNN:
    
    def __init__(self, data_dir, sample_file, checkpoint_dir, network_config, loss, batch_size, hidden_size, lr, optimizer_name, epochs, repetition) -> None:
        self.params = snn.params(network_config)
        self.output_size = 2
        self.model = SlayerMLP
        self.model_args = {
            "params": self.params,
            "input_size": (50, 63, 2),
            "hidden_size": hidden_size,
            "output_size": self.output_size,
        }
        self.checkpoint_dir = checkpoint_dir
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.epochs = epochs
        self.repetition = repetition
        self.loss = loss
        
        self.train_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"train_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=True,
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
        )
        self.test_dataset = ViTacDataset(
            path=data_dir,
            sample_file=f"test_80_20_{sample_file}.txt",
            output_size=self.output_size,
            spiking=True,
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
        )
        self.device = torch.device("cuda")
        # self.writer = SummaryWriter("/home/thilo/workspace/nn/VT_SNN/tensorboard/mlp")
        self.net = self.model(**self.model_args).to(self.device)
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.5)
        elif self.optimizer_name == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=self.lr, weight_decay=0.5)
        set_name = data_dir.split("/")[-1] if not len(data_dir.split("/")[-1]) == 0 else data_dir.split("/")[len(data_dir.split("/"))-2]
        self.result_name = f"{set_name[len('pre_'):]}_snn_loss{self.loss}_opti{optimizer_name}_batch{batch_size}_lr{lr}_maxEpoch{self.epochs}_{self.repetition}"
        self.writer = SummaryWriter(f"/home/thilo/workspace/data/tensorboard/final/models/{self.result_name}")    
        # self.writer = SummaryWriter(f"/home/thilo/workspace/data/tensorboard/final/{set_name[len('pre_'):]}_snn_loss{self.loss}_opti{optimizer_name}_batch{batch_size}_lr{lr}_maxEpoch{self.epochs}_{self.repetition}")    
        if self.loss == "NumSpikes":
            self.params["training"]["error"]["type"] = "NumSpikes"
            error = snn.loss(self.params).to(self.device)
            self.criteria = error.numSpikes
        elif self.loss == "WeightedNumSpikes":
            self.params["training"]["error"]["type"] = "WeightedNumSpikes"
            error = snn.loss(self.params).to(self.device)
            self.criteria = error.weightedNumSpikes
            
    def run_training(self):     
        for epoch in range(1, self.epochs + 1):
            self._train(epoch)
            if epoch % 10 == 0:
                self._test(epoch)
            if epoch % 100 == 0:
                self._save_model(epoch)

        with open(f"{self.checkpoint_dir}/args.pkl", "wb") as f:
            pickle.dump(args, f)
            
            
    def _train(self, epoch):
        correct = 0
        num_samples = 0
        self.net.train()
        for *data, target, label in self.train_loader:
            data = [d.to(self.device) for d in data]
            target = target.to(self.device)
            output = self.net.forward(*data)
            predicted = snn.predict.getClass(output)
            correct += torch.sum(predicted == label).data.item()
            num_samples += len(label)
            spike_loss = self.criteria(output, target)

            self.optimizer.zero_grad()
            spike_loss.backward()
            self.optimizer.step()
        train_acc = correct / num_samples
        train_loss = spike_loss / len(self.train_loader)
        self.writer.add_scalar("acc/train", train_acc, epoch)
        self.writer.add_scalar("loss/train", train_loss, epoch)
        return train_acc, train_loss


    def _test(self, epoch):
        correct = 0
        num_samples = 0
        self.net.eval()
        with torch.no_grad():
            for *data, target, label in self.test_loader:
                data = [d.to(self.device) for d in data]
                target = target.to(self.device)
                output = self.net.forward(*data)
                correct += torch.sum(
                    snn.predict.getClass(output) == label
                ).data.item()
                num_samples += len(label)
                spike_loss = self.criteria(output, target)  # numSpikes

            test_acc = correct / num_samples
            test_loss = spike_loss / len(self.test_loader)
            self.writer.add_scalar("acc/test", test_acc, epoch)
            self.writer.add_scalar("loss/test", test_loss, epoch)
            return test_acc, test_loss


    # def _save_model(self, epoch):
    #     log.info(f"Writing model at epoch {epoch}...")
    #     checkpoint_path = Path(self.checkpoint_dir) / f"weights_{epoch:03d}.pt"
    #     model_path = Path(self.checkpoint_dir) / f"model_{epoch:03d}.pt"
    #     torch.save(self.net.state_dict(), checkpoint_path)
    #     torch.save(self.net, model_path)
        
        
    def _save_model(self, epoch):
        log.info(f"Writing model at epoch {epoch}...")
        save_dir =  Path(self.checkpoint_dir) / f"model_{self.result_name}"
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = f"{save_dir}/weights-{epoch:03d}.pt"
        torch.save(self.net.state_dict(), checkpoint_path)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Train VT-SNN models.")

    parser.add_argument(
        "--epochs", type=int, help="Number of epochs.", required=True
    )
    parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path for saving checkpoints.",
        default="/home/thilo/workspace/data/vt_snn/models/05_model_mix_mini_50/",
    )
    parser.add_argument(
        "--network_config",
        type=str,
        help="Path SNN network configuration.",
        required=True,
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

    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to use.",
        choices=["NumSpikes", "WeightedNumSpikes"],
        required=True,
    )
    parser.add_argument("--optimizer", type=str, help="Adam or RMSprop", required=True)
    parser.add_argument("--repetition", type=str, help="repetition of this training", required=True)

    args = parser.parse_args()
    train_snn = TrainSNN(args.data_dir, args.sample_file, args.checkpoint_dir, args.network_config, args.loss, args.batch_size,  args.hidden_size, args.lr, args.optimizer, args.epochs, args.repetition)
    train_snn.run_training()



