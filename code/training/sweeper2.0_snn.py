# Import the W&B Python Library and log into W&B
import wandb
import math

from train_mlp import TrainMLP
from train_cnn3d import TrainCNN3D
from train_snn import TrainSNN



# project_name = "data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574_01"

def main():
    wandb.init()
    
    # data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data001_train_randComb_huge1000_v2e_thres15_final_adClearSep0.1-1.0_size2x7221"
    # data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574/"
    data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data001_train_randComb_huge1000_v2e_thres15_final_adClearSep0.1-1.0_size2x7221/"
    sample_file = 1
    checkpoint_dir = "/home/thilo/workspace/data/vt_snn/models/final/"
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    optimizer = wandb.config.optimizer
    repetition = wandb.config.repetition
    
    if wandb.config.nn == "cnn3d":
        data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data001_train_randComb_huge1000_v2e_thres15_final_adClearSep0.1-1.0_size2x7221_cnn3d"
        # data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data067_train_randComb_big1000_v2e_thres15_adClearSep0.1-1.0_size2x3250"
        nn = TrainCNN3D(data_dir, sample_file, checkpoint_dir, batch_size, lr, optimizer, epochs, repetition)
    
    # if wandb.config.nn == "cnn3d_data066":
    #     data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data066_4textures_big_v2e_thres15_adClearSep0.1-1.0_size2x1088_cnn3d"
    #     nn = TrainCNN3D(data_dir, sample_file, checkpoint_dir, batch_size, lr, optimizer, epochs, repetition)
        
    elif wandb.config.nn == "mlp":
        hidden_size = wandb.config.hidden_size
        nn = TrainMLP(data_dir, sample_file, checkpoint_dir, batch_size, hidden_size, lr, optimizer, epochs, repetition)
    
    elif wandb.config.nn == "snn":
        hidden_size = wandb.config.hidden_size
        loss = wandb.config.loss
        network_config = "./network_config/slip_detection.yml"
        nn = TrainSNN(data_dir, sample_file, checkpoint_dir, network_config, loss, batch_size, hidden_size, lr, optimizer, epochs, repetition)
        
    
    max_test_acc = 0
    max_test_acc_epoch = 0
    min_test_loss = math.inf
    min_test_loss_epoch = 0
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = nn._train(epoch)
        wandb.log({"train_acc": train_acc, "epoch": epoch, "optimizer": optimizer})
        wandb.log({"train_loss": train_loss, "epoch": epoch, "optimizer": optimizer})
        if epoch % 10 == 0:
            test_acc, test_loss = nn._test(epoch)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                max_test_acc_epoch = epoch
            # max_test_acc = max(test_acc, max_test_acc)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                min_test_loss_epoch = epoch
            min_test_loss = min(test_loss, min_test_loss)
            wandb.log({"max_test_acc": max_test_acc, "epoch": epoch, "optimizer": optimizer})
            wandb.log({"max_test_acc_epoch": max_test_acc_epoch, "epoch": epoch, "optimizer": optimizer})
            wandb.log({"test_acc": test_acc, "epoch": epoch, "optimizer": optimizer})
            wandb.log({"test_loss": test_loss, "epoch": epoch, "optimizer": optimizer})
            wandb.log({"min_test_loss": min_test_loss, "epoch": epoch, "optimizer": optimizer})
            wandb.log({"min_test_loss_epoch": min_test_loss_epoch, "epoch": epoch, "optimizer": optimizer})
        if epoch % 20 == 0:
            if wandb.config.nn == "cnn3d":
                nn._save_model(epoch)

if __name__ == "__main__":
    main()
