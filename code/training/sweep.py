# Import the W&B Python Library and log into W&B
import wandb
import math

from train_mlp import TrainMLP



project_name = "data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574_01"

def main():
    wandb.init()
    
    data_dir = "/home/thilo/workspace/data/vt_snn/preprocessed/pre_data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574"
    sample_file = 1
    checkpoint_dir = "/home/thilo/workspace/data/vt_snn/models/mlp/model_data064_2textures_big_v2e_thres15_adClearSep0.1-1.0_size2x574_mlp_optiAdam_batch256_lr0.007_01/"
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    hidden_size = wandb.config.hidden_size
    lr = wandb.config.lr
    optimizer = wandb.config.optimizer
    repetition = wandb.config.repetition
    
    train_mlp = TrainMLP(data_dir, sample_file, checkpoint_dir, batch_size, hidden_size, lr, optimizer, epochs, repetition)
    max_test_acc = 0
    max_test_acc_epoch = 0
    min_test_loss = math.inf
    min_test_loss_epoch = 0
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_mlp._train(epoch)
        wandb.log({"train_acc": train_acc, "epoch": epoch, "optimizer": optimizer})
        wandb.log({"train_loss": train_loss, "epoch": epoch, "optimizer": optimizer})
        if epoch % 10 == 0:
            test_acc, test_loss = train_mlp._test(epoch)
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
        # if epoch % 100 == 0:
        #     train_mlp._save_model(epoch)
    # wandb.log()

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "name": "sweepmania2.0",
    # "metric": {"goal": "maximize", "name": "test_acc"},
    "parameters": {
        "batch_size": {"values": [8,16,32,64,96,128,160,256,512]},
        "hidden_size": {"values": [32]},
        "lr": {"values": [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011,0.012,0.013,0.014]},
        "optimizer": {"values": ["Adam", "RMSprop"]},
        "epochs": {"values": [600]},
        "repetition": {"values": ["01", "02", "03"]}
    },
}

# 3: Start the sweep
# wandb.login()
# sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

# wandb.agent(sweep_id, function=main, count=10)

if __name__ == "__main__":
    main()
