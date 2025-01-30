import models
import train
import compressors
from utils import set_seed, load_data, get_device
import torch.optim as optim
import torch.nn as nn
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":

    trainloader, testloader, classes = load_data()
    device = get_device()

    config = {
        'param_usage': 0.001,
        'num_restarts': 1,
        'num_epochs': 30,
    }

    compress_configs = [
        {
            'compression_type': 'TopK',
            'lr': 0.01,
            'eta': None,
            'num_steps': None,
        },
        {
            'compression_type': 'ImpK_b',
            'lr': 0.01,
            'eta': 5.,
            'num_steps': 20,
        },
        {
            'compression_type': 'ImpK_c',
            'lr': 0.01,
            'eta': 1000000.,
            'num_steps': 20,
        }
    ]


    train_log, train_acc = {}, {}
    test_log, test_acc = {}, {}

    param_usage = config['param_usage']
    num_restarts = config['num_restarts']
    num_epochs = config['num_epochs']

    for compress_config in compress_configs:
        compression_type = compress_config['compression_type']
        lr = compress_config['lr']
        eta = compress_config['eta']
        num_steps = compress_config['num_steps']

        train_log[compression_type], train_acc[compression_type], test_log[compression_type], test_acc[compression_type] = [], [], [], []
        
        for num_restart in range(num_restarts):
            set_seed(52 + num_restart)
            net = models.ResNet18().to(device)

            if compression_type == 'TopK':
                compressor = compressors.TopK(param_usage)
            elif compression_type == 'RandK':
                compressor = compressors.RandK(param_usage)
            elif compression_type == 'ImpK_b':
                compressor = compressors.ImpK_b(net, param_usage)
            elif compression_type == 'ImpK_c':
                compressor = compressors.ImpK_c(net, param_usage)
            
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            train_loss, train_accuracy, test_loss, test_accuracy = train.train(
                model=net,
                optimizer=optimizer,
                compressor=compressor,
                criterion=criterion,
                train_dataset=trainloader,
                val_dataset=testloader,
                num_epochs=num_epochs,
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                device=device
            )
            train_log[compression_type].append(train_loss)
            train_acc[compression_type].append(train_accuracy)
            test_log[compression_type].append(test_loss)
            test_acc[compression_type].append(test_accuracy)

    print("Train Loss")
    print(train_log)
    print("Train Accuracy")
    print(train_acc)
    print("Test Loss")
    print(test_log)
    print("Test Accuracy")
    print(test_acc)


    fig_train, axs_train = plt.subplots(1, 2, figsize=(16, 7))
    fig_test, axs_test = plt.subplots(1, 2, figsize=(16, 7))


    for compress_config in compress_configs:
        compression_type = compress_config['compression_type']

        train_loss = np.array(train_log[compression_type])
        train_loss_mean = np.mean(train_loss, axis=0)
        train_loss_std = np.std(train_loss, axis=0)
        
        train_accuracy = np.array(train_acc[compression_type])
        train_accuracy_mean = np.mean(train_accuracy, axis=0)
        train_accuracy_std = np.std(train_accuracy, axis=0)
        
        test_loss = np.array(test_log[compression_type])
        test_loss_mean = np.mean(test_loss, axis=0)
        test_loss_std = np.std(test_loss, axis=0)
        
        test_accuracy = np.array(test_acc[compression_type])
        test_accuracy_mean = np.mean(test_accuracy, axis=0)
        test_accuracy_std = np.std(test_accuracy, axis=0)
        
        iters = list(range(len(train_loss_mean)))
        
        axs_train[0].plot(iters, train_loss_mean, label=f'{compression_type}, lr={compress_config["lr"]}')
        axs_train[0].fill_between(iters, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1)
        
        axs_train[1].plot(iters, train_accuracy_mean, label=f'{compression_type}, lr={compress_config["lr"]}')
        axs_train[1].fill_between(iters, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1)

        axs_test[0].plot(iters, test_loss_mean, label=f'{compression_type}, lr={compress_config["lr"]}')
        axs_test[0].fill_between(iters, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.1)
        
        axs_test[1].plot(iters, test_accuracy_mean, label=f'{compression_type}, lr={compress_config["lr"]}')
        axs_test[1].fill_between(iters, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.1)

    axs_train[0].set_title(f"Comparison on Train, different compression types, param_usage={param_usage}")
    axs_train[0].set_xlabel("Epoch")
    axs_train[0].set_ylabel("Loss")
    axs_train[0].legend()
    axs_train[0].grid()

    axs_train[1].set_title(f"Comparison on Train, different compression types, param_usage={param_usage}")
    axs_train[1].set_xlabel("Epoch")
    axs_train[1].set_ylabel("Accuracy")
    axs_train[1].legend()
    axs_train[1].grid()
        

    axs_test[0].set_title(f"Comparison on Test, different compression types, param_usage={param_usage}")
    axs_test[0].set_xlabel("Epoch")
    axs_test[0].set_ylabel("Loss")
    axs_test[0].legend()
    axs_test[0].grid()

    axs_test[1].set_title(f"Comparison on Test, different compression types, param_usage={param_usage}")
    axs_test[1].set_xlabel("Epoch")
    axs_test[1].set_ylabel("Accuracy")
    axs_test[1].legend()
    axs_test[1].grid()

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if the directory 'figures' exists, if not, create it
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save the train plot in the 'figures' directory
    fig_train.savefig(os.path.join(figures_dir, f"train_comparison_param_usage_{param_usage}_{date}.png"))

    # Save the test plot in the 'figures' directory
    fig_test.savefig(os.path.join(figures_dir, f"test_comparison_param_usage_{param_usage}_{date}.png"))

    fig_train.show()
    fig_test.show()