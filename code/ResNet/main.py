import models
import train
import compressors
from utils import set_seed, load_data, get_device
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    trainloader, testloader, classes = load_data()
    device = get_device()

    compression_types = [
        'TopK',
        'ImpK_b',
        'ImpK_c'
    ]
    param_usage = 0.1
    num_epochs = 5
    lr = 0.01
    num_restarts = 1

    train_log, train_acc = {}, {}
    test_log, test_acc = {}, {}

    for compression_type in compression_types:
        lr = 0.01
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
                eta=2. if compression_type == 'ImpK_b' else 100000.,
                num_steps=20,
                device=device
            )
            train_log[compression_type].append(train_loss)
            train_acc[compression_type].append(train_accuracy)
            test_log[compression_type].append(test_loss)
            test_acc[compression_type].append(test_accuracy)


    fig, axs = plt.subplots(1, 2, figsize=(16, 7))


    for compression_type in test_log:
        train_loss = np.array(train_log[compression_type])
        train_loss_mean = np.mean(train_loss, axis=0)
        train_loss_std = np.std(train_loss, axis=0)
        
        train_accuracy = np.array(train_acc[compression_type])
        train_accuracy_mean = np.mean(train_accuracy, axis=0)
        train_accuracy_std = np.std(train_accuracy, axis=0)
        
        iters = list(range(len(train_loss_mean)))
        
        axs[0].plot(iters, train_loss_mean, label=compression_type)
        axs[0].fill_between(iters, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1)
        
        axs[1].plot(iters, train_accuracy_mean, label=compression_type)
        axs[1].fill_between(iters, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.1)

    axs[0].set_title("Comparison on Train, different compression types, param_usage=0.2")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title("Comparison on Train, different compression types, param_usage=0.2")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid()

    plt.show()