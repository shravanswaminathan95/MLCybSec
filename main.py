import time

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import pickle
import sys
import csv
import os
import os.path as osp
import shutil
import pandas as pd
from IPython.display import display, HTML
from torchattacks import *
import datetime
import hashlib
from initializeDatasets import encryptFilesAndStore, decryptFilesAndVerify


# Some suggestions of our libraries that might be helpful for this project
#from collections import Counter          # an even easier way to count
#from multiprocessing import Pool         # for multiprocessing
#from tqdm import tqdm                    # fancy progress bars

# Load other libraries here.
# Keep it minimal! We should be easily able to reproduce your code.
# We only support sklearn and pytorch.
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision

# We preload pytorch as an example
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


# (2)define model
# Model with no dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # 16*24*24
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),  # 32*20*20
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32*10*10
            nn.Conv2d(32, 64, 5),  # 64*6*6
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64*3*3
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(-1, 64 * 3 * 3)
        out = self.fc_layer(out)

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)






# (3)define loss, optimizer
# Training function for all networks
def train(modelInp, optimizerInp, epoch, saveModel):
    modelInp.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #data.requires_grad = True
        optimizerInp.zero_grad()
        output = modelInp(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizerInp.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % log_interval == 0:
            #print('Network:Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #           100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(modelInp.state_dict(), saveModel)
    accuracy = 100 * correct / len(train_loader.dataset)
    train_acc.append(accuracy)
    print("Epoch = {}, Training Accuracy = {}".format(epoch, accuracy))


def test(modelInp, test_loader_arg, attackGeneratorList=None):
    modelInp.eval()
    test_loss = 0
    correct = 0

    for dataX, target in test_loader_arg:
        dataTmp = dataX
        if attackGeneratorList is not None:
            for attackType in attackGeneratorList:
                dataTmp = attackType(dataTmp, target)
        output = modelInp(dataTmp)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    try:
        accuracy = 100 * correct / len(test_loader_arg.dataset)
    except:
        accuracy = 100 * correct / len(test_loader_arg)
    test_acc.append(accuracy)
    try:
        test_loss /= len(test_loader_arg.dataset)
    except:
        test_loss /= len(test_loader_arg)
    test_losses.append(test_loss)

    print("Test accuracy: {}".format(accuracy))

    return accuracy




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(200)

    compute_mode = 'cpu'

    if compute_mode == 'cpu':
        device = torch.device('cpu')
    elif compute_mode == 'gpu':
        # If you are using pytorch on the GPU cluster, you have to manually specify which GPU device to use
        # It is extremely important that you *do not* spawn multi-GPU jobs.
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set device ID here
        device = torch.device('cuda')
    else:
        raise ValueError('Unrecognized compute mode')


    undefended_model_path = "undefended_model.pth"
    threshold_model_path = "threshold_model.pth"
    defended_model_path = "defended_model.pth"
    whiteboxRandomModel = "whiteboxRandomModel.pth"

    n_epochs = 4
    batch_size_train = 64
    batch_size_test = 1
    learning_rate_1 = 0.01
    learning_rate_2 = 0.05
    momentum_1 = 0.5
    momentum_2 = 0.25
    log_interval = 10

    imageDataForPlotting = []

    # defining list to save training and testing loss for future evaluation.
    train_losses = []
    train_counter = []
    test_losses = []
    # test_counter_4 = [i*len(train_loader.dataset) for i in range(n_epochs + 1)] todo - clean up
    train_acc = []
    test_acc = []

    model = Net().to(device)
    optimizerModel = optim.SGD(model.parameters(), lr=learning_rate_2, momentum=momentum_2)
    #todo, train the above model to the MNIST dataset. This will improve the attack vastly.


    # (1)load data
    # torchvision dataloaders to download MNIST dataset.
    transform = transforms.Compose([transforms.ToTensor()])
    dataSetPath = './dataManualDownload/original'
    if decryptFilesAndVerify(dataSetPath):
        raise Exception("Datasets could not be validated")
    else:
        print("Datasets authenticated. Starting training....")

    dataset = datasets.MNIST(root=dataSetPath, train=[True, False], transform=transform, download=True)
    trainSet, testSet = torch.utils.data.random_split(dataset, [50000, 10000])

    train_loader = data.DataLoader(trainSet, batch_size=batch_size_train, shuffle=True)
    test_loader = data.DataLoader(testSet, batch_size=batch_size_test, shuffle=True)

    networkOriginal = Net().to(device)
    optimizerOriginal = optim.SGD(networkOriginal.parameters(), lr=learning_rate_2, momentum=momentum_2)

    train(model, optimizerModel, 1, whiteboxRandomModel)

    for epoch in range(1, n_epochs + 1):
        train(networkOriginal, optimizerOriginal, epoch, undefended_model_path)
        test(networkOriginal, test_loader)

    attackDict = {
        "DeepFool"  :   DeepFool(model, steps=10),  # used to be 1000 steps
        "FGSM"      :   FGSM(model, eps=0.05),
        "PGD"       :   PGD(model, eps=0.05, alpha=0.5, steps=7, random_start=True),
        "CW"        :   CW(model, c=100, lr=0.01, steps=10, kappa=10),  # used to be 1000 steps,
        "FFGSM"     :   FFGSM(model, eps=0.25, alpha=0.1),
        "VANILA"    :   VANILA(model),
        "APGD"      :   APGD(model, eps=0.05, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
        #"AutoAttack":  AutoAttack(model, eps=0.05, n_classes=10, version='standard'),
        "DIFGSM"    :   DIFGSM(model, eps=0.05, alpha=2 / 255, steps=100, diversity_prob=0.5, resize_rate=0.9)
    }

    print("################################################################################")
    print("Training Summary - ")
    for epoch in range(1, n_epochs + 1):
        print('.....Epoch %d, Train Accuracy: %f, Test Accuracy: %f' % (epoch, train_acc[epoch - 1], test_acc[epoch - 1]))
    print("################################################################################")

    flgAttackIndv = False

    if flgAttackIndv:
        for attack in list(attackDict.keys()):
            print("Attacking model with adversarial images from ", attack)
            begin_time = datetime.datetime.now()
            test(networkOriginal, test_loader, [attackDict[attack]])
            print(".....^ Execution time - ", datetime.datetime.now() - begin_time)
    else:
        #trying combinatorial attacks
        print("Attacking with a combination of FGSM and PGD and Deepfool")
        begin_time = datetime.datetime.now()
        test(networkOriginal, test_loader, [attackDict["FGSM"],
                                            attackDict["PGD"],
                                            attackDict["DeepFool"]])
        print(".....^ Execution time - ", datetime.datetime.now() - begin_time)

        print("Attacking with a combination of FFGSM and PGD")
        begin_time = datetime.datetime.now()
        test(networkOriginal, test_loader, [attackDict["FFGSM"],
                                            attackDict["PGD"]])

        print(".....^ Execution time - ", datetime.datetime.now() - begin_time)

        print("Attacking with a combination of FFGSM, PGD and DeepFool")
        begin_time = datetime.datetime.now()
        test(networkOriginal, test_loader, [attackDict["FFGSM"],
                                            attackDict["PGD"],
                                            attackDict["DeepFool"]])

        print(".....^ Execution time - ", datetime.datetime.now() - begin_time)







