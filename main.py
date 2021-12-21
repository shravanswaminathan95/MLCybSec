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
from itertools import chain, combinations
import threading
#from threading import Thread, Lock
from multiprocessing import Process, Pool, Pipe, Lock

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


dictTestParams = {}

undefended_model_path = "undefended_model.pth"
threshold_model_path = "threshold_model.pth"
defended_model_path = "defended_model.pth"
whiteboxRandomModel = "whiteboxRandomModel.pth"
attackDict = {} #filled by main
listAdvModels = []

path = os.path.abspath(__file__)
wspacepath = os.path.dirname(path)
pathElements = wspacepath.split(os.sep)
wspacepath = os.sep.join(pathElements)
pathElements.extend(["_out"])
outDir = os.sep.join(pathElements)
print("wspacepath=", wspacepath)
print("outDir=", outDir)

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
def train(modelInp, optimizerInp, epoch, saveModel, attackGeneratorList=None):
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


#This function is fucked up and against the idea of loading model from repo
def trainWithAdversary(modelInpPath, nEpoch, saveModel, attackGeneratorList, attackNameList):

    # load the model
    modelInp = Net().to(device)
    modelInp.load_state_dict(torch.load(modelInpPath))
    modelInp.eval()
    optimizerInp = optim.SGD(modelInp.parameters(), lr=learning_rate_2, momentum=momentum_2)

    for epoch in range(1, nEpoch + 1):
        modelInp.train()
        correct = 0
        for batch_idx, (dataX, target) in enumerate(train_loader):
            for attackType in attackGeneratorList:
                dataTmp = attackType(dataX, target)
                optimizerInp.zero_grad()
                output = modelInp(dataTmp)
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
        print("Training adversarial model with {}; Epoch = {}, Training Accuracy = {}".format(attackNameList, epoch, accuracy))

    print("################################################################################")
    print("Adversarial({}) model {} is trained and stored".format(attackNameList, saveModel))
    print("################################################################################")

def test(modelInp, test_loader_arg, attackGeneratorList=None, attackNameList='', id=0, lock=None):
    begin_time = datetime.datetime.now()
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

    runTime = (datetime.datetime.now() - begin_time)
    print("Attacking with: {} \n Test accuracy: {}\n Execution time: {} \n ################################################################################"
          .format(attackNameList, accuracy, runTime))
    recordTestParams(id, attackNameList, accuracy, runTime, lock)
    #print("Execution time - ", datetime.datetime.now() - begin_time)
    return accuracy

def recordTestParams(id, attackSet, accuracy, runTime, lock):
    global dictTestParams

    if lock:
        lock.acquire(blocking=True)
        dictTestParams[id] = (attackSet, accuracy, runTime)
        lock.release()
    else:
        print("recordTestParams has not lock over dictTestParams, therefore storing skipped")

def dumpTestParams(jsonName):
    global dictTestParams
    with open(os.sep.join([outDir, jsonName]), 'w', encoding='utf-8') as dumpFile:
        json.dump(dictTestParams, dumpFile, ensure_ascii=False, indent=4)

    dictTestParams = {}


def evaluateAttacks(originalModel, substituteModel, lock):

    global attackDict
    epsilonSmall = 0.05
    epsilonMiddle = 0.1
    epsilonLarge = 0.15

    epsilonUse = epsilonSmall

    attackDict = {
        "DeepFool": DeepFool(substituteModel, steps=10),  # used to be 1000 steps
        "FGSM": FGSM(substituteModel, eps=epsilonUse),
        "PGD": PGD(substituteModel, eps=epsilonUse, alpha=0.5, steps=7, random_start=True),
        "CW": CW(substituteModel, c=100, lr=0.01, steps=10, kappa=10),  # used to be 1000 steps,
        # "FFGSM"       :   FFGSM(model, eps=epsilonUse, alpha=0.1),
        # "VANILA"      :   VANILA(model),
        # "APGD"        :   APGD(model, eps=epsilonUse, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
        # "AutoAttack"  :   AutoAttack(model, eps=0.05, n_classes=10, version='standard'),
        # "DIFGSM"      :   DIFGSM(model, eps=epsilonUse, alpha=2 / 255, steps=100, diversity_prob=0.5, resize_rate=0.9),
        "FAB": FAB(substituteModel, eps=epsilonUse, steps=10, n_classes=10, n_restarts=1, targeted=False),
        # steps used to be 100
        # "FAB2"        :   FAB(model, eps=epsilonUse, steps=10, n_classes=10, n_restarts=1, targeted=True)
    }
    attackThreadPoolDict = {}
    attackKeys = list(attackDict.keys())

    # Initializing threading helper dict
    for i in range(len(attackKeys)):
        attackThreadPoolDict[i + 1] = []

    # enumerate(chain.from_iterable(combinations(attackKeys, r) for r in range(len(attackKeys) + 1)), 1)
    for itr, attackSet in enumerate(
            chain.from_iterable(combinations(attackKeys, r) for r in range(len(attackKeys) + 1)), 1):
        if attackSet:
            print("Generated attack set: {}".format(attackSet))
            attackModelList = []
            attackNameList = []
            for attackX in list(attackSet):
                attackNameList.append(attackX)
                attackModelList.append(attackDict[attackX])
            # trying combinatorial attacks
            procInst = Process(target=test,
                               args=(originalModel, test_loader, attackModelList, attackNameList, itr, lock))

            #threadInst = threading.Thread(target=test,
            #                              args=(originalModel, test_loader, attackModelList, attackNameList, itr,))
            attackThreadPoolDict[len(list(attackSet))].append(procInst)

    print("Starting threads for generating adversarial samples... \n .....The command line might be dead for a while")
    # Threads are used to execute the adversarial set computation.
    # Threads are grouped with the number of cascaded attacks in order to make the join more efficient
    # A simple asscending order of attack depth would also work.
    # Attack depth = number of cascaded layers in the attack
    for attackDepth in list(attackThreadPoolDict.keys()):
        print("Generating adversarial samples for {} cascaded attacks".format(attackDepth))
        for threadX in attackThreadPoolDict[attackDepth]:
            threadX.start()
        print("################################################################################")
        for threadX in attackThreadPoolDict[attackDepth]:
            threadX.join()


def adversarialTraining(modelPath):
    advTrainThreadPoolDict = {}
    global attackDict, listAdvModels
    nEpoch = 4

    attackKeys = list(attackDict.keys())

    # Initializing threading helper dict
    for i in range(len(attackKeys)):
        advTrainThreadPoolDict[i + 1] = []

    for itr, attackSet in enumerate(
            chain.from_iterable(combinations(attackKeys, r) for r in range(len(attackKeys) + 1)), 1):
        if attackSet:
            print("Generated attack set: {}".format(attackSet))
            attackModelList = []
            attackNameList = []
            advModelNameStr = "advModel_"
            for attackX in list(attackSet):
                advModelNameStr += str(attackX)
                attackNameList.append(attackX)
                attackModelList.append(attackDict[attackX])
            advModelNameStr += ".pth"
            listAdvModels.append(advModelNameStr)
            # trying combinatorial attacks
            procInst = Process(target=trainWithAdversary,
                               args=(modelPath, nEpoch, advModelNameStr, attackModelList, attackNameList,))

            #threadInst = threading.Thread(target=trainWithAdversary,
            #                              args=(modelPath, nEpoch, advModelNameStr, attackModelList, attackNameList,))
            advTrainThreadPoolDict[len(list(attackSet))].append(procInst)

    print("Starting threads for training models with adversarial samples... \n .....The command line might be dead for a while")
    # Threads are used to execute the adversarial set computation.
    # Threads are grouped with the number of cascaded attacks in order to make the join more efficient
    # A simple asscending order of attack depth would also work.
    # Attack depth = number of cascaded layers in the attack
    for attackDepth in list(advTrainThreadPoolDict.keys()):
        print("Training adversarial model for {} cascaded defense".format(attackDepth))
        for threadX in advTrainThreadPoolDict[attackDepth]:
            threadX.start()
        print("################################################################################")
        for threadX in advTrainThreadPoolDict[attackDepth]:
            threadX.join()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dictTestParamsMutex = Lock()

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

    substituteModel = Net().to(device)
    optimizerSubstituteModel = optim.SGD(substituteModel.parameters(), lr=learning_rate_2, momentum=momentum_2)


    train(substituteModel, optimizerSubstituteModel, 1, whiteboxRandomModel)



    for epoch in range(1, n_epochs + 1):
        train(networkOriginal, optimizerOriginal, epoch, undefended_model_path)
        test(networkOriginal, test_loader)

    print("################################################################################")
    print("Training Summary - ")
    for epoch in range(1, n_epochs + 1):
        print('.....Epoch %d, Train Accuracy: %f, Test Accuracy: %f' % (epoch, train_acc[epoch - 1], test_acc[epoch - 1]))
    print("################################################################################")

    evaluateAttacks(originalModel=networkOriginal,
                    substituteModel=substituteModel,
                    lock=dictTestParamsMutex)

    dumpTestParams("cascadedAttackAccuracy.json")


    adversarialTraining(undefended_model_path)

    for advModelPath in listAdvModels:
        advModel = Net().to(device)
        advModel.load_state_dict(torch.load(advModelPath))
        advModel.eval()
        evaluateAttacks(originalModel=advModel,
                        substituteModel=substituteModel,
                        lock=dictTestParamsMutex)
        dumpJsonName = advModelPath.replace(".pth", "")
        dumpJsonName = advModelPath.replace("advModel_", "")
        dumpJsonName = "cascadedAttackAccuracy_AdvModel_" + dumpJsonName + ".json"
        dumpTestParams(dumpJsonName)
















