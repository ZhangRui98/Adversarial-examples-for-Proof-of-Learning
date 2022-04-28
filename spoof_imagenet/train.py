# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data
print(torch.__version__, torchvision.__version__)

from model import resnet20, resnet50, _weights_init
from model_imagenet import resnet18
from folder import ImageFolder
from utils import *


os.environ['CUDA_VISIBLE_DEVICES']='0'
########### config #########
parser = argparse.ArgumentParser(description='spoof PoL - train')
parser.add_argument('--dataset', type=str, default="imagenet10", help="dataset")
parser.add_argument('--batchsize', type=int, default=128, help="batch size")
parser.add_argument('--epoch', type=int, default=200, help="training epoch")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--freq', type=int, default=100, help='frequence of saving checkpoints')
parser.add_argument('--seed', type=int, default=0, help="lucky number")
args = parser.parse_args()
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))

########### config #########

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
print("Running on %s" % device)

#net = resnet18().to(device)
net = resnet18().to(device)
net.apply(_weights_init)

print("model have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))

# before train parameter
bt_para = list((_.detach().clone() for _ in net.parameters()))
bt_weight = copy.deepcopy(net.state_dict())

criterion = nn.CrossEntropyLoss().to(device)

apply_transform = transforms.Compose([
                transforms.Scale(128),
                transforms.CenterCrop(128),
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

apply_transform2 = transforms.Compose([
                transforms.Scale(128),
                transforms.CenterCrop(128),
                # transforms.RandomResizedCrop(128),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

train_dataset = ImageFolder(
            root="/data/home/Jingyu_Li/project/Resnet_ImageNet/train/cut_train",
            transform=apply_transform,
            classes_idx=(0, 10))


test_dataset = ImageFolder(
            root="/data/home/Jingyu_Li/project/Resnet_ImageNet/cut_val",
            transform=apply_transform2,
            classes_idx=(0, 10))

print(train_dataset.__len__())
train_size = train_dataset.__len__()
sequence = create_sequences(args.batchsize, train_size, args.epoch)

sequence = np.reshape(sequence, -1)
np.save("proof/{}/indices.npy".format(args.dataset), sequence)
subset = torch.utils.data.Subset(train_dataset, sequence)
train_loader = torch.utils.data.DataLoader(subset, batch_size=args.batchsize, num_workers=0, pin_memory=True, drop_last=True)

test_loader = Data.DataLoader(
    dataset=test_dataset,
    shuffle=True,
    batch_size=args.batchsize
)

optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

save_path = "proof/{}".format(args.dataset)
folder = os.path.exists(save_path)
if not folder:
	os.makedirs(save_path)

state = {'net': net.state_dict(),
         'optimizer': optimizer.state_dict()}
torch.save(state, "proof/{}/model_step_0".format(args.dataset))

print("--------- train start!----------")
a = []
loss_list = []
train_acc_list = []
test_acc_list = []
iter_count = 0
start_time = time.time()

for epoch in range(1):

    
    for i, data in enumerate(train_loader):
        net.train()
        iter_count = iter_count + 1 # count iters num

        optimizer.zero_grad()
        inputs, labels = data
        pred = net(inputs.to(device))
        loss = criterion(pred, labels.to(device))

        loss.backward()
        optimizer.step()

        if iter_count % args.freq == 0: 

            state = {'net': net.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, "proof/{}/model_step_%d".format(args.dataset) % (iter_count))
    
            #scheduler.step()
            net.eval()
            train_acc = test_accuracy(train_loader, net, 1000)
            test_acc = test_accuracy(test_loader, net, 1000)
            print("iter:",iter_count,"loss: %.3f train acc: %.3f test acc: %.3f" % (loss.item(), train_acc, test_acc))

    a.append(iter_count)
    loss_list.append(loss.item())
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


print(train_acc_list)
print(test_acc_list)
train_time = time.time() - start_time
print("train time:", train_time)
# after train parameter

plt.plot(a, train_acc_list, label = "train_acc")
plt.plot(a, test_acc_list, label = "test_acc")
plt.grid()
plt.legend()
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.savefig("img/train_acc.png")
plt.close("all")

plt.plot(a, loss_list)
plt.grid()
plt.xlabel("iterations")
plt.ylabel("loss")
plt.savefig("img/train_loss.png")
plt.close("all")

at_para = list((_.detach().clone() for _ in net.parameters()))

dis = 0
for gx, gy in zip(at_para, bt_para): 
    dis += ((gx - gy) ** 2).sum()

print("|| Wn-W0 || distance: ", math.sqrt(dis.item()))
