# -*- coding: utf-8 -*-
import argparse
import numpy as np
import math
import copy
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
print(torch.__version__, torchvision.__version__)


from model import _weights_init, resnet20, resnet50
from model_imagenet import resnet18
from folder import ImageFolder
from utils import *

########### config #########
parser = argparse.ArgumentParser(description='fool PoL - verify')
parser.add_argument('--dataset', type=str, default="imagenet", help="CIFAR10 or CIFAR100")
parser.add_argument('--model', type=str, default="resnet18", help="resnet20 or resnet50")
parser.add_argument('--iter', type=int, default=20000, help="training iterations")
parser.add_argument('--lr', type=float, default=0.01, help = "steps from W0' to Wn")
parser.add_argument('--t', type=int, default=20, help = "steps from W0' to Wn")
parser.add_argument('--k', type=int, default=100, help = "equal to freq / instep")
parser.add_argument('--round', type=int, default=10, help="dlg training rounds")
parser.add_argument('--batchsize', type=int, default=128, help="train dlg batch size")
parser.add_argument('--seed', type=int, default=0, help="lucky number")
args = parser.parse_args()
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))

init_threshold = 0.01 # from source code 0.01
order = ['1', '2', 'inf', 'cos']
threshold = [1000, 10, 0.1, 0.01]

########### config #########
if args.seed > 0:
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
print("Running on %s" % device)

net = resnet18().to(device)

state = torch.load("proof/{}/model_step_0".format(args.dataset))
net.load_state_dict(state['net'])

cur_param = list((_.detach().clone() for _ in net.parameters()))
print("model have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))

criterion = nn.CrossEntropyLoss().to(device)

apply_transform = transforms.Compose([
                transforms.Scale(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

train_dataset = ImageFolder(
            root="/data/home/Jingyu_Li/project/Resnet_ImageNet/train/cut_train",
            transform=apply_transform,
            classes_idx=(0, 10))

sequence = np.load("proof/{}/indices.npy".format(args.dataset))
subset = torch.utils.data.Subset(train_dataset, sequence)
train_loader = torch.utils.data.DataLoader(subset, batch_size=args.batchsize, num_workers=0, pin_memory=True, drop_last=True)

print("--------- load checkpoint!----------")

state = torch.load("proof/{}/model_step_0".format(args.dataset))
net.load_state_dict(state['net'])

print("--------- verify init!----------")
# verify init



print("--------- verify start!----------")
max = [0, 0, 0, 0]
min = [1000, 1000, 1000, 1000]
sum = [0, 0, 0, 0]
avg = [0, 0, 0, 0]

valid_count = 0
optimizer = torch.optim.SGD(net.parameters(), lr = args.lr)

for idd in range(1):
    
    cur_param = list((_.detach().clone() for _ in net.parameters()))
    step_count = 0
    for j, data in enumerate(train_loader):
        step_count += 1
        img, label = data
        img = img.to(device)
        label = label.to(device)
        # train dummy data
        
        optimizer.zero_grad()
        pred = net(img)

        loss = criterion(pred, label) 

        loss.backward()
        optimizer.step()


        if step_count % args.k == 0:
            print("t:", int(step_count / args.k))
            flag = 0
            # verify k round
            dummy_param = list((_.detach().clone() for _ in net.parameters()))
            state = torch.load("proof/{}/model_step_%d".format(args.dataset) % (args.k * step_count / args.k))
            net.load_state_dict(state['net'])

            target_param = list((_.detach().clone() for _ in net.parameters()))

            dist_list = [[] for i in range(len(order))]
            res = parameter_distance(target_param, dummy_param, order=order)
            for idx in range(len(res)):
                if res[idx] < min[idx]:
                    min[idx] = res[idx]
                if res[idx] > max[idx]:
                    max[idx] = res[idx]
                sum[idx]+=res[idx]
                avg[idx] = sum[idx] / (step_count / args.k)
            print("max:", max)
            print("min:", min)
            print("avg:", avg)
            for j in range(len(order)):
                dist_list[j].append(res[j])
            dist_list = np.array(dist_list)
            for k in range(len(order)):
                print(f"Distance metric: {order[k]} || threshold: {threshold[k]}")
                print(f"Average distance: {np.average(dist_list[k])}")
                above_threshold = np.sum(dist_list[k] > threshold[k])
                if above_threshold == 0:
                    print("None of the steps is above the threshold, the proof-of-learning is valid.")
                else:
                    print(f"{above_threshold} / {dist_list[k].shape[0]} "
                        f"({100 * np.average(dist_list[k] > threshold[k])}%) "
                        f"of the steps are above the threshold, the proof-of-learning is invalid.")
                    flag = 1
            
            if flag == 0:
                valid_count += 1

            print("=> valid rate: (%d/ %d), total: %d" % (valid_count, step_count/args.k, args.t))
            print("")
            if step_count == args.iter:
                break
