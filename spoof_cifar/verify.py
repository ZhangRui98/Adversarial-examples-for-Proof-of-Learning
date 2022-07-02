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
from utils import *

########### config #########
parser = argparse.ArgumentParser(description='fool PoL - verify')
parser.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10 or CIFAR100")
parser.add_argument('--model', type=str, default="resnet20", help="resnet20 or resnet50")
parser.add_argument('--iter', type=int, default=3000, help="training iterations")
parser.add_argument('--lr', type=float, default=0.01, help="steps from W0' to Wn")
parser.add_argument('--t', type=int, default=30, help="steps from W0' to Wn")
parser.add_argument('--k', type=int, default=100, help="equal to freq / instep")
parser.add_argument('--round', type=int, default=10, help="dlg training rounds")
parser.add_argument('--batchsize', type=int, default=128, help="train dlg batch size")
parser.add_argument('--seed', type=int, default=0, help="lucky number")
args = parser.parse_args()
for k in args.__dict__:
    print(k + ": " + str(args.__dict__[k]))

init_threshold = 0.01  # from source code 0.01
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

if args.model == "resnet20":
    net = resnet20().to(device)
elif args.model == "resnet50":
    net = resnet50().to(device)

state = torch.load("spoof/{}/model_step_0".format(args.t, args.dataset))
net.load_state_dict(state['net'])

cur_param = list((_.detach().clone() for _ in net.parameters()))
print("model have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))

criterion = nn.CrossEntropyLoss().to(device)

transform_apply = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform_train = transforms.Compose([
                                    # transforms.RandomRotation(),  # 随机旋转
                                    transforms.RandomCrop(32, padding=4),  # 填充后裁剪
                                    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                                    # transforms.Resize((32,32)),
                                    transforms.ToTensor(),
                                    # transforms.ColorJitter(brightness=1),  # 颜色变化。亮度
                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])

train_dataset = MyDataset("spoof/{}/dataset.txt".format(args.t, args.dataset), transform=transform_apply)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=False)
print(train_dataset.__len__)
print("--------- load checkpoint!----------")

state = torch.load("spoof/{}/model_step_0".format(args.t, args.dataset))
net.load_state_dict(state['net'])

print("--------- verify init!----------")
# verify init

p_list = []
for name, param in net.named_parameters():
    if 'fc' in name or 'conv' in name or 'linear' in name:
        if 'weight' in name:
            p_list.append(check_weights_initialization(param, 'resnet_cifar'))
        # elif 'bias' in name:
        #     weight = net.state_dict()[name.replace('bias', 'weight')]
        #     p_list.append(check_weights_initialization([weight, param], 'default_bias'))

if np.min(p_list) < init_threshold:
    print(f"The initialized weights does not follow the initialization strategy.\n"
          f"The minimum p value is {np.min(p_list)} < threshold ({init_threshold}).\n"
          f"The proof-of-learning is not valid.\n")
    init_valid = 0
else:
    print(f"The minimum p value is {np.min(p_list)} > threshold ({init_threshold}).\n"
          f"The proof-of-learning passed the initialization verification.\n")
    init_valid = 1

print("--------- verify start!----------")
max = [0, 0, 0, 0]
min = [1000, 1000, 1000, 1000]
sum = [0, 0, 0, 0]
avg = [0, 0, 0, 0]

valid_count = 0
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
dec_lr = [100, 150]
num_batch = train_dataset.__len__()/args.batchsize
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[round(i * num_batch) for i in dec_lr],
                                           gamma=0.1)
# print('111')
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
            state = torch.load(
                "spoof/{}/model_step_%d".format(args.t, args.dataset) % (args.k * step_count / args.k))
            net.load_state_dict(state['net'])

            target_param = list((_.detach().clone() for _ in net.parameters()))

            dist_list = [[] for i in range(len(order))]
            res = parameter_distance(target_param, dummy_param, order=order)
            for idx in range(len(res)):
                if res[idx] < min[idx]:
                    min[idx] = res[idx]
                if res[idx] > max[idx]:
                    max[idx] = res[idx]
                sum[idx] += res[idx]
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

            print("=> valid rate: (%d/ %d), total: %d" % (valid_count, step_count / args.k, args.t))
            print("")
            if step_count == args.iter:
                break
