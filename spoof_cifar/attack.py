# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import math
import copy
from pprint import pprint
from initial_break import *

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
print(torch.__version__, torchvision.__version__)
from model import resnet20, resnet50, _weights_init
from utils import *
from spoof_attack1 import attack1
from spoof_attack2 import attack2
from spoof_attack3 import attack3
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spoof PoL - spoof')
    parser.add_argument('--attack', type=int, default=2, help="1,2,3 for Attack1,2,3")
    parser.add_argument('--iter', type=int, default=78125, help="Wt iterations")
    parser.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10 or CIFAR100")
    parser.add_argument('--model', type=str, default="resnet20", help="resnet20 or resnet50")
    parser.add_argument('--lr', type=float, default=0.01, help = "lr")
    parser.add_argument('--t', type=int, default=20, help = "t from W0' to Wn")
    parser.add_argument('--k', type=int, default=100, help = "equal to freq")
    parser.add_argument('--batchsize', type=int, default=128, help = "equal to freq")
    parser.add_argument('--retry', type=int, default=30, help = "retry times")
    parser.add_argument('--gd', type=int, default=15, help = "grad diff")
    parser.add_argument('--nd', type=int, default=20, help = "noise diff")
    parser.add_argument('--round', type=int, default=1, help="dlg training rounds")
    parser.add_argument('--verify', type=int, default=1, help="verify or not")
    parser.add_argument('--seed', type=int, default=0, help="lucky number")
    parser.add_argument('--cut', type=int, default=100, help="cut batch")
    args = parser.parse_args()
    if args.attack == 3:
        attack3(args)
    elif args.attack == 1:
        attack1(args)
    else:
        attack2(args)
