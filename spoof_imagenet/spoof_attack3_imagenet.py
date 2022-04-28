# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import math
import copy
from pprint import pprint
from initial_break import generate_random, check, generate_random_bias, check_bias, generate_uniform, check_uniform

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader,random_split
print(torch.__version__, torchvision.__version__)

from model_imagenet import resnet18
from model import resnet20, resnet50, _weights_init
from folder import ImageFolder
from utils import *

########### config #########
parser = argparse.ArgumentParser(description='spoof PoL - spoof')
parser.add_argument('--iter', type=int, default=20300, help="Wt iterations")
parser.add_argument('--dataset', type=str, default="imagenet10", help="CIFAR10 or CIFAR100")
parser.add_argument('--model', type=str, default="resnet18", help="resnet20 or resnet50")
parser.add_argument('--lr', type=float, default=0.01, help = "lr")
parser.add_argument('--t', type=int, default=20, help = "t from W0' to Wn")
parser.add_argument('--k', type=int, default=100, help = "equal to freq")
parser.add_argument('--batchsize', type=int, default=128, help = "equal to freq")
parser.add_argument('--retry', type=int, default=30, help = "retry times")
parser.add_argument('--gd', type=int, default=2, help = "grad diff")
parser.add_argument('--nd', type=int, default=5, help = "noise diff")
parser.add_argument('--round', type=int, default=1, help="dlg training rounds")
parser.add_argument('--verify', type=int, default=1, help="verify or not")
parser.add_argument('--seed', type=int, default=0, help="lucky number")
parser.add_argument('--cut', type=int, default=50, help="cut batch")
args = parser.parse_args()
dif_k = 0
experiment_times = 1
different_t_settings = [10]
total_time_comsume =[[] for i in different_t_settings]
total_diff_dis = [[] for i in different_t_settings]
for diff_k in different_t_settings:
    args.t = diff_k
    for test_times in range(experiment_times):
        diff = [[], [], [], []]
        for k in args.__dict__:
            print(k + ": " + str(args.__dict__[k]))

        init_threshold = 0.01
        order = ['1', '2', 'inf', 'cos']
        threshold = [1000, 10, 0.1, 0.01]

        ########### config #########
        if args.seed > 0:
            seed = args.seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        print("Running on %s" % device)

        net = resnet18().to(device)

        # net=torch.nn.DataParallel(net)
        net.apply(_weights_init)

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


        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=int(args.batchsize*args.k/args.cut)
        )

        print("--------- load checkpoint!----------")

        net_load = resnet18().to(device)
        # net_load = torch.nn.DataParallel(net_load)
        state = torch.load("proof/{}/model_step_0".format(args.dataset),map_location={'cuda:0':'cuda:0'})
        # net_load.module.load_state_dict(state['net'])
        net_load.load_state_dict(state['net'])
        # before train parameter
        bt_param = list((_.detach().clone() for _ in net_load.parameters()))

        state = torch.load("proof/{}/model_step_".format(args.dataset) + str(args.iter),map_location={'cuda:0':'cuda:0'})
        # net_load.module.load_state_dict(state['net'])
        net_load.load_state_dict(state['net'])
        # after train parameter
        at_param = list((_.detach().clone() for _ in net_load.parameters()))

        # net.module.load_state_dict(state['net'])
        net.load_state_dict(state['net'])
        t = net.state_dict()
        p_list = []
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    x = generate_random(param).reshape_as(param)
                    # x = generate_uniform(param).reshape_as(param)
                    t[name].copy_(x)
                    param.data.copy_(x)
                    p_list.append(check(param))

        cur_param = list((_.detach().clone() for _ in net.parameters()))

        dis = 0
        for gx, gy in zip(at_param, bt_param):
            dis += ((gx - gy) ** 2).sum()

        dis_W0 = 0
        for gx, gy in zip(bt_param, cur_param):
            dis_W0 += ((gx - gy) ** 2).sum()

        dis_cur = 0
        for gx, gy in zip(at_param, cur_param):
            dis_cur += ((gx - gy) ** 2).sum()


        # print("sum distance: ", count)
        # print("avg distance: ", count / args.iter)
        print("|| Wt-W0 || distance: ", math.sqrt(dis.item()))
        print("|| W0-W0'|| distance: ", math.sqrt(dis_W0.item()))
        print("|| Wt-W0'|| distance: ", math.sqrt(dis_cur.item()))
        print("")

        print("--------- verify init!----------")
        # verify init

        p_list = []
        if args.model == "resnet18":
            for name, param in net.named_parameters():
                if 'fc' in name or 'conv' in name or 'linear' in name:
                    if 'weight' in name:
                        p_list.append(check_weights_initialization(param, 'resnet_cifar'))
                    # elif 'bias' in name:
                    #     weight = net.state_dict()[name.replace('bias', 'weight')]
                    #     p_list.append(check_weights_initialization([weight, param], 'default_bias'))
        elif args.model == "resnet50":
            print("TODO")
            # for name, param in net.named_parameters():
            #     if len(param.shape) == 4:
            #         p_list.append(check_weights_initialization(param, 'default'))
            #     elif 'weight' in name and 'fc' in name:
            #          p_list.append(check_weights_initialization(param, 'default'))
            #     elif 'bias' in name and ('fc' in name or 'linear' in name):
            #         weight = net.state_dict()[name.replace('bias', 'weight')]
            #         p_list.append(check_weights_initialization([weight, param], 'default_bias'))

        if np.min(p_list) < init_threshold:
            print(f"The initialized weights does not follow the initialization strategy.\n"
                  f"The minimum p value is {np.min(p_list)} < threshold ({init_threshold}).\n"
                  f"The proof-of-learning is not valid.\n")
            init_valid = 0
        else:
            print(f"The minimum p value is {np.min(p_list)} > threshold ({init_threshold}).\n"
                  f"The proof-of-learning passed the initialization verification.\n")
            init_valid = 1

        save_path = "spoof/{}/dataset".format(args.dataset)
        folder = os.path.exists(save_path)
        if not folder:
            os.makedirs(save_path)
        state = {'net': net.state_dict()}
        torch.save(state, "spoof/{}/model_step_0".format(args.dataset))
        f = open("spoof/{}/dataset.txt".format(args.dataset),"w")

        print("--------- spoof start!----------")

        img_idx=0
        tt_sum = 0
        original_weight = copy.deepcopy(net.state_dict())
        step_weight = copy.deepcopy(net.state_dict())
        at_weight = copy.deepcopy(net_load.state_dict())

        for key in at_weight.keys():
            tmp = torch.sub(at_weight[key], original_weight[key])
            step_weight[key] = torch.div(tmp, args.t)

        cur_param = list(_.detach().clone() for _ in net.parameters())
        k_step_dy_dx = list(torch.div(torch.sub(cur, at), (args.t)) for (at, cur) in zip(at_param, cur_param))

        valid_count = 0

        for ix, data in enumerate(train_loader):
            img, label = data
            dummy_label = label.to(device)
            break

        for i in range(args.t):
            print("step:", i+1)
            cur_param = list((_.detach().clone() for _ in net.parameters()))
            target_param = list((torch.add(-s, cur) for (s, cur) in zip(k_step_dy_dx, cur_param)))
            original_weight = copy.deepcopy(net.state_dict())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

            for j in range(args.cut):
                start_time = time.time()
                cur_net = net.state_dict()
                zero_dy_dx = list(torch.zeros(cur.shape).to(device) for cur in cur_param) # equal to zeros
                # generate dummy data and label
                for k in range(args.retry):
                    net.load_state_dict(cur_net)
                    dummy_data = torch.zeros(int(args.k * args.batchsize / args.cut), 3, 128, 128).to(device).requires_grad_(True)
                    optimizer = torch.optim.LBFGS([dummy_data, ], tolerance_grad=1e-3)
                    print(dummy_data.shape)

                    for iters in range(args.round):

                        def closure():
                            optimizer.zero_grad()
                            x = torch.clamp(dummy_data + img.to(device), 0, 1)
                            dummy_pred = net(x)
                            dummy_loss = criterion(dummy_pred, dummy_label)
                            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, zero_dy_dx): ###
                                grad_diff += ((gx + gy) ** 2).sum()

                            grad_diff += 0.4*(dummy_data ** 2).sum()
                            grad_diff.backward()

                            return grad_diff

                        optimizer.step(closure)

                        if (iters+1) % 1 == 0:
                            current_loss = closure()
                            d2r = (dummy_data** 2).sum().item()
                            d2grad = current_loss.item() - 0.4*d2r
                            print("step: %d iters: %d cut:%d loss: %.8f" % (i+1, iters+1, j+1, current_loss.item()))
                            print(" d2(r, 0):", d2r)
                            print(" d2(w, w):", d2grad)

                    if (d2grad < args.gd and d2r < args.nd):
                        break
                    for ix, data in enumerate(train_loader):
                        img, label = data
                        dummy_label = label.to(device)
                        break
                    print("retry!!")


                # show fake data
                fake_img = torch.clamp(dummy_data.cpu() + img.cpu(), 0, 1)

                for k in range(int(args.k / args.cut)):
                    saveImg(fake_img[k], "spoof/{}/dataset/%d.png".format(args.dataset) % img_idx)
                    f.write("spoof/{}/dataset/%d.png %d\n".format(args.dataset) % (img_idx, dummy_label[k].item()))
                    img_idx += 1

                for num_img in range(int(args.k / args.cut)):
                    x = fake_img[num_img * args.batchsize: (num_img+1) * args.batchsize].detach().to(device)
                    y = dummy_label[num_img * args.batchsize: (num_img+1) * args.batchsize].detach()

                    optimizer_net.zero_grad()
                    pred = net(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer_net.step()
                
                step_time = time.time() - start_time
                print("step time:", step_time)
                tt_sum += step_time

            if args.verify:
                if 1==1:
                    flag = 0
                    dummy_param = list((_.detach().clone() for _ in net.parameters()))

                    dis = 0
                    for gx, gy in zip(dummy_param, cur_param):
                        dis += ((gx - gy) ** 2).sum()
                    print("dum - cur:", dis.item())

                    dis = 0
                    for gx, gy in zip(target_param, cur_param):
                        dis += ((gx - gy) ** 2).sum()
                    print("tar - cur:", dis.item())

                    dis = 0
                    for gx, gy in zip(dummy_param, target_param):
                        dis += ((gx - gy) ** 2).sum()
                    print("dum - tar:", dis.item())

                    # target_param = list((torch.add(-s, cur) for (s, cur) in zip(step_dy_dx, cur_param)))
                    dist_list = [[] for i in range(len(order))]
                    res = parameter_distance(target_param, cur_param, order=order)
                    for j in range(len(order)):
                        dist_list[j].append(res[j])
                    dist_list = np.array(dist_list)
                    for k in range(len(order)):
                        print(f"{order[k]} : {np.average(dist_list[k])}")

                    dist_list = [[] for i in range(len(order))]
                    res = parameter_distance(target_param, dummy_param, order=order)
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

                    print("=> valid rate: (%d/ %d), total: %d" % (valid_count, i+1, args.t))
                    print("")
                
                diff = np.concatenate([diff, dist_list], 1)
                new_weight = copy.deepcopy(net.state_dict())
                for key in original_weight.keys():
                    new_weight[key] = torch.add(original_weight[key], step_weight[key])
                net.load_state_dict(new_weight)

                state = {'net': net.state_dict()}
                torch.save(state, "spoof/{}/model_step_%d".format(args.dataset) % (args.k * (i+1)))


        print("--------- conclusion ----------")
        if init_valid:
            print("init valid success")
        else:
            print("init valid not success")
        print("valid success steps (%d / %d)" % (valid_count, args.t))
        different_order = {order[0]:[],order[1]:[], order[2]:[],order[3]:[]}
        for k, name in enumerate(order):
            print("order:", name)
            print("distance:", np.mean(diff[k]), "  min: ", np.min(diff[k]), "  max: ", np.max(diff[k]))
            different_order[name]={"mean":np.mean(diff[k]),"min": np.min(diff[k]), "max": np.max(diff[k])}
        total_diff_dis[dif_k].append(different_order)
        total_time_comsume[dif_k].append(tt_sum)
    dif_k+=1
final_result=[{order[0]:0, order[1]:0, order[2]:0, order[3]:0} for i in range(len(different_t_settings))]
for i in range(len(different_t_settings)):
    for name in order:
        mean_tmp = []
        min_tmp = []
        max_tmp = []
        for j in range(experiment_times):
            mean_tmp.append(total_diff_dis[i][j][name]['mean'])
            min_tmp.append(total_diff_dis[i][j][name]['min'])
            max_tmp.append(total_diff_dis[i][j][name]['max'])
        final_result[i][name] = {"mean": np.mean(mean_tmp), "min": np.min(min_tmp), "max": np.max(max_tmp)}
    print("current t is ", different_t_settings[i], ",   final_eps: ", final_result[i])
    print("current t is ", different_t_settings[i], "total_time_comsume: ", total_time_comsume[i])