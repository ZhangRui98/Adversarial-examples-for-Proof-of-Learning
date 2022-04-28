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


def attack1(args):
    dif_k = 0
    experiment_times = 1
    different_t_settings = [1]
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
                device = "cuda:2"
            print("Running on %s" % device)

            if args.model == "resnet20":
                net = resnet20().to(device)
            elif args.model == "resnet50":
                net = resnet50().to(device)
            net.apply(_weights_init)

            cur_param = list((_.detach().clone() for _ in net.parameters()))
            print("model have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))

            criterion = nn.CrossEntropyLoss().to(device)

            transform_apply = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])])  

            if args.dataset == "CIFAR10":
                train_data = datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transform_apply)
                test_data = datasets.CIFAR10("data/cifar10", train=False, download=True, transform=transform_apply)
                test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)
                # _, train_data = random_split(dataset=train_data, lengths=[25000,25000])
            elif args.dataset == "CIFAR100":
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
                train_data = datasets.CIFAR100("data/cifar100", train=True, download=True, transform=transform)
                test_data = datasets.CIFAR100("data/cifar100", train=False, download=True, transform=transform_test)
                test_loader = DataLoader(dataset=test_data, batch_size=args.batchsize, shuffle=False)
            train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)

            print("--------- load checkpoint!----------")
            if args.dataset == "CIFAR10":
                net_load = resnet20().to(device)
            elif args.dataset == "CIFAR100":
                net_load = resnet50().to(device)
            state = torch.load("proof/{}/model_step_0".format(args.dataset),map_location={'cuda:2':'cuda:2'})
            net_load.load_state_dict(state['net'])
            # before train parameter
            bt_param = list((_.detach().clone() for _ in net_load.parameters()))

            state = torch.load("proof/{}/model_step_".format(args.dataset) + str(args.iter),map_location={'cuda:1':'cuda:2'})
            net_load.load_state_dict(state['net'])

            # after train parameter
            at_param = list((_.detach().clone() for _ in net_load.parameters()))

            # generate initial weights
            net.load_state_dict(state['net'])
            Initial_gen(args, net)
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
            init_valid = Verify_init(args, net, init_threshold)
            if init_valid == 0 :
                exit(0)

            save_path = "spoof/{}/dataset".format(args.dataset)
            folder = os.path.exists(save_path)
            if not folder:
                os.makedirs(save_path)
            state = {'net': net.state_dict()}
            torch.save(state, "spoof/{}/model_step_0".format(args.dataset))
            f = open("spoof/{}/dataset.txt".format(args.dataset),"w")



            iter_count = 0
            optimizer = torch.optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=1e-4)
            total_time_start = time.time()
            # net.apply(_weights_init)
            train_time=time.time()
            for epoch in range(227):
                net.train()
                stt = time.time()
                print("epoch:", epoch + 1)
                for i, data in enumerate(train_loader):
                    iter_count = iter_count + 1  # count iters num

                    optimizer.zero_grad()
                    inputs, labels = data
                    pred = net(inputs.to(device))
                    loss = criterion(pred, labels.to(device))

                    loss.backward()
                    optimizer.step()
                net.eval()
                train_acc = test_accuracy(train_loader, net)
                test_acc = test_accuracy(test_loader, net)
                print("iter:", iter_count, "loss: %.3f train acc: %.3f test acc: %.3f" % (loss.item(), train_acc, test_acc))
                print("time: ",time.time()-stt)
            print("train time:", time.time()-train_time)
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
            # for ix, data in enumerate(train_loader):
            #     img, label = data
            #     # if label.to(device) == dummy_label:
            #     #     break
            #     dummy_label = label.to(device)
            #     break

            for i in range(args.t):
                print("step:", i+1)
                cur_param = list((_.detach().clone() for _ in net.parameters()))
                target_param = list((torch.add(-s, cur) for (s, cur) in zip(k_step_dy_dx, cur_param)))
                original_weight = copy.deepcopy(net.state_dict())
                # target_param = list((_.detach().clone() for _ in net.parameters()))
                optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01,momentum=0.9, weight_decay=1e-4)
                loader_id=0
                for j in range(args.k):
                    cur_net = net.state_dict()
                    start_time = time.time()
                    tmp_param = list((_.detach().clone() for _ in net.parameters()))
                    step_dy_dx = list(torch.sub(tmp, cur) for (cur, tmp) in zip(cur_param, tmp_param))
                    zero_dy_dx = list(torch.zeros(cur.shape).to(device) for cur in cur_param) # equal to zeros
                    # generate dummy data and label
                    for k in range(args.retry):
                        net.load_state_dict(cur_net)
                        dummy_data = torch.zeros(args.batchsize, 3, 32, 32).to(device).requires_grad_(True)
                        optimizer = torch.optim.LBFGS([dummy_data, ], max_iter=100)
                        for ix, data in enumerate(train_loader, start=loader_id):
                            img, label = data
                            # if label.to(device) == dummy_labeld
                            #     break
                            dummy_label = label.to(device)
                            break

                        for iters in range(args.round):

                            def closure():
                                optimizer.zero_grad()
                                x = torch.clamp(dummy_data + img.to(device), 0, 1)
                                dummy_pred = net(x)
                                dummy_loss = criterion(dummy_pred, dummy_label)
                                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True, retain_graph=True)
                                # d2 = torch.autograd.functional.jacobian(dummy_loss, net.parameters())


                                grad_diff = 0
                                for gx, gy in zip(dummy_dy_dx, k_step_dy_dx):
                                    grad_diff += ((gx + gy) ** 2).sum()

                                grad_diff += 1*(dummy_data ** 2).sum()
                                grad_diff.backward()

                                return grad_diff

                            optimizer.step(closure)

                            if (iters+1) % 1 == 0:
                                current_loss = closure()
                                d2r = (dummy_data** 2).sum().item()
                                d2grad = current_loss.item() - d2r
                                print("step: %d iters: %d loss: %.8f" % (j+1, iters+1, current_loss.item()))
                                print(" d2(r, 0):", d2r)
                                print(" d2(w, w):", d2grad)

                        if (d2grad < args.gd and d2r < args.nd):
                            print("label:", label)
                            break
                        for ix, data in enumerate(train_loader):
                            img, label = data
                            print("label:",label)
                            # if label.to(device) == dummy_label:
                            #     break
                            dummy_label = label.to(device)
                            break
                        loader_id += 1
                        if loader_id > 25000:
                            loader_id = 0
                        print("retry!!")


                    # show fake data
                    fake_img = torch.clamp(dummy_data.cpu() + img.cpu(), 0, 1)

                    for k in range(args.batchsize):
                        saveImg(fake_img[k], "spoof/{}/dataset/%d.png".format(args.dataset) % img_idx)
                        f.write("spoof/{}/dataset/%d.png %d\n".format(args.dataset) % (img_idx, dummy_label[k].item()))
                        img_idx += 1
                        # plt.imshow(np.transpose(img[k].numpy(),(1,2,0)))
                        # plt.show()
                        # plt.imshow(np.transpose(fake_img[k].detach().numpy(),(1,2,0)))
                        # plt.show()
                    step_time = time.time() - start_time
                    print("step time:", step_time)
                    tt_sum += step_time

                    if args.verify:
                        flag = 0

                        x = fake_img.detach().to(device)
                            #x = torch.unsqueeze(x, 0).to(device)
                        y = dummy_label.detach()
                            #y = torch.unsqueeze(y, 0).to(device)

                        optimizer_net.zero_grad()
                        pred = net(x)
                        loss = criterion(pred, y)
                        loss.backward()
                        optimizer_net.step()

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
                print("total_time: ",tt_sum)

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

    print("tottal time: ", time.time()-total_time_start)

if __name__ == '__main__':
########### config #########
    parser = argparse.ArgumentParser(description='spoof PoL - spoof')
    parser.add_argument('--iter', type=int, default=78125, help="Wt iterations")
    parser.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10 or CIFAR100")
    parser.add_argument('--model', type=str, default="resnet20", help="resnet20 or resnet50")
    parser.add_argument('--lr', type=float, default=0.01, help = "lr")
    parser.add_argument('--t', type=int, default=1, help = "t from W0' to Wn")
    parser.add_argument('--k', type=int, default=1, help = "equal to freq")
    parser.add_argument('--batchsize', type=int, default=128, help = "equal to freq")
    parser.add_argument('--retry', type=int, default=1, help = "retry times")
    parser.add_argument('--gd', type=float, default=500, help = "grad diff")
    parser.add_argument('--nd', type=int, default=500, help = "noise diff")
    parser.add_argument('--round', type=int, default=10, help="dlg training rounds")
    parser.add_argument('--verify', type=int, default=1, help="verify or not")
    parser.add_argument('--seed', type=int, default=0, help="lucky number")
    args = parser.parse_args()