import torch
import numpy as np
from model import _weights_init, resnet20
import torch.nn as nn
from scipy import stats
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
# from utils import *

seed = 777
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
net = resnet20()
net.apply(_weights_init)


def ks_test(reference, rvs):
    device ='cpu'
    with torch.no_grad():
        ecdf = torch.arange(rvs.shape[0]).float() / torch.tensor(rvs.shape)
        return torch.max(torch.abs(reference(torch.sort(rvs)[0].to(device)).to(device) - ecdf.to(device)))


def check(param):
    fan = nn.init._calculate_correct_fan(param, 'fan_in')
    gain = nn.init.calculate_gain('leaky_relu', 0)
    std = gain / np.sqrt(fan)
    reference = torch.distributions.normal.Normal(0, std).cdf
    param = param.reshape(-1)
    ks_stats = ks_test(reference, param).cpu().item()
    return stats.kstwo.sf(ks_stats, param.shape[0])

def generate_random(param):
    fan = nn.init._calculate_correct_fan(param, 'fan_in')
    gain = nn.init.calculate_gain('leaky_relu', 0)
    std = gain / np.sqrt(fan)
    reference = torch.distributions.normal.Normal(0, std).cdf
    size = param.shape
    param = param.reshape(-1)
    device = 'cpu'
    with torch.no_grad():
        ecdf = torch.arange(param.shape[0]).float() / torch.tensor(param.shape)
        # print(reference)
        # print(reference(torch.sort(rvs)[0]))
        # print(ecdf)
        rvs = param.clone().detach().cpu()
        sorted_param = torch.sort(rvs)
        num = rvs.shape[0]
        sig = np.sqrt(std)
        e = torch.distributions.normal.Normal(0, std).sample(rvs.shape)
        while check(e.reshape(size)) < 0.01:
            e = torch.distributions.normal.Normal(0, std).sample(rvs.shape)
        sorted_e = torch.sort(e)
        rvs[sorted_param.indices] = sorted_e.values
        return rvs.clone().detach().requires_grad_(True)


def check_bias(param,weight):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / np.sqrt(fan_in)
    reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    param = param.reshape(-1)
    ks_stats = ks_test(reference, param.cpu()).cpu().item()
    return stats.kstwo.sf(ks_stats, param.shape[0])


def generate_random_bias(param,weight):
    # fan = nn.init._calculate_correct_fan(param, 'fan_in')
    # gain = nn.init.calculate_gain('leaky_relu', 0)
    # std = gain / np.sqrt(fan)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / np.sqrt(fan_in)
    reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    # reference = torch.distributions.normal.Normal(0, std).cdf
    size = param.shape
    param = param.reshape(-1)
    device = 'cpu'
    with torch.no_grad():
        ecdf = torch.arange(param.shape[0]).float() / torch.tensor(param.shape)
        # print(reference)
        # print(reference(torch.sort(rvs)[0]))
        # print(ecdf)
        rvs = param.clone().detach().cpu()
        sorted_param = torch.sort(rvs)
        num = rvs.shape[0]
        # sig = np.sqrt(std)
        e = torch.distributions.uniform.Uniform(-bound, bound).sample(rvs.shape)
        while check_bias(e.reshape(size), weight) < 0.01:
            e = torch.distributions.uniform.Uniform(-bound, bound).sample(rvs.shape)
        sorted_e = torch.sort(e)
        rvs[sorted_param.indices] = sorted_e.values
        return rvs.clone().detach().requires_grad_(True)


def check_uniform(param):
    fan = nn.init._calculate_correct_fan(param, 'fan_in')
    gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    param = param.reshape(-1)
    ks_stats = ks_test(reference, param.cpu()).cpu().item()
    return stats.kstwo.sf(ks_stats, param.shape[0])


def generate_uniform(param):
    # fan = nn.init._calculate_correct_fan(param, 'fan_in')
    # gain = nn.init.calculate_gain('leaky_relu', 0)
    # std = gain / np.sqrt(fan)
    fan = nn.init._calculate_correct_fan(param, 'fan_in')
    gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std
    reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    # reference = torch.distributions.normal.Normal(0, std).cdf
    size = param.shape
    param = param.reshape(-1)
    device = 'cpu'
    with torch.no_grad():
        ecdf = torch.arange(param.shape[0]).float() / torch.tensor(param.shape)
        # print(reference)
        # print(reference(torch.sort(rvs)[0]))
        # print(ecdf)
        rvs = param.clone().detach().cpu()
        sorted_param = torch.sort(rvs)
        num = rvs.shape[0]
        # sig = np.sqrt(std)
        e = torch.distributions.uniform.Uniform(-bound, bound).sample(rvs.shape)
        while check_uniform(e.reshape(size)) < 0.01:
            e = torch.distributions.uniform.Uniform(-bound, bound).sample(rvs.shape)
        sorted_e = torch.sort(e)

        rvs[sorted_param.indices] = sorted_e.values
        return rvs.clone().detach().requires_grad_(True)




# net_load = resnet20()
# state = torch.load("checkpoint/model_step_20000")
# net_load.load_state_dict(state['net'])
#
# p_list = []
# t = net_load.state_dict()
# for name, param in net_load.named_parameters():
#     if 'fc' in name or 'conv' in name or 'linear' in name:
#         if 'weight' in name:
#             x = generate_random(param).reshape_as(param)
#             t[name].copy_(x)
#             param.data.copy_(x)
#             p_list.append(check(param))
#
# init_threshold = 0.01
# if np.min(p_list) < init_threshold:
#     print(f"The initialized weights does not follow the initialization strategy.\n"
#           f"The minimum p value is {np.min(p_list)} < threshold ({init_threshold}).\n"
#           f"The proof-of-learning is not valid.\n")
#     init_valid = 0
# else:
#     print(f"The minimum p value is {np.min(p_list)} > threshold ({init_threshold}).\n"
#           f"The proof-of-learning passed the initialization verification.\n")
#     init_valid = 1
