import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import torch.nn as nn
from torch.utils.data import Dataset
from initial_break import *
from PIL import Image
import torchvision


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def get_parameters(para):
    # get weights from a torch model as a list of numpy arrays
    parameter = torch.cat([i.data.reshape([-1]) for i in para])
    return parameter

def parameter_distance(model1, model2, order=2):
    # compute the difference between 2 checkpoints
    weights1 = get_parameters(model1)
    weights2 = get_parameters(model2)
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    for o in orders:
        if o == 'inf':
            o = np.inf
        if o == 'cos' or o == 'cosine':
            res = (1 - torch.dot(weights1, weights2) /
                   (torch.norm(weights1) * torch.norm(weights2))).cpu().numpy()
        else:
            if o != np.inf:
                try:
                    o = int(o)
                except:
                    raise TypeError("input metric for distance is not understandable")
            res = torch.norm(weights1 - weights2, p=o).cpu().numpy()
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list

def ks_test(reference, rvs):
    device = torch.device('cpu')
    with torch.no_grad():
        ecdf = torch.arange(rvs.shape[0]).float() / torch.tensor(rvs.shape)
        return torch.max(torch.abs(reference(torch.sort(rvs)[0]).to(device) - ecdf.to(device)))


def check_weights_initialization(param, method):
    if method == 'default':
        # kaimin uniform (default for weights of nn.Conv and nn.Linear)
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', np.sqrt(5))
        std = gain / np.sqrt(fan)
        bound = np.sqrt(3.0) * std
        reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    elif method == 'resnet_cifar':
        # kaimin normal
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('leaky_relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.normal.Normal(0, std).cdf
    elif method == 'resnet':
        # kaimin normal (default in conv layers of pytorch resnet)
        fan = nn.init._calculate_correct_fan(param, 'fan_out')
        gain = nn.init.calculate_gain('relu', 0)
        std = gain / np.sqrt(fan)
        reference = torch.distributions.normal.Normal(0, std).cdf
    elif method == 'default_bias':
        # default for bias of nn.Conv and nn.Linear
        weight, param = param
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / np.sqrt(fan_in)
        reference = torch.distributions.uniform.Uniform(-bound, bound).cdf
    
    else:
        raise NotImplementedError("Input initialization strategy is not implemented.")

    param = param.reshape(-1)
    ks_stats = ks_test(reference, param.cpu()).cpu().item()
    return stats.kstwo.sf(ks_stats, param.shape[0])

def test_accuracy(test_loader, model):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            img, labels = data
            img, labels = img.to(device), labels.to(device)
            out = model(img)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            # if (total >= num):
            #     break
    
    return correct / total



#以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    #stpe1:初始化
    def __init__(self, txt, transform=None, target_transform=None, gray = False):
        self.gray = gray
        fh = open(txt, 'r')#打开标签文件
        imgs = []#创建列表，装东西
        for line in fh:#遍历标签文件每行
            line = line.rstrip()#删除字符串末尾的空格
            words = line.split()#通过空格分割字符串，变成列表
            imgs.append((words[0],int(words[1])))#把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):#检索函数
        fn, label = self.imgs[index]#读取文件名、标签
        if (self.gray == True):
            img = Image.open(fn).convert('L') #灰度图
        else:
            img = Image.open(fn).convert('RGB')#通过PIL.Image读取图片
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

def saveImg(img, filename, Gray=False):
     # torchvision.utils.save_image(img, imgPath)
     # 改写：torchvision.utils.save_image
     grid = torchvision.utils.make_grid(img, nrow=8, padding=2, pad_value=0,
                                        normalize=False, range=None, scale_each=False)
     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
     im = Image.fromarray(ndarr)
     # im.show()
     if Gray:
         im.convert('L').save(filename)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
     else:
         im.save(filename)


def Initial_gen(args, net):
    t = net.state_dict()
    p_list = []
    if args.model == 'resnet20':
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    x = generate_random(param).reshape_as(param)
                    # x = generate_uniform(param).reshape_as(param)
                    t[name].copy_(x)
                    param.data.copy_(x)
                    p_list.append(check(param))
                elif 'bias' in name:
                    weight = net.state_dict()[name.replace('bias', 'weight')]
                    x = generate_random_bias(param, weight).reshape_as(param)
                    t[name].copy_(x)
                    param.data.copy_(x)
                    p_list.append(check_bias(param, weight))
    elif args.model =='resnet50':
        for name, param in net.named_parameters():
            if len(param.shape) == 4:
                # x = generate_random(param).reshape_as(param)
                x = generate_uniform(param).reshape_as(param)
                t[name].copy_(x)
                param.data.copy_(x)
                p_list.append(check(param))
            elif 'weight' in name and 'fc' in name:
                x = generate_uniform(param).reshape_as(param)
                t[name].copy_(x)
                param.data.copy_(x)
                p_list.append(check(param))
            elif 'bias' in name and ('fc' in name or 'linear' in name):
                weight = net.state_dict()[name.replace('bias', 'weight')]
                x = generate_random_bias(param, weight).reshape_as(param)
                t[name].copy_(x)
                param.data.copy_(x)
                p_list.append(check_bias(param, weight))


def Verify_init(args, net, init_threshold):
    p_list = []
    if args.model == "resnet20":
        for name, param in net.named_parameters():
            if 'fc' in name or 'conv' in name or 'linear' in name:
                if 'weight' in name:
                    p_list.append(check_weights_initialization(param, 'resnet_cifar'))
                elif 'bias' in name:
                    weight = net.state_dict()[name.replace('bias', 'weight')]
                    p_list.append(check_weights_initialization([weight, param], 'default_bias'))
    elif args.model == "resnet50":
        for name, param in net.named_parameters():
            if len(param.shape) == 4:
                p_list.append(check_weights_initialization(param, 'default'))
            elif 'weight' in name and 'fc' in name:
                p_list.append(check_weights_initialization(param, 'default'))
            elif 'bias' in name and ('fc' in name or 'linear' in name):
                weight = net.state_dict()[name.replace('bias', 'weight')]
                p_list.append(check_weights_initialization([weight, param], 'default_bias'))

    if np.min(p_list) < init_threshold:
        print(f"The initialized weights does not follow the initialization strategy.\n"
            f"The minimum p value is {np.min(p_list)} < threshold ({init_threshold}).\n"
            f"The proof-of-learning is not valid.\n")
        init_valid = 0
    else:
        print(f"The minimum p value is {np.min(p_list)} > threshold ({init_threshold}).\n"
            f"The proof-of-learning passed the initialization verification.\n")
        init_valid = 1
    return init_valid