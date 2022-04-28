# "Adversarial Examples" for Proof-of-Learning

This repository is an implementation of the paper ["Adversarial Examples" for Proof-of-Learning](https://arxiv.org/abs/2108.09454). In this paper, we introduce the a method that successfully 
attack the concept of proof-of-learning in ML  [Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633). 
Inspired by research on adversarial examples. For more details, please read the paper.

We test our code on two datasets: CIFAR-10,CIFAR-100 and a subset of ImageNet. 

### Dependency
Our code is implemented and tested on PyTorch. Following packages are used:
```
numpy
pytorch==1.6.0
torchvision==0.7.0
scipy==1.6.0
```
### Spoof
To spoof a model on CIFAR-10 and CIFAR-100 with different attacks:
```
python spoof_cifar/attack.py --attack [1,2,or 3 for three attacks] --dataset ['CIFAR100' or 'CIFAR10'] --model [models defined in model.py] --t [spoof steps] --verify [1 or 0]
```
We use 'resnet20' for CIFAR-10 and 'resnet50' for CIFAR-100. t is the spoof steps, denoted by T in the paper and here t =\frac{T}{100}.
We use '--cut' to fit different devices when 'cut' is set to 100, attack3 is same with attack2.

To spoof a model on the subset of ImageNet with different attacks:
```
python spoof_imagenet/spoof_imagenet.py --t [spoof steps] --verify [1 or 0]
python spoof_imagenet/spoof_attack3_imagenet.py --t [spoof steps] --verify [1 or 0]
```
'verify' is to control whether to verify the model.

### Model generation
To train a model and create a proof-of-learning:
```
python PoL/train.py --save-freq [checkpointing interval] --dataset ['CIFAR100' or 'CIFAR10'] --model ['resnet50' or 'resnet20']
python spoof_imagenet/train.py --freq [checkpointing interval]
```
`save-freq` is checkpointing interval, denoted by k in the paper[Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633). 
To spoof the model, put the generated model in 'spoof_cifar/proof/[dataset]'. 
To generated CIFAR10 and CIFAR100 models with high accuracy:
```
python spoof_cifar/train.py --save-freq [checkpointing interval] --dataset ['CIFAR100' or 'CIFAR10'] --model ['resnet50' or 'resnet20']
```

To verify a given proof-of-learning:
```
python PoL/verify.py --model-dir [path/to/the/proof] --dist [distance metric] --q [query budget] --delta [slack parameter]
python spoof_imagenet/verify.py --k [checkpointing interval]

```
Setting q to 0 or smaller will verify the whole proof, otherwise the top-q iterations for each epoch will be verified. More information about `q` and `delta` can be found in the paper. For `dist`, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. Note that if using more than one, the top-q iterations in terms of all distance metrics will be verified.

Please make sure `lr`, `batch-sizr`, `epochs`, `dataset`, `model`, and `save-freq` are consistent with what used in `train.py`.


### Questions or suggestions
If you have any questions or suggestions, feel free to raise an issue or send me an email at zhangrui98@zju.edu.cn

### Citing this work
If you use this repository for academic research, you are highly encouraged (though not required) to cite our paper:
```
@article{zhang2021adversarial,
  title={" Adversarial Examples" for Proof-of-Learning},
  author={Zhang, Rui and Liu, Jian and Ding, Yuan and Wu, Qingbiao and Ren, Kui},
  journal={arXiv preprint arXiv:2108.09454},
  year={2021}
}
```
