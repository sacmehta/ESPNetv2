# ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network

This repository contains the source code of our paper, [ESPNetv2](https://arxiv.org/abs/1811.11431).


## Structure
This repository contains source code and pretrained for the following:
 * **Object classification:** We provide source code along with pre-trained models at different network complexities 
 for the ImageNet dataset. Click [here](imagenet) for more details.
 * **Semantic segmentation:** We provide source code along with pre-trained models on the Cityscapes dataset. Check [here](segmentation) for more details. 
 
## Requirements
 
To run this repository, you should have following softwares installed:
 * PyTorch - We tested with v0.4.1
 * OpenCV - We tested with version 3.4.3
 * Python3 - Our code is written in Python3. We recommend to use [Anaconda](https://www.anaconda.com/) for the same.
 
 ## Instructions to install Pytorch and OpenCV with Anaconda
 
Assuming that you have installed Anaconda successfully, you can follow the following instructions to install the packeges:
 
### PyTorch
```
conda install pytorch torchvision -c pytorch
```

Once installed, run the following commands in your terminal to verify the version:
```
import torch
torch.__version__ 
```
This should print something like this `0.4.1.post2`. 

If your version is different, then follow PyTorch website [here](https://pytorch.org/) for more details.

### OpenCV
```
conda install pip
pip install --upgrade pip
pip install opencv-python
```

Once installed, run the following commands in your terminal to verify the version:
```
import cv2
cv2.__version__ 
```
This should print something like this `3.4.3`.

 
## Implementation note

You will see that `EESP` unit, the core building block of the ESPNetv2 architecture, has a `for` loop to process the input at different dilation rates. 
You can parallelize it using **Streams** in PyTorch. It improves the inference speed. 

A snippet to parallelize a `for` loop in pytorch is shown below:
```
# Sequential version
output = [] 
a = torch.randn(1, 3, 10, 10)
for i in range(4):
    output.append(a)
torch.cat(output, 1)
```

``` 
# Parallel version
num_branches = 4
streams = [(idx, torch.cuda.Stream()) for idx in range(num_branches)]
output = []
a = torch.randn(1, 3, 10, 10)
for idx, s in streams:
    with torch.cuda.stream(s):
        output.append(a)
torch.cuda.synchronize()
torch.cat(output, 1)
```

**Note:** We have not tested it (for training as well as inference) across multiple GPUs. If you want to use Streams and facing issues, please use PyTorch forums to resolve your queries. 
