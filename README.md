# ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network

**IMPORTANT NOTE 1 (7 June, 2019)**: We have released new code base that supports several datasets and models, including ESPNetv2. Please see [here](https://github.com/sacmehta/EdgeNets) for more details.

**IMPORTANT NOTE 2 (7 June, 2019)**: This repository is obsolete and we are not maintaining it anymore.

This repository contains the source code of our paper, [ESPNetv2](https://arxiv.org/abs/1811.11431) which is accepted for publication at CVPR'19. 

***Note:*** New segmentation models for the PASCAL VOC and the Cityscapes are coming soon. Our new models achieves mIOU of [68.0](http://host.robots.ox.ac.uk:8080/anonymous/DAMVRR.html) and [66.15](https://www.cityscapes-dataset.com/anonymous-results/?id=2267c613d55dd75d5301850c913b1507bf2f10586ca73eb8ebcf357cdcf3e036) on the PASCAL VOC and the Cityscapes test sets, respectively. 

<table>
    <tr>
        <td colspan=2 align="center"><b>Real-time semantic segmentation using ESPNetv2 on iPhone7 (see <a href='https://github.com/sacmehta/EdgeNets'>EdgeNets</a>  for details)<b></td>
    </tr>
    <tr>
        <td>
            <img src="https://github.com/sacmehta/EdgeNets/blob/master/images/espnetv2_iphone7_video_1.gif" alt="Seg demo on iPhone7"></img>
        </td>
        <td>
            <img src="https://github.com/sacmehta/EdgeNets/blob/master/images/espnetv2_iphone7_video_2.gif" alt="Seg demo on iPhone7"></img>
        </td>
    </tr>
</table>

## Comparison with SOTA methods
Compared to state-of-the-art efficient networks, our network delivers competitive performance while being much more **power efficient**. Sample results are shown in below figure. For more details, please read our paper.

  <table width="100%" align="center" border=1>
    <tr>
        <td width="50%">
            <img src="/images/effCompare.png" width="80%"/>
        </td>
        <td width="50%">
            <img src="/images/powerTX2.png" width="80%"/>
        </td>
    </tr>
    <tr>
        <td>
          <p align="center"><b>FLOPs vs. accuracy on the ImageNet dataset</b></p>
        </td>
        <td>
          <p align="center"><b>Power consumption on TX2 device</b></b>
      </td>
    <tr>
  </table>



If you find our project useful in your research, please consider citing:

```
@inproceedings{mehta2019espnetv2,
  title={ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network},
  author={Sachin Mehta and Mohammad Rastegari and Linda Shapiro and Hannaneh Hajishirzi},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{mehta2018espnet,
  title={ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation},
  author={Sachin Mehta and Mohammad Rastegari and Anat Caspi and Linda Shapiro and Hannaneh Hajishirzi},
  booktitle={ECCV},
  year={2018}
}
```

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

**Note:** 
 * we have used above strategy to measure inference related statistics, including power consumption and run time on a single GPU.
 * We have not tested it (for training as well as inference) across multiple GPUs. If you want to use Streams and facing issues, please use PyTorch forums to resolve your queries. 
