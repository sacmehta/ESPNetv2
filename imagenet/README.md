# [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431)

This repository contains the source code for training on the ImageNet dataset along with the pre-trained models

## Training and Evaluation on the ImageNet dataset

Below are the commands to train and test the network at scale `s=1.0`.

### Training
To train the network, you can use the following command:

``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch-size 512 --s 1.0 --data <Location of ImageNet dataset>
```

### Evaluation
To evaluate our pretrained models (or the ones trained by you), you can use `evaluate.py` file.

Use below command to evaluate the performance of our model at scale `s=1.0` on the ImageNet dataset.
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --batch-size 512 --s 1.0 --weightFile ./pretrained_weights/espnetv2_s_1.0.pth --data <Location of ImageNet dataset>
```

## Results and pre-trained models
We release the pre-trained models at different computational complexities. Following state-of-the-art methods, we measure top-1 accuracy on a  
cropped center view of size 224x224.

Below table provide details about the performance of our model on the ImageNet validation set at different computational complexities along with links to download the pre-trained weights.


| s | Params | FLOPs  | top-1 (val) | Link |
| -------- |--------|--------|-------| -------|
| 0.5 | 1.24   | 28.37  | 57.7  | [here](pretrained_weights/espnetv2_s_0.5.pth) |
| 1.0 | 1.67   | 85.72  | 66.1  | [here](pretrained_weights/espnetv2_s_1.0.pth) |
| 1.25 | 1.96   | 123.39 | 67.9  | [here](pretrained_weights/espnetv2_s_1.25.pth) |
| 1.5 | 2.31   | 168.6  | 69.2  |  [here](pretrained_weights/espnetv2_s_1.5.pth) |
| 2.0 | 3.49   | 284.8  | 72.1  | [here](pretrained_weights/espnetv2_s_2.0.pth) |


## ImageNet dataset preparation
To prepare the dataset, follow instructions [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

