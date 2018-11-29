#  ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network

This repository contains the source code that we used for semantic segmentation in our paper of our paper, [ESPNetv2](https://arxiv.org/abs/1811.11431).


## Training and Evaluation on the Cityscapes dataset

Below are the commands to train and test the network at scale `s=1.0`.

### Training

You can train the network with and without pre-trained weights on the ImageNet dataset using following commands:

 * Without pre-trained weights
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size 10 --s 1.0
```

 * With pre-trained weights
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size 10 --s 1.0 --pretrained ../imagenet/pretrained_weights/espnetv2_s_1.0.pth
```

### Evaluation
To evaluate our pre-trained models (or the ones trained by you), you can use `gen_cityscapes.py` file.

Use below command to evaluate the performance of our model at scale `s=1.0` on the Cityscapes dataset.
```
CUDA_VISIBLE_DEVICES=0 python gen_cityscapes.py --s 1.0 --pretrained ./pretrained_models/espnetv2_segmentation_s_1.0.pth
```

## Performance on the Cityscapes dataset

Our model is 2% accurate than the [ESPNet](https://github.com/sacmehta/ESPNet) while being much more efficient. See more details in the paper. 

| Model | Params | FLOPs | mIOU | Pretrained Model |  
| -- | -- | -- | -- | -- |
| ESPNet | 0.364 M  | 424 M   |  60.3 | [here](https://github.com/sacmehta/ESPNet)  |
| ESPNetv2 (s=0.5) | 99 K  | 54 M  | 54.7 | [here](pretrained_models/espnetv2_segmentation_s_0.5.pth) |
| ESPNetv2 (s=1.5) |  725 K | 322 M  | 62.1  | [here](pretrained_models/espnetv2_segmentation_s_1.5.pth) |

