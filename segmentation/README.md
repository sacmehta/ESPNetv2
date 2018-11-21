#  ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

This repository contains the source code of our paper, [ESPNetv2]().


## Training and Evaluation on the Cityscapes dataset

Below are the commands to train and test the network at scale `s=1.0`.

### Training

You can train the network with and without pre-trained weights on the ImageNet dataset using following commands:

 * With pre-trained weights
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch-size 512 --s 1.0 --pretrained <location of the pretrained ImageNet weights>
```
 * Without pre-trained weights
``` 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch-size 512 --s 1.0
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

