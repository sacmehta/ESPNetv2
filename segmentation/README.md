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

Our model achieves an mIOU of 55.64 on the CamVid test set. We used the dataset splits (train/val/test) provided [here](https://github.com/alexgkendall/SegNet-Tutorial). We trained the models at a resolution of 480x360. For comparison  with other models, see [SegNet paper](https://ieeexplore.ieee.org/document/7803544/).

Note: We did not use the 3.5K dataset for training which was used in the SegNet paper.

| Model | Params | Model size | mIOU | Pretrained Model |  
| -- | -- | -- |
| ESPNet |   |   |  |   |
| ESPNetv2 (s=0.5) |   |   |  |   |
 | ESPNetv2 (s=1.0) |   |   |  |   |
| ESPNetv2 (s=1.5) |   |   |  |   |
