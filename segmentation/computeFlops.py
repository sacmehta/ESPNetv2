import loadData as ld
import os
import torch
import pickle
from cnn import SegmentationModel as net
import numpy as np
__author__ = "Sachin Mehta"


from FlopsCompute import *

input = torch.Tensor(1, 3, 224, 224).cuda()
scale = [0.5, 1.0, 1.5, 2.0]
for s in scale:
    model = net.EESPNet_Seg(20, s=s, pretrained='', gpus=1)

    model = add_flops_counting_methods(model)
    model = model.cuda().eval()
    model.start_flops_count()

    _ = model(input)

    flops = model.compute_average_flops_cost() #+ (model.classifier.in_features * model.classifier.out_features)

    print('Scale: ', s)
    print('Flops: ', flops / 1e6 / 2)
    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Params: ', n_params / 1e6)
