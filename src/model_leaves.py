import torchvision as tv
import torch as t
from torch import nn
import timm


def model_leaves_py(classes_num):
    res50 = tv.models.resnet50(pretrained=True)
    for one in res50.parameters():
        one.requires_grad = False
    # try not to fix layer3
    # for param in res50.layer3.parameters():
    #     param.requires_grad = True
    # try not to fix layer4
    for param in res50.layer4.parameters():
        param.requires_grad = True
    # res50.fc = mlp
    res50.fc = nn.Linear(res50.fc.in_features, classes_num)
    return res50

def model_leaves_timm(classes_num):
    res50 = timm.create_model('resnet50d', pretrained=True)

    #for one in res50.parameters():
    #    one.requires_grad = False
    #for param in res50.layer4.parameters():
    #    param.requires_grad = True
    
    res50.fc = nn.Linear(res50.fc.in_features, classes_num)
    nn.init.xavier_uniform_(res50.fc.weight)
    return res50