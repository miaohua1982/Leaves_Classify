import torch as t
from torchvision import transforms
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
import torchvision as tv
import os
import time
from leavesds import LeavesDatasets, LeavesDatasets_Alb
from model_leaves import model_leaves_py, model_leaves_timm
import torch_utils as tu

import albumentations
from albumentations import pytorch as AT
import cv2

label_path = "classify-leaves/train.csv"
img_base_path = "classify-leaves"

'''
1、使用softlabel
2、优化器使用AdamW
3、使用cosine lr scheduler
4、使用mixup
5、使用timm的新baseline resnet50d
6、model的fc部分，和前面部分使用不同learning rate
7、使用flod(n_flod=5), model ensemble
8、使用tta(test time argument)
'''

def get_train_validate_dataloader(img_base_path, label_path, batch_size=64, alb=False):
    train_transform = transforms.Compose(
        [transforms.Resize(288),
        transforms.RandomRotation(90),
        #transforms.RandomResizedCrop(224,scale=(0.3,1.0)),
        transforms.RandomAffine(degrees=45, shear=(10, 20, 10, 20), scale=(0.75, 1.2), translate=(0.1, 0.1)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    valid_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    alb_train_transform = albumentations.Compose([
        albumentations.Resize(224, 224, interpolation=cv2.INTER_AREA),
        albumentations.RandomRotate90(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        #tu.randAugment(N=2,M=6,p=1,cut_out=True),    
        albumentations.Normalize(),
        AT.ToTensorV2(),    # change to torch.tensor, and permute CHW to HWC
        ])
    
    alb_valid_transform = albumentations.Compose([
        albumentations.Normalize(),
        AT.ToTensorV2(),    # change to torch.tensor, and permute CHW to HWC
        ])

    if alb:
        trainset = LeavesDatasets_Alb(img_base_path, label_path, alb_train_transform, ratio=0.8, is_train=True)
        validateset = LeavesDatasets_Alb(img_base_path, label_path, alb_valid_transform, ratio=0.8, is_train=False)
    else:
        trainset = LeavesDatasets(img_base_path, label_path, train_transform, ratio=0.8, is_train=True)
        validateset = LeavesDatasets(img_base_path, label_path, valid_transform, ratio=0.8, is_train=False)

    trainloader = t.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    validateloader = t.utils.data.DataLoader(validateset, batch_size=batch_size, shuffle=False, drop_last=False)

    return trainloader, validateloader, trainset, validateset

def accu(pred, target, mixup=False):
    p = pred.argmax(dim=1)
    y = target.argmax(dim=1) if mixup else target
    return (p == y).float().mean()

def model_validate(net, criterion, validateloader):
    print('[%s] we ara validating model on validate dataset...' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    net.eval()
    running_loss = 0.0
    running_acc = 0.0
    counter = 0
    for data in validateloader:
        inputs, labels = data  # labels: [batch_size, 1]

        outputs = net(inputs)  # outputs: [batch_size, ]
        # loss
        loss = criterion(outputs, labels)
        # acc
        acc = (outputs.argmax(dim=1) == labels).float().sum()
        counter += outputs.shape[0]
        # loss & acc
        running_loss += loss.item()
        running_acc += acc.item()
    net.train()
    return  running_loss/len(validateloader), running_acc/counter

def model_train(net, trainloader, criterion, optimizer, scheduler, epoch, spot_step, mixup_fn=None, MIXUP=False):
    running_loss = 0.0
    running_acc = 0.0
    total_loss = 0.0
    total_acc = 0.0
    counter = 0
    net.train()

    for i, data in enumerate(trainloader):
        inputs, labels = data  # labels: [batch_size, 1]
        
        # do mixup
        if mixup_fn:
            x, y = mixup_fn(inputs, labels)
        else:
            x, y = inputs, labels
        # clear previous grad
        optimizer.zero_grad()

        outputs = net(x)  # outputs: [batch_size, 176]
        # print(outputs.shape)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        acc = accu(outputs, y, mixup=MIXUP)
        # loss & acc
        running_loss += loss.item()
        running_acc += acc.item()

        total_loss += loss.item()
        total_acc += (outputs.argmax(dim=1) == labels).float().sum()
        counter += outputs.shape[0]

        if (i+1) % spot_step == 0:  # print loss every 20 mini batch
            print('[%s] [%d, %5d] loss: %.5f, accu: %.5f' %
                (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch+1, i+1, running_loss/spot_step, running_acc/spot_step))
            running_loss = 0.0
            running_acc = 0.0

        scheduler.step()

    return total_loss/len(trainloader), total_acc/counter
    

def train(classes_num, spot_step, model_path):
    MIXUP = 0.1
    batch_size = 64
    epochs = 50 
    learning_rate = 0.003  # learning rate

    print('[%s] We start to train model......' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    if model_path is None:
        # net = model_leaves_py(classes_num)
        net = model_leaves_timm(classes_num)
    else:
        net = t.load(model_path)

    trainloader, validateloader, trainset, validateset = get_train_validate_dataloader(img_base_path, label_path, batch_size, True)

    train_criterion = tu.SoftTargetCrossEntropy()
    test_criterion = nn.CrossEntropyLoss()
    params_01x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
    optimizer = optim.AdamW([{"params":params_01x, 'lr':learning_rate*0.1}, {"params":net.fc.parameters()}], 
                            lr=learning_rate, weight_decay=2e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(trainloader), eta_min=learning_rate/20)
    mixup_fn = tu.Mixup(prob=MIXUP, switch_prob=0.0, onehot=True, label_smoothing=0.05, num_classes=trainset.get_classes_num())
    
    for epoch in range(epochs):
        # train
        one_epoch_avgloss, one_epoch_acc = model_train(net, trainloader, train_criterion, optimizer, scheduler, epoch, spot_step, mixup_fn, MIXUP)
        print('[%s] [%d, %5d] loss: %.5f, accu: %.5f' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), epoch+1, epochs, one_epoch_avgloss, one_epoch_acc))
        # validate
        val_avgloss, val_acc = model_validate(net, test_criterion, validateloader)
        print('[%s] In validate set loss: %.5f, accu: %.5f' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), val_avgloss, val_acc))

        t.save(net.state_dict(), 'model_storage/model_leaf_%s_%d_%.6f.pkl' % (time.strftime('%Y-%m-%d',time.localtime(time.time())), epoch+1, val_acc))
        print('current lr is changing to', optimizer.param_groups[0]['lr'])

    print('[%s] Finished Training' % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))


if __name__ == '__main__':
    classes_num = 176
    spot_step = 20
    model_path = None
    train(classes_num, spot_step, model_path)
