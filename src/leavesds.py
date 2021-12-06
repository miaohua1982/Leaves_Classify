import os
import torch as t
import pandas as pd
from utils import read_image
import cv2

class LeavesDatasets:
    def __init__(self, image_path, label_path, transformers, ratio=0.8, is_train=True):
        self.train_ds = pd.read_csv(label_path, names=['image','label'], header=1)
        self.labels = self.train_ds.label.unique().tolist()
        self.labelinds = {k:i for i, k in enumerate(self.labels)}
        self.image_path_base = image_path
        self.transformers = transformers
        self.is_train = is_train
        self.ratio = ratio
        self.train_num = int(len(self.train_ds)*ratio)
        if is_train:
            self.size_of_data = self.train_num
        else:
            self.size_of_data = len(self.train_ds) - self.train_num

    def ind2label(self, ind):
        return self.labels[ind]

    def get_classes_num(self):
        return len(self.labels)
    
    def __len__(self):
        return self.size_of_data

    def __getitem__(self, idx):
        if self.is_train:
            one_image = self.train_ds.iloc[idx].tolist()
        else:
            one_image = self.train_ds.iloc[idx+self.train_num].tolist()
        img_path = os.path.join(self.image_path_base, one_image[0])
        if self.transformers:
            img = read_image(img_path, convert_np=False)
            img = self.transformers(img)
        else:
            img = read_image(img_path, convert_np=True)
            img = t.from_numpy(img)
        
        label = self.labelinds[one_image[1]]
        return img, label

class LeavesDatasets_Alb(LeavesDatasets):
    def __init__(self, image_path, label_path, transformers, ratio=0.8, is_train=True):
        super(LeavesDatasets_Alb, self).__init__(image_path, label_path, transformers, ratio, is_train)

    def __getitem__(self, idx):
        if self.is_train:
            one_image = self.train_ds.iloc[idx].tolist()
        else:
            one_image = self.train_ds.iloc[idx+self.train_num].tolist()
        img_path = os.path.join(self.image_path_base, one_image[0])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #  (224, 224, 3) with np.ndarray
        if self.transformers:
            img = self.transformers(image=img)['image']  # [3, 224, 224]
                
        label = self.labelinds[one_image[1]]
        return img, label

class LeavesDatasets2:
    def __init__(self, train_ds, image_path, transformers):
        self.train_ds = train_ds
        self.labels = self.train_ds.label.unique().tolist()
        self.labelinds = {k:i for i, k in enumerate(self.labels)}
        self.image_path_base = image_path
        self.transformers = transformers

    def set_labels(self, uniform_labels):
        self.labels = uniform_labels
        self.labelinds = {k:i for i, k in enumerate(self.labels)}
        
    def ind2label(self, ind):
        return self.labels[ind]

    def get_classes_num(self):
        return len(self.labels)
    
    def __len__(self):
        return self.train_ds.shape[0]

    def __getitem__(self, idx):
        one_image = self.train_ds.iloc[idx].tolist()
        img_path = os.path.join(self.image_path_base, one_image[0])
        if self.transformers:
            img = read_image(img_path, convert_np=False)
            img = self.transformers(img)
        else:
            img = read_image(img_path, convert_np=True)
            img = t.from_numpy(img)
        
        label = self.labelinds[one_image[1]]
        return img, label

class LeavesDatasets_Alb2(LeavesDatasets2):
    def __init__(self, train_ds, image_path, transformers):
        super(LeavesDatasets_Alb2, self).__init__(train_ds, image_path, transformers)

    def __getitem__(self, idx):
        one_image = self.train_ds.iloc[idx].tolist()
        img_path = os.path.join(self.image_path_base, one_image[0])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #  (224, 224, 3) with np.ndarray
        if self.transformers:
            img = self.transformers(image=img)['image']  # [3, 224, 224]
                
        label = self.labelinds[one_image[1]]
        return img, label

def get_n_splits(label_path, flod_splits):
    ds = pd.read_csv(label_path, names=['image','label'], header=1)
    train_idxs = []
    test_idxs = []
    for train_idx, test_idx in flod_splits.split(ds.image, ds.label):
         train_idxs.append(train_idx)
         test_idxs.append(test_idx)
    
    return train_idxs, test_idxs, ds