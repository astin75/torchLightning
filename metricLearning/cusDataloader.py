from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import pytorch_lightning as pl
import glob
import cv2
import numpy as np
import random

import natsort

class plLoader(pl.LightningDataModule):
    def __init__(self, train_path, label_path,
                 batch_size, workers,dataset, val_path=None,
                 train_transforms=None, val_transforms=None):
        super().__init__()
        self.dataset = dataset
        self.train_path = train_path
        self.label_path = label_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.workers = workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def train_dataloader(self):

        return DataLoader(self.dataset(
            self.train_path, self.label_path),
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            shuffle=True,
            pin_memory=self.workers > 0)

class cosineDataLoader(Dataset):
    def __init__(self, imgPath, labelPath, transform=False):
        super().__init__()
        self.inputShape = (224, 224)
        self.imgPath = imgPath
        if transform:
            self.transforms = transform
        else:
            self.transformsFlag = False
            self.transforms = transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.categories = self.txt_readline(labelPath)
        self.imgList = self.txt_readline(imgPath)
        self.gtDict = self.categorieDictionary()


    def txt_readline(self, path):
        f = open(path,'r')
        gtList = f.readlines()

        return gtList

    def __len__(self):
        return len(self.imgList)

    def categorieDictionary(self):
        gtDict = {}
        for i in self.categories:
            name = i.replace("\n","")
            gtDict[name] = []

        for imgFile in self.imgList:
            imgPath = imgFile.replace("\\", "/").replace("\n", "")
            yLabel = imgPath.split("/")[1]
            gtDict[yLabel].append(imgPath)

        return gtDict



    def __getitem__(self, idx):
        imgFile = self.imgList[idx]
        imgPath = imgFile.replace("\\", "/").replace("\n","")
        yLabel = imgPath.split("/")[1]
        classSTR = imgPath.split("/")[1]
        is_positive = True
        p = random.random()

        true_yLabel = int(self.categories.index(yLabel + "\n"))
        if p >= 0.5:
            gt = int(self.categories.index(yLabel+"\n"))
            temp_yLabel = np.random.choice(len(self.categories), 1)
            while is_positive:
                temp_yLabel = int(temp_yLabel)
                if temp_yLabel != gt:
                    tempClassName = self.categories[temp_yLabel].replace("\n","")
                    ranNumber = int(np.random.choice(len(self.gtDict[tempClassName]), 1))
                    compareImgPath = self.gtDict[tempClassName][ranNumber]

                    is_positive = False
                else:
                    temp_yLabel = np.random.choice(len(self.categories), 1)

        else:
            true_yLabel = int(self.categories.index(yLabel+"\n"))
            temp_yLabel = true_yLabel
            compareImgPath = imgPath
            is_positive = True


        img = cv2.imread(imgPath)
        img = cv2.resize(img, (self.inputShape[0], self.inputShape[1]))

        compareImg = cv2.imread(compareImgPath)
        compareImg = cv2.resize(compareImg, (self.inputShape[0], self.inputShape[1]))

        x = self.transforms(img)
        compare = self.transforms(compareImg)
        y = torch.tensor(true_yLabel)
        temp_yLabel = torch.tensor(int(temp_yLabel))

        return {
            'image': x,
            'label': y,
            'is_positive': is_positive,
            'compare_label': temp_yLabel,
            'compare' : compare
        }

def imshow(npimg, value, predict=False, path = "none"):
    if path != "none":
        f = open(path,'r')
        gtList = f.readlines()

        for n, i in enumerate(npimg):
            img = i.numpy()
            img = np.transpose(img, (1, 2, 0))
            if predict:
                print("GT : {0}, Predict : {1}".format(gtList[value[n]], gtList[predict]))
            else:
                print("GT:",gtList[value[n]])
            cv2.imshow("ee", img)
            cv2.waitKey(0)

if __name__ == "__main__":
    dataset = cosineDataLoader("train.txt", "labels.txt")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    dataiter = iter(dataloader)
    for epoch in range(500):
        for i, data in enumerate(dataloader,0):

            imshow(data['compare'], data['compare_label'], path="labels.txt")

