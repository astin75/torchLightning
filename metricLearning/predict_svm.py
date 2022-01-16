import pickle

from tqdm import tqdm
from sklearn import svm
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import accuracy_score
import cv2
from cusDataloader import cosineDataLoader
from model.plModel import plModelCosine
from model import models
import numpy as np

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


net = models.resnet50(dims=256)
ckpt = "saved/plCosine_natural_images/version_0/checkpoints/last.ckpt"
trained_module = plModelCosine.load_from_checkpoint(ckpt,model=net)
trained_module.eval()
dataset = cosineDataLoader("test.txt", "labels.txt")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

predictions = []
y_train = []


with open('svm.pkl', 'rb') as f:
    svm_cls = pickle.load(f)

for idx, data in enumerate(tqdm(dataloader)):
    img = data['image']
    label = data['label']
    pred = trained_module(img)
    prediction = svm_cls.predict(pred.detach().numpy())
    predictions.append(prediction)
    y_train.append(label)
    #imshow(data['image'], data['label'], path="labels.txt",predict=prediction )

print(accuracy_score(predictions, y_train))