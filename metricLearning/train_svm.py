import pickle

from tqdm import tqdm
from sklearn import svm
from torch.utils.data.dataloader import DataLoader

from cusDataloader import cosineDataLoader
from model.plModel import plModelCosine
from model import models
import numpy as np


net = models.resnet50(dims=256)
ckpt = "saved/plCosine_natural_images/version_0/checkpoints/last.ckpt"
trained_module = plModelCosine.load_from_checkpoint(ckpt,model=net)
trained_module.eval()
dataset = cosineDataLoader("train.txt", "labels.txt")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

x_train = []
y_train = []

for idx, data in enumerate(tqdm(dataloader)):
    img = data['image']
    label = data['label']
    pred = trained_module(img)
    x_train.append(pred.detach().numpy())
    y_train.append(label)

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

print('fitting svm')
svm_cls = svm.SVC()
svm_cls.fit(x_train, y_train)

with open('svm.pkl', 'wb') as f:
    pickle.dump(svm_cls, f)
    print('svm model saved')