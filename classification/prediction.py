import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from cusDataloader import classificationDataLoader
from cusDataloader import imshow

from model.plModel import plModel
from model import models


net = models.resnet50(dims=8)


fld = "version_18"
path = "./saved/plClassifi_natural_images/{0}/checkpoints/last.ckpt".format(fld)
model = plModel.load_from_checkpoint(path, model=net)
model.eval()
dataset = classificationDataLoader("test.txt", "labels.txt")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
dataiter = iter(dataloader)
showFlag = True
for i, data in enumerate(dataloader, 0):
    image = data['image']

    predict = model(image)
    predict = torch.argmax(predict).cpu().numpy()
    label = data['label'].cpu().numpy()
    if showFlag:
        imshow(data['image'], data['label'], predict=predict,path="labels.txt")
    if label != predict:
        print(label, predict)
