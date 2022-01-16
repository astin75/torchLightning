import glob
import natsort
import random

src = natsort.natsorted(glob.glob("natural_images/**/*.jpg"))
classes = natsort.natsorted(glob.glob("natural_images/*"))
trainList = []
testList = []
labelList = []
for i in classes:
    imgList = natsort.natsorted(glob.glob(i+"/*.jpg"))
    random.shuffle(imgList)
    size = len(imgList)
    trainSize = size * 0.8
    for n, j in enumerate(imgList):
        img = j.replace("\\","/")
        label = img.split("/")[1]

        if trainSize > n:
            trainList.append(img)
        else:
            testList.append(img)

        if n ==0:
            labelList.append(label)

with open("train.txt", "w") as txt:
    for n in trainList:
        txt.writelines(str(n)+"\n")

with open("test.txt", "w") as txt:
    for n in testList:
        txt.writelines(str(n)+"\n")

with open("labels.txt", "w") as txt:
    for n in labelList:
        txt.writelines(str(n)+"\n")

print("done")