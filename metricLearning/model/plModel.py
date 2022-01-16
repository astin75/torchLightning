import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from plClassification.torchLightning.metricLearning.module import cosineModule


class plModelCosine(pl.LightningModule):
    def __init__(self, model, num_classes=5, cfg=None, epoch_length=None) -> None:
        super().__init__()
        self.model = model
        self.cosineModule = cosineModule.MarginCosineProudct(256, num_classes)
        self.save_hyperparameters(ignore='model')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.top_1 = Accuracy(top_k=1)
        self.learning_rate = 0.001

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        image = batch['image']
        label = batch['label']

        feature = self.model(image)
        pred = self.cosineModule([feature, label])
        loss = self.loss_fn(pred, label)

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_top1', self.top_1(pred, label),
                 logger=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):

        optim = torch.optim.Adam([{'params': self.model.parameters()},
                                  {'params':self.cosineModule.parameters()}],
                                 lr=self.learning_rate)


        scheduler = ReduceLROnPlateau(
            optim, factor=0.1, patience=10, verbose=True)
        return {"optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                'monitor': 'train_loss'},
                }
