import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

class plModel(pl.LightningModule):
    def __init__(self, model, cfg=None, epoch_length=None) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.top_1 = Accuracy(top_k=1)
        self.learning_rate = 0.001

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        image = batch['image']
        label = batch['label']
        pred = self.model(image)
        loss = self.loss_fn(pred, label)

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_top1', self.top_1(pred, label),
                 logger=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


        scheduler = ReduceLROnPlateau(
            optim, factor=0.1, patience=10, verbose=True)
        return {"optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                'monitor': 'train_loss'},
                }
