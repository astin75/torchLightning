import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

class plModel(pl.LightningModule):
    def __init__(self, model, epoch_length=None) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.top_1 = Accuracy(top_k=1)
        self.learning_rate = 0.001

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        #self.enable_running_stats(self.model)
        self.automatic_optimization = False

        #opt = self.optimizers()
        image = batch['image']
        label = batch['label']
        pred = self.model(image)
        loss = self.loss_fn(pred, label)
        # self.manual_backward(loss)
        #
        # opt.step()

        # # 2nd step
        # self.disable_running_stats(self.model)
        # pred = self.model(image)
        # loss = self.loss_fn(pred, label)
        # self.manual_backward(loss)
        # opt.second_step(zero_grad=True)

        # sch = self.lr_schedulers()
        # sch.step()

        self.log('train_loss', loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)
        self.log('train_top1', self.top_1(pred, label),
                 logger=True, on_step=True, on_epoch=True)
        # self.log('train_top5', self.top_5(pred, label),
        #          logger=True, on_step=True, on_epoch=True)

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     train_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     sch = self.lr_schedulers()
    #     sch.step(train_loss)

    def configure_optimizers(self):
        # cfg = self.hparams.cfg
        # epoch_length = self.hparams.epoch_length
        optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # scheduler = CosineAnnealingWarmUpRestarts(
        #     optim,
        #     epoch_length*4,
        #     T_mult=2,
        #     eta_max=cfg['optimizer_options']['lr'],
        #     T_up=epoch_length,
        #     gamma=0.96)

        scheduler = ReduceLROnPlateau(
            optim, factor=0.1, patience=10, verbose=True)
        return {"optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                'monitor': 'train_loss'},
                # 'monitor': 'train_loss',
                # 'interval': 'step'
                }
