import pytorch_lightning as pl
import torch
from cusDataloader import classificationDataLoader
from cusDataloader import plLoader
from model.plModel import plModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import platform

from model import models
from yamlHelper import make_model_name, load_yaml_file


cfg = load_yaml_file("test.yml")


net = models.resnet50(dims=8)
data_module = plLoader("train.txt","labels.txt", batch_size=32, workers=4, dataset=classificationDataLoader)
train_module = plModel(net, cfg)

callbacks = [
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(monitor='val_top1', save_last=True,
                    every_n_epochs=cfg['save_freq'])
]

trainer = pl.Trainer(
    max_epochs=cfg['epochs'],
    logger=TensorBoardLogger(cfg['save_dir'],
                             make_model_name(cfg)),
    gpus=cfg['gpus'],
    accelerator='ddp' if platform.system() != 'Windows' else None,
    plugins=DDPPlugin(
        find_unused_parameters=False) if platform.system() != 'Windows' else None,
    callbacks=callbacks,
    **cfg['trainer_options'])
if __name__ == '__main__':
    trainer.fit(train_module, data_module)
