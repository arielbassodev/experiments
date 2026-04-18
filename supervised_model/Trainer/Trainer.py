from torch.utils.data import DataLoader

from Data_loading import *
from Data_loading.Data_loading import DataLoading
from Encoder import *
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from Encoder.Encoder import VitModel
import torch
import torch.nn as nn
import Scheduler

train_data = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/Dataset/Lsfb_dataset/isol",
    split="train",
    n_labels=500,
    sequence_max_length=50
))

test_data = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/Dataset/Lsfb_dataset/isol",
    split="test",
    n_labels=500,
    sequence_max_length=50
))

train_loader = DataLoading(train_data, 512)()
test_loader  = DataLoading(test_data , 512)()
model = VitModel(80, 150, 4, 2).to(device)

from torch.optim.lr_scheduler import MultiStepLR
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

import torch.optim as optim
import pytorch_lightning as L
import torchmetrics as TM


class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 500
        self.train_bal_acc = TM.Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.val_bal_acc = TM.Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        # self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        sign, target, mask = batch
        sign = sign.to(device).to(torch.float32).to(device)
        target = target.type(torch.LongTensor)
        target = target.to(device)
        mask = mask.to(device).to(torch.float32).to(device)
        logits = self.model(sign, mask)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, target)
        self.train_bal_acc(preds, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_bal_acc', self.train_bal_acc, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sign, target, mask = batch
        sign = sign.to(torch.float32).to(device)
        target = target.type(torch.LongTensor)
        target = target.to(device)
        mask = mask.to(torch.float32).to(device)
        logits = self.model(sign, mask)
        loss = self.criterion(logits, target)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, target)
        self.val_bal_acc(preds, target)
        # self.val_recall(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log("val_bal_acc", self.val_bal_acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Créez l'optimiseur
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

my_module = Module()
import pytorch_lightning as L
trainer = L.Trainer(max_epochs=1000)
trainer.fit(
    my_module,
    train_loader,
    test_loader,
)
