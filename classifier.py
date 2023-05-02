import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

class ResNet18Classifier(pl.LightningModule):
    def __init__(self, num_classes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, self.hparams.num_classes)

    def forward(self, x):
        return self.resnet18(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def model_name (self):
        return "resnet18"

