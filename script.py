import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from classifier import ResNet18Classifier
from egonature import EgoNatureDataModule
from settings import *

# if not os.path.exists(data_dir):
#     download_and_extract_archive(url=DATA_URL, download_root=data_dir, remove_finished=True)

# train_dataset = EgoNatureDataset("train", modality, data_dir, transform=transform_data)
# val_dataset  = EgoNatureDataset("val" , modality, data_dir, transform=transform_data)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
# val_loader  = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

batch_size = 64 
data_module = EgoNatureDataModule(data_dir=data_dir, batch_size=batch_size, modality=modality, data_url=DATA_URL, num_workers=8, val_fold=2, transform=transform_data)

# Define the model 
num_classes = data_module.num_classes()
model = ResNet18Classifier(num_classes=num_classes, lr=1e-3)

logger = TensorBoardLogger("tb_logs", name=f"nature_{model.model_name()}{modality}", default_hp_metric=False)

accelerator = 'cuda' if torch.cuda.is_available() else 'mps'

max_epochs = 15 if modality != 'Sub' else 30

# Train the model 
checkpoint_callback = ModelCheckpoint(
    dirpath="best_models",
    filename=f"{model.model_name()}_{modality}",
    save_top_k=1,
    monitor="val_acc",
    mode="max"
)

trainer = Trainer(accelerator=accelerator, devices=1, max_epochs=max_epochs, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model=model, datamodule=data_module)
