import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Utility Script to use the model trained for the ML project on the EgoNature Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--models_folder", help="best model folder path", required=True)
    parser.add_argument("-mod", "--modality", help="modality", choices=["Con", "Sub", "ConSub"], required=True)
    parser.add_argument("-f", "--fold", help="fold", choices=['0','1','2'], required=True)
    args = parser.parse_args()
    config = vars(args)
    # check if all the paths are valid
    assert os.path.exists(config["models_folder"]), "model path does not exist"
    return config

def main ():
    config = parse_args()

    # grab first command line argument
    fold = int(config["fold"])
    modality = config["modality"]
    model_dirpath = config["models_folder"]
    transform = transform_data(modality)

    print(f"Validation fold {fold}")

    batch_size = 64 
    data_module = EgoNatureDataModule(data_dir=data_dir, batch_size=batch_size, modality=modality, data_url=DATA_URL, num_workers=8, val_fold=2, transform=transform)

    # Define the model 
    num_classes = data_module.num_classes()
    model = ResNet18Classifier(num_classes=num_classes, lr=1e-3)

    logger = TensorBoardLogger("tb_logs", name=f"nature_{model.model_name()}{modality}_fold{fold}", default_hp_metric=False)

    accelerator = 'cuda' if torch.cuda.is_available() else 'mps'

    max_epochs = 15 if modality != 'Sub' else 30

    # Train the model 
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dirpath,
        filename=f"{model.model_name()}_{modality}_fold{fold}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    trainer = Trainer(accelerator=accelerator, devices=1, max_epochs=max_epochs, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()