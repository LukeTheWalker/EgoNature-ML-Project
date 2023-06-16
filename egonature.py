
import os
from typing import Tuple
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl

def get_classes(modality: str) -> list:
        assert modality in ["Con", "Sub", "ConSub"]
        if modality == "Con":
            return ["Entrance", "Monumental Building", "Greenhouse", "Succulents", "Sicilian Garden", "Leftmost Garden", "Passageway", "Central Garden", "Rightmost Garden"]
        elif modality == "Sub":
            return ["Dune", "Dune-Scrubland", "Marshy-Forestry", "Marshy-Coastal", "Marshy", "Coastal", "Cliffs", "Cliffs-Forestry", "Entrance"]
        elif modality == "ConSub":
            return ["Entrance", "Monumental Building", "Greenhouse", "Succulents", "Leftmost Garden", "Passageway", "Central Garden", "Rightmost Garden", "Dune", "Dune-Scrubland", "Marshy-Forestry", "Marshy-Coastal", "Marshy", "Coastal", "Cliffs", "Cliffs-Forestry", "Entrance"]


def num_classes(modality: str) -> int:
    return len(get_classes(modality))

class EgoNatureDataset(Dataset):
    def __init__(self, stage: str, modality: str, data_dir: str, fold: int, transform=None):
        assert stage in ["train", "test"]
        self.stage = stage
        self.modality = modality
        self.data_dir = data_dir
        self.fold = fold
        self.transform = transform
        self.main_folder = "EgoNature-Dataset/EgoNature-Dataset/EgoNature - CNN"
        self.image_names, self.labels = self.load_image_names_and_labels()

    def load_image_names_and_labels(self) -> Tuple[list, list]:
        assert os.path.exists(self.data_dir)
        
        if self.modality not in ["Con", "Sub", "ConSub"]:
            raise ValueError("modality must be one of [Con, Sub, ConSub]")

        dfs = pd.DataFrame()       
        fname = f"{self.stage}_{self.modality}_{self.fold}.txt"
        file_path = os.path.join(self.data_dir, self.main_folder, fname)
        # assert and print path in case of error
        assert os.path.exists(file_path), f"file_path: {file_path}"
        df = pd.read_csv(file_path, delimiter=",", header=None)
        df.columns = ["image_names", "labels"]
        df["modality"] = self.modality
        dfs = pd.concat([dfs, df], ignore_index=True)

        image_names = dfs["image_names"].tolist()
        labels = dfs["labels"].astype(int).tolist()
        return image_names, labels

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.main_folder, self.image_names[index])
        assert os.path.exists(image_path)
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[index])#, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.image_names)
    
    @property
    def num_classes(self):
        return num_classes(self.modality)

    @property
    def classes(self):
        return get_classes(self.modality)
    
class EgoNatureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, modality: str, data_url: str, num_workers: int, fold: int, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.modality = modality
        self.data_url = data_url
        self.fold = fold
        self.transform = transform
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            download_and_extract_archive(url=self.data_url, download_root=self.data_dir, remove_finished=True)
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = EgoNatureDataset("train",  fold = self.fold, modality=self.modality, data_dir=self.data_dir, transform=self.transform)
            self.val_dataset   = EgoNatureDataset("test" ,  fold = self.fold, modality=self.modality, data_dir=self.data_dir, transform=self.transform)
        else:
            raise ValueError("stage must be [fit]")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def num_classes(self):
        return num_classes(self.modality)