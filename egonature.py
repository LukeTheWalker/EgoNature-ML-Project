
import os
from typing import Tuple
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl

class EgoNatureDataset(Dataset):
    def __init__(self, mode: str, modality: str, data_dir: str, transform=None):
        assert mode in ["train", "val", "test"]
        self.mode = 'train' if mode == 'train' else 'test'
        self.data_dir = data_dir
        self.transform = transform
        self.modality = modality
        self.main_folder = "EgoNature-Dataset/EgoNature-Dataset/EgoNature - CNN"
        self.image_names, self.labels = self.load_image_names_and_labels()

    def load_image_names_and_labels(self) -> Tuple[list, list]:
        assert os.path.exists(self.data_dir)
        
        if self.modality not in ["Con", "Sub", "ConSub"]:
            raise ValueError("modality must be one of [Con, Sub, ConSub]")

        dfs = pd.DataFrame()       
        for fold in range(3):
            fname = f"{self.mode}_{self.modality}_{fold}.txt"
            file_path = os.path.join(self.data_dir, self.main_folder, fname)
            assert os.path.exists(file_path)
            df = pd.read_csv(file_path, delimiter=",", header=None)
            df.columns = ["image_names", "labels"]
            df["modality"] = self.modality
            df["fold"] = fold
            dfs = pd.concat([dfs, df], ignore_index=True)

        image_names = dfs["image_names"].tolist()
        labels = dfs["labels"].astype(int).tolist()
        return image_names, labels

    @property
    def classes(self):
        if self.modality == "Con":
            return ["Entrance", "Monumental Building", "Greenhouse", "Succulents", "Sicilian Garden", "Leftmost Garden", "Passageway", "Central Garden", "Rightmost Garden"]
        elif self.modality == "Sub":
            return ["Dune", "Dune-Scrubland", "Marshy-Forestry", "Marshy-Coastal", "Marshy", "Coastal", "Cliffs", "Cliffs-Forestry", "Entrance"]
        elif self.modality == "ConSub":
            return ["Entrance", "Monumental Building", "Greenhouse", "Succulents", "Leftmost Garden", "Passageway", "Central Garden", "Rightmost Garden", "Dune", "Dune-Scrubland", "Marshy-Forestry", "Marshy-Coastal", "Marshy", "Coastal", "Cliffs", "Cliffs-Forestry", "Entrance"]

    @property
    def num_classes(self):
        return len(self.classes)

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
    
class EgoNatureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, modality: str, data_url: str, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.modality = modality
        self.data_url = data_url
        self.transform = transform
        self.data_url = data_url

    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            download_and_extract_archive(url=self.data_url, download_root=self.data_dir, remove_finished=True)
    
    def setup(self, stage=None):
        self.train_dataset = EgoNatureDataset("train", self.modality, self.data_dir, transform=self.transform)
        self.test_dataset  = EgoNatureDataset("val" , self.modality, self.data_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=8
        )
