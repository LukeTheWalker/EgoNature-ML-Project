import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from egonature import EgoNatureDataset
from classifier import ResNet18Classifier
from settings import data_dir, transform_data
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Utility Script to use the model trained for the ML project on the EgoNature Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_path", help="models path", required=True)
    parser.add_argument("-mod", "--modality", help="modality", choices=["Con", "Sub", "ConSub"], required=True)
    parser.add_argument("-f", "--fold", help="fold", choices=['0','1','2'], default=['0','1','2'], nargs="+")
    args = parser.parse_args()
    config = vars(args)
    # check if all the paths are valid
    assert os.path.exists(config["model_path"]), "model path does not exist"
    return config


def main():
    if not os.path.exists("confusion_matrices"):
        os.mkdir("confusion_matrices")

    config = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transform_data(config["modality"])

    # Load the test data
    batch_size = 64
    # self.test_dataset  = EgoNatureDataset("test", folds = [0,1,2], modality=self.modality, data_dir=self.data_dir, transform=self.transform)        

    all_preds = []
    all_labels = []
    # Load the saved model
    for fold in config["fold"]:
        dataset = EgoNatureDataset('test', fold=fold, modality=config["modality"], data_dir=data_dir, transform=transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        # model = ResNet18Classifier.load_from_checkpoint(re.sub(r'fold\d', f'fold{fold}', config["model"]))
        model = ResNet18Classifier.load_from_checkpoint(os.path.join(config["model_path"], "resnet18_" + config["modality"] + "_fold" + fold + ".ckpt"))
        model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Calculate predictions for the test data
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Predicting labels'):
                images = images.to(device)
                preds = model(images)
                preds = nn.functional.softmax(preds, dim=1)
                preds = torch.argmax(preds, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(all_labels, all_preds)

    # Display the confusion matrix
    f,ax = plt.subplots(1,1,figsize=(15,15))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, 
        display_labels=dataset.classes
    )

    # calculate the accuracy
    accuracy = np.trace(conf_mat) / np.sum(conf_mat)
    print(f"Accuracy: {accuracy}")

    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, xticks_rotation=90)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')

    f.tight_layout()
    f.savefig(f'confusion_matrices/confusion_matrix{config["modality"]}_fold{"".join(config["fold"])}.png', dpi=300)

if __name__ == "__main__":
    main()