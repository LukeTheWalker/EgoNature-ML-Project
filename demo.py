import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from classifier import ResNet18Classifier
from settings import transform_data
from egonature import classes

def plot_results (class_labels: list[str], class_probs: list[float], img: str, gt: int):
    # Load image
    img = plt.imread(img)

    # Plot image and class probabilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.axis("off")
    y_pos = np.arange(len(class_labels))
    colors = np.random.rand(len(class_labels), 3)
    ax2.barh(y_pos, class_probs, align="center", color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_labels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.tick_params(axis="y", length=0, labelleft=False, labelright=True, pad=-312)
    ax2.set_xlabel("Probability")
    ax2.set_xlim([0, 100])
    ax2.set_xticks([])
    
    fig.tight_layout()
    fig.savefig("results.png")
    

def find_label (img_name: str, modality: str) -> int:
    dfs = pd.DataFrame()       
    for fold in range(0, 3):
        fname = f"test_{modality}_{fold}.txt"
        file_path = os.path.join("data", "EgoNature-Dataset", "EgoNature-Dataset", "EgoNature - CNN", fname)
        assert os.path.exists(file_path)
        df = pd.read_csv(file_path, delimiter=",", header=None)
        df.columns = ["image_names", "labels"]
        dfs = pd.concat([dfs, df], ignore_index=True)

    return dfs[dfs["image_names"] == img_name]["labels"].values[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Utility Script to use the model trained for the ML project on the EgoNature Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", help="model path", required=True)
    parser.add_argument("-i", "--input", help="input image path", required=True)
    parser.add_argument("-mod", "--modality", help="modality", choices=["Con", "Sub", "ConSub"], required=True)
    args = parser.parse_args()
    config = vars(args)
    # check if all the paths are valid
    assert os.path.exists(config["model"]), "model path does not exist"
    assert os.path.exists(config["input"]), "input image path does not exist"
    return config


def main ():
    config = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transform_data(config["modality"])

    model = ResNet18Classifier.load_from_checkpoint(config["model"])
    model.to(device)
    model.eval()

    image = Image.open(config["input"])
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        preds = model(image)
        preds = nn.functional.softmax(preds, dim=1)

    pred_classes = classes(config["modality"])
    preds = preds.squeeze(0).cpu().numpy()

    label = find_label(os.path.basename(config["input"]), config["modality"])

    for i, pred in enumerate(preds):
        print(f"{pred_classes[i]}: {pred*100:.2f}%", end="")
        if i == label:
            print(" <--- True label", end="")
        print()

    plot_results(pred_classes, preds*100, config["input"], label)

    


if __name__ == "__main__":
    main()