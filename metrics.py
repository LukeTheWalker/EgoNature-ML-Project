import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from classifier import ResNet18Classifier
from settings import *
from egonature import EgoNatureDataModule, EgoNatureDataset, get_classes

def parse_args():
    parser = argparse.ArgumentParser(description="Utility Script to use the model trained for the ML project on the EgoNature Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", help="model folder path", required=True)
    parser.add_argument("-mod", "--modality", help="modality", choices=["Con", "Sub", "ConSub"], required=True)
    args = parser.parse_args()
    config = vars(args)
    # check if all the paths are valid
    assert os.path.exists(config["model"]), "model path does not exist"
    return config


def main ():
    config = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transform_data(config["modality"])
    classes = get_classes(config["modality"])

    all_distribs = np.array([])

    # if the distributions have already been calculated, load them from file
    if os.path.exists(os.path.join("bkp", f"all_distribs_{config['modality']}.npy")):
        all_distribs = np.load(os.path.join("bkp", f"all_distribs_{config['modality']}.npy"))
    else: 
        for fold in range(0, 3):    
            model = ResNet18Classifier.load_from_checkpoint(os.path.join(config["model"], f"resnet18_{config['modality']}_fold{fold}.ckpt"))
            model.to(device)
            model.eval()

            dataset = EgoNatureDataset('test', folds=[0,1,2], modality=config["modality"], data_dir=data_dir, transform=transform)
            test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

            model_predictions = np.array([])

            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc='Predicting labels'):
                    images = images.to(device)
                    preds = model(images)
                    preds = nn.functional.softmax(preds, dim=1)
                    model_predictions = np.vstack((model_predictions, preds.cpu().numpy())) if model_predictions.size else preds.cpu().numpy()

            all_distribs = np.dstack(   (all_distribs, model_predictions)) if all_distribs.size else model_predictions
        np.save(os.path.join("bkp", f"all_distribs_{config['modality']}.npy"), all_distribs)

    # get all labels from the dataset
    all_labels = np.array([])
    if os.path.exists(os.path.join("bkp", f"all_labels_{config['modality']}.npy")):
        all_labels = np.load(os.path.join("bkp", f"all_labels_{config['modality']}.npy"))
    else:
        dataset = EgoNatureDataset('test', folds=[0,1,2], modality=config["modality"], data_dir=data_dir, transform=transform)
        test_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8)
        for images, labels in tqdm(test_loader, desc='Getting labels'):
            all_labels = np.append(all_labels, labels.numpy())

        # save all distributions and labels to file
        np.save(os.path.join("bkp", f"all_labels_{config['modality']}.npy"), all_labels)

    print(all_distribs.shape)
    print(all_labels.shape)

    # i want to create a dataframe with columns:
    # - fold1 / precision
    # - fold1 / recall
    # - fold1 / f1
    # - fold2 / precision
    # - fold2 / recall
    # - fold2 / f1
    # - fold3 / precision
    # - fold3 / recall
    # - fold3 / f1
    # - mean / precision
    # - mean / recall
    # - mean / f1

    # and each row is a class

    results = pd.DataFrame(columns=["fold 0 / precision", "fold 0 / recall", "fold 0 / f1", "fold 1 / precision", "fold 1 / recall", "fold 1 / f1", "fold 2 / precision", "fold 2 / recall", "fold 2 / f1", "combined / precision", "combined / recall", "combined / f1", "mean / precision", "mean / recall", "mean / f1"], index=classes)
    accuracies = np.array([])

    # for each fold, calculate the accuracy
    for i in range(0, all_distribs.shape[2]):
        preds = np.argmax(all_distribs[:,:,i], axis=1)
        accuracy = np.mean(preds == all_labels)
        accuracies = np.append(accuracies, accuracy)
        print(f"Fold {i} Accuracy: {accuracy*100:.2f}%")
        # for each class, calculate the precision and recall and F1 score
        for c in range(0, len(classes)):
            tp = np.sum(preds[all_labels == c] == c)
            fp = np.sum(preds[all_labels != c] == c)
            fn = np.sum(preds[all_labels == c] != c)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            # print(f"{classes[c]}: precision {precision*100:.2f}%, recall {recall*100:.2f}%, F1 {f1*100:.2f}%")
            results.loc[classes[c], f"fold {i} / precision"] = precision
            results.loc[classes[c], f"fold {i} / recall"] = recall
            results.loc[classes[c], f"fold {i} / f1"] = f1
        
    
    mean_predictions = np.mean(all_distribs, axis=2)
    preds = np.argmax(mean_predictions, axis=1)
    accuracy = np.mean(preds == all_labels)
    print(f"combined Accuracy: {accuracy*100:.2f}%")
    # for each class, calculate the precision and recall and F1 score
    for c in range(0, len(classes)):
        tp = np.sum(preds[all_labels == c] == c)
        fp = np.sum(preds[all_labels != c] == c)
        fn = np.sum(preds[all_labels == c] != c)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        # print(f"{classes[c]}: precision {precision*100:.2f}%, recall {recall*100:.2f}%, F1 {f1*100:.2f}%")
        results.loc[classes[c], f"combined / precision"] = precision
        results.loc[classes[c], f"combined / recall"] = recall
        results.loc[classes[c], f"combined / f1"] = f1

    results["mean / precision"] = results[["fold 0 / precision", "fold 1 / precision", "fold 2 / precision"]].mean(axis=1)
    results["mean / recall"] = results[["fold 0 / recall", "fold 1 / recall", "fold 2 / recall"]].mean(axis=1)
    results["mean / f1"] = results[["fold 0 / f1", "fold 1 / f1", "fold 2 / f1"]].mean(axis=1)

    # calculate the mean and standard deviation of the accuracy
    print(f"Mean accuracy: {np.mean(accuracies)*100:.5f}%")

    print(results)
    # save using two decimal places
    results = results.astype(float)
    # results = np.floor(results * 1000) / 1000
    # results[["mean / precision", "mean / recall", "mean / f1"]].to_csv(os.path.join("results", f"results_{config['modality']}.csv"), float_format="%.3f")
    results.to_csv(os.path.join("results", f"results_{config['modality']}.csv"), float_format="%.3f")
        

if __name__ == "__main__":
    main()