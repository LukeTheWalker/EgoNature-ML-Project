# run all scripts
#
# Usage: python run_all_scripts.py
#

import os
import shutil
import subprocess

# get all the modalities
modalities = ["Con", "Sub", "ConSub"]

# get all the folds
folds = [0, 1, 2]

# run all the train

for modality in modalities:
    for fold in folds:
        # check if model is already trained
        if not os.path.exists(f"best_models/resnet18_{modality}_fold{fold}.ckpt"):
            print(f"Training {modality} fold {fold}")
            subprocess.run(["python", "train.py", "-m", "best_models", "-mod", modality, "-f", str(fold)])

# compute all the matrices
for modality in modalities:
    for fold in folds:
        if not os.path.exists(f"confusion_matrices/confusion_matrix{modality}_fold{fold}.png"):
            subprocess.run(["python", "confusion.py", "-m", "best_models", "-mod", modality, "-f", str(fold)])
    if not os.path.exists(f"confusion_matrices/confusion_matrix{modality}_fold012.png"):
        subprocess.run(["python", "confusion.py", "-m", "best_models", "-mod", modality])
                    
# compute all the metrics

for modality in modalities:
    if not os.path.exists(f"results/results_{modality}.csv"):
        subprocess.run(["python", "metrics.py", "-m", "best_models", "-mod", modality])

if not os.path.exists("results"):
    os.mkdir("results")

# run the 4 demos for each modality

# first demo: py demo.py -m best_models/resnet18_ConSub_fold2.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_02_21_000_0000059.jpeg -mod ConSub

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_ConSub_fold2.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature - CNN/2018_02_21_000_0000059.jpeg", "-mod", "ConSub"])
shutil.move("results/results.png", "results/results_good1_2018_02_21_000_0000059.png")

# second demo: py demo.py -m best_models/resnet18_Sub_fold1.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_03_01_000_0006167.jpeg -mod Sub

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_Sub_fold1.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature - CNN/2018_03_01_000_0006167.jpeg", "-mod", "Sub"])
shutil.move("results/results.png", "results/results_good2_2018_03_01_000_0006167.png")

# third demo: py demo.py -m best_models/resnet18_Con_fold1.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_03_01_000_0005496.jpeg -mod Con

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_Con_fold1.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature - CNN/2018_03_01_000_0005496.jpeg", "-mod", "Con"])
shutil.move("results/results.png", "results/results_bad1_2018_03_01_000_0005496.png")

# fourth demo: py demo.py -m best_models/resnet18_ConSub_fold2.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_02_21_000_0000011.jpeg -mod ConSub 

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_ConSub_fold2.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature - CNN/2018_02_21_000_0000011.jpeg", "-mod", "ConSub"])
shutil.move("results/results.png", "results/results_bad2_2018_02_21_000_0000011.png")

