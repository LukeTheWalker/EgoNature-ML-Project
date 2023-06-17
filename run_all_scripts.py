# run all scripts
#
# Usage: python run_all_scripts.py
#

import os
import shutil
import subprocess

# get all the models
models = os.listdir("best_models")

# get all the modalities
modalities = ["Con", "Sub", "ConSub"]

# get all the folds
folds = [0, 1, 2]

# run all the train
for model in models:
    for modality in modalities:
        for fold in folds:
            subprocess.run(["python", "train.py", "-m", "best_models", "-mod", modality, "-f", str(fold)])

# compute all the matrices
for model in models:
    for modality in modalities:
        for fold in folds:
            subprocess.run(["python", "confusion.py", "-m", "best_models", "-mod", modality, "-f", str(fold)])
        subprocess.run(["python", "confusion.py", "-m", "best_models", "-mod", modality])
                       
# compute all the metrics
for model in models:
    for modality in modalities:
        subprocess.run(["python", "metrics.py", "-m", "best_models", "-mod", modality])

# run the 4 demos for each modality

# first demo: py demo.py -m best_models/resnet18_ConSub_fold2.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_02_21_000_0000059.jpeg -mod ConSub

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_ConSub_fold2.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_02_21_000_0000059.jpeg", "-mod", "ConSub"])
shutil.move("results/result.png", "results/results_good1_2018_02_21_000_0000059.png")

# second demo: py demo.py -m best_models/resnet18_Sub_fold1.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_03_01_000_0006167.jpeg -mod Sub

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_Sub_fold1.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_03_01_000_0006167.jpeg", "-mod", "Sub"])
shutil.move("results/result.png", "results/results_good2_2018_03_01_000_0006167.png")

# third demo: py demo.py -m best_models/resnet18_Con_fold1.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_03_01_000_0005496.jpeg -mod Con

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_Con_fold1.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_03_01_000_0005496.jpeg", "-mod", "Con"])
shutil.move("results/result.png", "results/results_bad1_2018_03_01_000_0005496.png")

# fourth demo: py demo.py -m best_models/resnet18_ConSub_fold2.ckpt -i data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_02_21_000_0000011.jpeg -mod ConSub 

subprocess.run(["python", "demo.py", "-m", "best_models/resnet18_ConSub_fold2.ckpt", "-i", "data/EgoNature-Dataset/EgoNature-Dataset/EgoNature\ -\ CNN/2018_02_21_000_0000011.jpeg", "-mod", "ConSub"])
shutil.move("results/result.png", "results/results_bad2_2018_02_21_000_0000011.png")

