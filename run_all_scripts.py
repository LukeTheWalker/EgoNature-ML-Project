# run all scripts
#
# Usage: python run_all_scripts.py
#

import os
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

