import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from egonature import EgoNatureDataset
from classifier import ResNet18Classifier
from settings import data_dir, modality, transform_data
from torch.utils.data import DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test data
batch_size = 64
# self.test_dataset  = EgoNatureDataset("test", folds = [0,1,2], modality=self.modality, data_dir=self.data_dir, transform=self.transform)        
dataset = EgoNatureDataset('test', folds=[0,1,2], modality=modality, data_dir=data_dir, transform=transform_data)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Load the saved model
model = ResNet18Classifier.load_from_checkpoint(f"best_models/resnet18_{modality}.ckpt")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Calculate predictions for the test data
all_preds = []
all_labels = []
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

disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True, xticks_rotation=90)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion matrix')

f.tight_layout()
f.savefig(f'confusion_matrix{modality}.png', dpi=300)

