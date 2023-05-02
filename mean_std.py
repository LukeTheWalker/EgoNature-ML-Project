import torch
from torch.utils.data import DataLoader
from settings import data_dir, modality
from egonature import EgoNatureDataset
from torchvision import transforms

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

transform_data = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = EgoNatureDataset("train", modality, data_dir, transform=transform_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

mean, std = get_mean_and_std(dataloader)

print(f"Mean: {mean}")
print(f"Std: {std}")

