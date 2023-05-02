from torchvision import transforms

mean = {
    'Con': [0.4333, 0.4387, 0.4212],
    'Sub': [0.4607, 0.4685, 0.4494],
    'ConSub': [0.4333, 0.4387, 0.4212]
    }

std  = {
    'Con': [0.2328, 0.2349, 0.2380],
    'Sub': [0.2123, 0.2104, 0.2143],
    'ConSub': [0.2328, 0.2349, 0.2380]
    }

data_dir = 'data' 
modality = "Con" #["Con", "Sub", "ConSub"]:
DATA_URL = "https://iplab.dmi.unict.it/EgoNature/EgoNature-Dataset.zip"
transform_data = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean[modality], std=std[modality])
])