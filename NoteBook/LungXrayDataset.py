from torch.utils.data import Dataset
from PIL import Image
import torch

class LungXrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[index])  # Assuming labels is a list or array
        
        return image, label