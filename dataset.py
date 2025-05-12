from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform if transform else transforms.ToTensor()

        class_names = sorted(set(label for _, label in samples))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.class_to_idx[label_str], dtype=torch.long)
        return image, label
