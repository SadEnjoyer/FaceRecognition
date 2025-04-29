import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from config import ALIGNED_DIR

class FaceDataset(Dataset):
    def __init__(self, root_dir=ALIGNED_DIR, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = []

        for class_id in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_id)
            if not os.path.isdir(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.samples.append((img_path, int(class_id)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
