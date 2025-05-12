from torchvision.models import resnet50
import torch.nn as nn

class ResNetBackbone(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super().__init__()
        base = resnet50(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # без fc
        self.embedding = nn.Linear(base.fc.in_features, embedding_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
