import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

resnet = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)

print(resnet)

feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])

batch_data = torch.randn(64, 7, 224, 224)

output = feature_extractor(batch_data)

print(output.shape)

