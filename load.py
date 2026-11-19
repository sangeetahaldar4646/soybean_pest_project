import torch
from torchvision import models
import torch.nn as nn

# Reconstruct the model structure
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 3)
)

# Load the weights (weights also load directly if named .zip or .pth)
state_dict = torch.load('resnet50_ft.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

print("ResNet50 fine-tuned weights successfully loaded.")