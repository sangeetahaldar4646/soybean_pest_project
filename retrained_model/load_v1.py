import os
import torch
from torchvision import models
import torch.nn as nn

# 1. Automatically detect the folder where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'resnet50_ft_v1.pth')

# 2. Reconstruct the model structure
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 3)
)

# 3. Load the weights using the correct path
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

print("✅ ResNet50 v1 fine-tuned weights successfully loaded.")