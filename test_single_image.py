import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 1. Configuration
model_path = 'resnet50_ft.pth'
# Provide the path to the single image you want to test here
image_path = r'D:\MCA Project\Images\healthy_2.jpg' 
class_names = ['Caterpillar', 'Diabrotica Speciosa', 'Healthy']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Reconstruct Model Architecture
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)

# 3. Load Weights
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 4. Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Single Image Classification Function
def classify_single_image(image_path, model, device, class_names):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' does not exist.")
        return

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = F.softmax(output[0], dim=0) * 100
    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = class_names[predicted_idx.item()]

    # Print Result
    print("\n" + "=" * 50)
    print(" SINGLE IMAGE TEST RESULT")
    print("=" * 50)
    print(f" Image Path     : {image_path}")
    print(f" Predicted Class: {predicted_class}")
    print(f" Confidence     : {confidence.item():.2f}%")
    print("-" * 50)
    print(" Probabilities for all classes:")
    for idx, name in enumerate(class_names):
        print(f"   - {name}: {probabilities[idx].item():.2f}%")
    print("=" * 50 + "\n")

# Run inference on the single image
classify_single_image(image_path, model, device, class_names)