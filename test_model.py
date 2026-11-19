import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# 1. Configuration
model_path = 'resnet50_ft.pth'
folder_path = r'D:\MCA Project\Images' # Path to your folder containing testing images
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

# Helper function to extract ground truth label from file name
def get_actual_label(filename):
    # Extracts text before underscores or digits (e.g. catepiller_1.jpg -> catepiller)
    base_name = os.path.splitext(filename)[0]
    match = re.match(r"([a-zA-Z]+)", base_name)
    if match:
        extracted = match.group(1).lower()
        if "catepiller" in extracted or "caterpillar" in extracted:
            return "Caterpillar"
        elif "diabrotica" in extracted:
            return "Diabrotica Speciosa"
        elif "healthy" in extracted:
            return "Healthy"
    return "Unknown"

# 5. Generic Classification Function
def classify_folder(folder_path, model, device, class_names):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    results = []

    # Loop through any file in the folder
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        
        # Process file if it is an image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            # Skip non-image files safely without failing
            continue

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = F.softmax(output[0], dim=0) * 100
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = class_names[predicted_idx.item()]
        
        # Get actual class from filename for the summary table
        actual_class = get_actual_label(file_name)

        row_data = {
            'Image Name': file_name,
            'Actual Class': actual_class,
            'Predicted Class': predicted_class,
            'Match': 'Yes' if actual_class == predicted_class else 'No',
            'Confidence (%)': round(confidence.item(), 2)
        }

        # Add individual class probabilities
        for idx, name in enumerate(class_names):
            row_data[f'{name} (%)'] = round(probabilities[idx].item(), 2)

        results.append(row_data)

    # 6. Generate Table
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv("classification_results.csv", index=False)
    print("\nSaved table to 'classification_results.csv'")

# Run inference
classify_folder(folder_path, model, device, class_names)