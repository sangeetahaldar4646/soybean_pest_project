import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# 1. Configuration (Updated for new v1 model)
model_path = r'retrained_model\resnet50_ft_v1 (1).pth'
folder_path = r'D:\MCA Project\Images' # Path to your folder containing testing images
class_names = ['Caterpillar', 'Diabrotica Speciosa', 'Healthy']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Reconstruct Model Architecture
model = models.resnet50()
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)

# 3. Load Weights Safely
if not os.path.exists(model_path):
    print(f"❌ Error: Model file '{model_path}' not found!")
else:
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("✅ New v1 Model loaded successfully for Batch Testing!\n")

# 4. Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Helper function to extract ground truth label from file name
def get_actual_label(filename):
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

    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            continue

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = F.softmax(output[0], dim=0) * 100
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = class_names[predicted_idx.item()]
        
        actual_class = get_actual_label(file_name)

        row_data = {
            'Image Name': file_name,
            'Actual Class': actual_class,
            'Predicted Class': predicted_class,
            'Match': 'Yes' if actual_class == predicted_class else 'No',
            'Confidence (%)': round(confidence.item(), 2)
        }

        for idx, name in enumerate(class_names):
            row_data[f'{name} (%)'] = round(probabilities[idx].item(), 2)

        results.append(row_data)

    if not results:
        print("❌ No valid images processed.")
        return

    # 6. Generate Table
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 90)
    print("CLASSIFICATION SUMMARY TABLE (New v1 Model)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    
    # Save to CSV
    output_csv = "classification_results_v1.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n📁 Saved table successfully to '{output_csv}'")

# Run inference
if __name__ == "__main__":
    classify_folder(folder_path, model, device, class_names)