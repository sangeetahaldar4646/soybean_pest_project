import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np

# 1. Load your local Run 1 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 3)
)
model.load_state_dict(torch.load(r'D:\MCA Project\resnet50_ft.pth', map_location=device))
model.to(device).eval()

# 2. Pure PyTorch Grad-CAM Hook
gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    activations.append(output)

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

class_names = ['Caterpillar', 'Diabrotica Speciosa', 'Healthy']

# 3. Test image path (Apne Images folder ki kisi bhi image ka naam yahan likhein)
img_path = r'D:\MCA Project\Images\diabrotica_2.jpg'
raw_img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = transform(raw_img).unsqueeze(0).to(device)
input_tensor.requires_grad = True

# 4. Forward & Backward pass
outputs = model(input_tensor)
pred_idx = outputs.argmax(dim=1).item()
confidence = F.softmax(outputs[0], dim=0)[pred_idx].item() * 100

model.zero_grad()
outputs[0, pred_idx].backward()

grads = gradients[0].cpu().data.numpy()[0]
acts = activations[0].cpu().data.numpy()[0]
weights = np.mean(grads, axis=(1, 2))

cam = np.zeros(acts.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# 5. Create Jet Heatmap using pure NumPy + PIL (No matplotlib needed!)
cam_resized = np.array(Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)) / 255.0
r = np.clip(1.5 - np.abs(4 * cam_resized - 3), 0, 1)
g = np.clip(1.5 - np.abs(4 * cam_resized - 2), 0, 1)
b = np.clip(1.5 - np.abs(4 * cam_resized - 1), 0, 1)
heatmap = np.stack([r, g, b], axis=-1)

img_np = np.array(raw_img.resize((224, 224))) / 255.0
overlay = np.uint8(np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1) * 255)

# Combine Original + Grad-CAM side-by-side
orig_pil = raw_img.resize((224, 224))
cam_pil = Image.fromarray(overlay)
combined = Image.new('RGB', (468, 260), (255, 255, 255))
combined.paste(orig_pil, (10, 30))
combined.paste(cam_pil, (234, 30))

draw = ImageDraw.Draw(combined)
draw.text((55, 10), "Original UAV Image", fill=(0, 0, 0))
draw.text((240, 10), f"{class_names[pred_idx]} ({confidence:.2f}%)", fill=(200, 0, 0))

output_path = r'D:\MCA Project\gradcam_output.png'
combined.save(output_path)
print(f"Success! Result: {class_names[pred_idx]} ({confidence:.2f}%)")
print(f"Saved at: {output_path}")
combined.show()