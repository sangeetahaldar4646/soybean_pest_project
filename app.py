import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Soybean Pest Detection with Grad-CAM", layout="centered")

st.title("🌱 Soybean Pest & Disease Detection")
st.write("Upload a UAV leaf image to classify it and visualize the Grad-CAM explanation.")

# 1. Load Model (Cached so it loads only once)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 3)
    )
    # Load your trained weights from local folder
    model.load_state_dict(torch.load(r'D:\MCA Project\resnet50_ft.pth', map_location=device))
    model.to(device).eval()
    return model, device

model, device = load_model()
class_names = ['Caterpillar', 'Diabrotica Speciosa', 'Healthy']

# 2. File Uploader in Streamlit
uploaded_file = st.file_uploader("Choose an UAV leaf image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open uploaded image
    raw_img = Image.open(uploaded_file).convert('RGB')
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # 3. Hooks for Grad-CAM
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    # Forward & Backward pass
    outputs = model(input_tensor)
    pred_idx = outputs.argmax(dim=1).item()
    confidence = F.softmax(outputs[0], dim=0)[pred_idx].item() * 100

    model.zero_grad()
    outputs[0, pred_idx].backward()

    # Remove hooks
    h1.remove()
    h2.remove()

    # Compute Grad-CAM
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Generate Jet Heatmap using pure NumPy (No matplotlib needed!)
    cam_resized = np.array(Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)) / 255.0
    r = np.clip(1.5 - np.abs(4 * cam_resized - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_resized - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_resized - 1), 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)

    img_np = np.array(raw_img.resize((224, 224))) / 255.0
    overlay = np.uint8(np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1) * 255)
    overlay_img = Image.fromarray(overlay)

    # 4. Display Results on Web App
    st.success(f"**Prediction:** {class_names[pred_idx]}  |  **Confidence:** {confidence:.2f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(raw_img, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay_img, use_container_width=True)