import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Soybean Pest Detection", page_icon="🌱", layout="wide")
st.title("🌱 Detection & Classification of Soybean Pests (UAV Imagery)")
st.markdown("Upload a UAV soybean leaf image to detect pests and visualize model attention using **Grad-CAM**.")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 3)
    )
    model.load_state_dict(torch.load('resnet50_ft.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()
class_names = ['Caterpillar', 'Diabrotica Speciosa', 'Healthy']

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# SIDEBAR UPLOADER
# ---------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload UAV Leaf Image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # ---------------------------------------------------------
    # PURE PYTORCH GRAD-CAM HOOKS
    # ---------------------------------------------------------
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
    probs = F.softmax(outputs[0], dim=0)
    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item() * 100

    model.zero_grad()
    outputs[0, pred_idx].backward()

    h1.remove()
    h2.remove()

    # Generate Heatmap
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Overlay Heatmap on Image
    cam_resized = np.array(Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)) / 255.0
    r = np.clip(1.5 - np.abs(4 * cam_resized - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_resized - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_resized - 1), 0, 1)
    heatmap = np.stack([r, g, b], axis=-1)

    img_np = np.array(image.resize((224, 224))) / 255.0
    overlay = np.uint8(np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1) * 255)
    visualization = Image.fromarray(overlay)

    # ---------------------------------------------------------
    # 3-COLUMN DISPLAY LAYOUT
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.subheader("1. Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. Grad-CAM Focus")
        st.image(visualization, caption="Red regions show pest damage focus", use_container_width=True)

    with col3:
        st.subheader("3. Diagnosis Result")
        if class_names[pred_idx] == 'Healthy':
            st.success(f"**Status:** {class_names[pred_idx]} ({confidence:.2f}%)")
        else:
            st.error(f"**Pest Detected:** {class_names[pred_idx]} ({confidence:.2f}%)")

        st.write("### Class Probabilities:")
        for i, name in enumerate(class_names):
            prob_val = float(probs[i].item())
            st.progress(prob_val, text=f"{name}: {prob_val*100:.2f}%")
else:
    st.info("👈 Please upload a leaf image from the left sidebar to test the model.")