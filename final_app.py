import os
import time
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Soybean Pest Detection | UAV AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #F4F9F4;
        border: 1px solid #C8E6C9;
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. MODEL LOADING & PREPROCESSING
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(model.fc.in_features, 3)
    )
    model.load_state_dict(
        torch.load(r"D:\MCA Project\resnet50_ft.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model, device


model, device = load_model()
class_names = ["Caterpillar", "Diabrotica Speciosa", "Healthy"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Domain knowledge dictionary for practical decision support
pest_info = {
    "Caterpillar": {
        "severity": "High Foliar Damage Risk",
        "symptoms": (
            "Irregular holes chewed through leaf tissue; severe defoliation"
            " reduces photosynthesis."
        ),
        "action": (
            "Inspect lower canopy leaves; apply targeted biological control"
            " (Bacillus thuringiensis) or recommended selective larvicides if"
            " threshold is exceeded."
        ),
    },
    "Diabrotica Speciosa": {
        "severity": "Moderate-to-High Defoliation Risk",
        "symptoms": (
            "Distinctive small, round 'shot-hole' perforations across leaf"
            " blades caused by adult leaf beetles."
        ),
        "action": (
            "Monitor beetle density per meter row; deploy yellow sticky traps"
            " or targeted foliar treatment if defoliation exceeds 30% in"
            " vegetative stage."
        ),
    },
    "Healthy": {
        "severity": "No Action Required",
        "symptoms": (
            "Intact green leaf lamina with normal chlorophyll distribution and"
            " no visible insect feeding lesions."
        ),
        "action": (
            "Continue routine UAV surveillance and standard irrigation/nutrient"
            " management."
        ),
    },
}

# ---------------------------------------------------------
# 3. SIDEBAR CONTROLS (INPUT SELECTION)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🛸 UAV Image Input")
    uploaded_file = st.file_uploader(
        "Choose a soybean leaf image", type=["jpg", "jpeg", "png", "webp"]
    )

    image = None
    image_label = ""

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_label = uploaded_file.name

    st.divider()
    st.caption("**Model:** Fine-Tuned ResNet-50")
    st.caption("**Explainability:** Pure PyTorch Grad-CAM (Layer 4)")
    st.caption(f"**Compute Device:** `{str(device).upper()}`")

# ---------------------------------------------------------
# 4. MAIN HEADER & TABS
# ---------------------------------------------------------
st.title("🌱 Detection & Classification of Soybean Pests (UAV Imagery)")
st.markdown(
    "Automated crop health diagnostics combining **ResNet-50 Transfer"
    " Learning** with **Grad-CAM Visual Explainability**."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Pest Diagnosis",
    "📊 Model & Training Insights",
    "ℹ️ System Architecture",
])

# =========================================================
# TAB 1: LIVE DIAGNOSIS & GRAD-CAM
# =========================================================
with tab1:
    if image is not None:
        # Run Inference & Measure Latency
        start_time = time.time()
        input_tensor = transform(image).unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        # Hooks for Pure PyTorch Grad-CAM
        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        target_layer = model.layer4[-1]
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

        model.zero_grad()
        outputs[0, pred_idx].backward()

        h1.remove()
        h2.remove()

        inference_ms = (time.time() - start_time) * 1000
        predicted_class = class_names[pred_idx]

        # Compute Grad-CAM Heatmap manually
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Generate Jet Heatmap overlay
        cam_resized = (
            np.array(
                Image.fromarray(np.uint8(255 * cam)).resize(
                    (224, 224), Image.BILINEAR
                )
            )
            / 255.0
        )
        r = np.clip(1.5 - np.abs(4 * cam_resized - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * cam_resized - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * cam_resized - 1), 0, 1)
        heatmap = np.stack([r, g, b], axis=-1)

        resized_img = image.resize((224, 224))
        img_np = np.array(resized_img) / 255.0
        overlay = np.uint8(np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1) * 255)
        cam_visualization = Image.fromarray(overlay)

        # Top Row: KPI Summary Cards
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Diagnosis Status", value=predicted_class)
        m2.metric(label="Model Confidence", value=f"{confidence:.2f}%")
        m3.metric(label="Inference Speed", value=f"{inference_ms:.1f} ms")

        st.write("")

        # Middle Row: Visual Comparison & Probability Breakdown
        col_vis, col_diag = st.columns([1.4, 1])

        with col_vis:
            with st.container(border=True):
                st.subheader("Visual Attention Analysis")
                img_c1, img_c2 = st.columns(2)
                with img_c1:
                    st.image(
                        resized_img,
                        caption=f"Input Leaf ({image_label})",
                        use_container_width=True,
                    )
                with img_c2:
                    st.image(
                        cam_visualization,
                        caption="Grad-CAM Attention Heatmap",
                        use_container_width=True,
                    )
                st.caption(
                    "🔴 **Red/Yellow zones** indicate the exact leaf regions"
                    " driving the model's prediction."
                )

        with col_diag:
            with st.container(border=True):
                st.subheader("Classification Probabilities")
                if predicted_class == "Healthy":
                    st.success(f"**Result:** Healthy Crop ({confidence:.2f}%)")
                else:
                    st.error(
                        f"**Alert:** {predicted_class} Infestation"
                        f" ({confidence:.2f}%)"
                    )

                for i, name in enumerate(class_names):
                    prob_val = float(probs[i].item())
                    st.progress(
                        prob_val, text=f"{name}: **{prob_val*100:.2f}%**"
                    )

            # Agronomic Recommendation Card
            with st.container(border=True):
                st.subheader("📋 Field Advisory")
                info = pest_info[predicted_class]
                st.markdown(f"**Risk Level:** {info['severity']}")
                st.markdown(f"**Visual Symptoms:** {info['symptoms']}")
                st.markdown(f"**Recommended Action:** {info['action']}")

    else:
        st.info(
            "👈 Upload a UAV soybean leaf image from the left sidebar to begin"
            " analysis."
        )

# =========================================================
# TAB 2: MODEL PERFORMANCE & RESEARCH METRICS
# =========================================================
with tab2:
    st.subheader("ResNet-50 Fine-Tuning Performance")
    st.markdown(
        "Evaluation metrics recorded on the validation dataset of UAV soybean"
        " leaf imagery."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Accuracy", "95.0%")
    k2.metric("Backbone Architecture", "ResNet-50")
    k3.metric("Input Resolution", "224 × 224 px")
    k4.metric("Target Classes", "3 Categories")

    st.write("")

    c_left, c_right = st.columns(2)
    with c_left:
        with st.container(border=True):
            st.markdown("### Training Configuration")
            st.markdown("""
- **Transfer Learning Strategy:** Pre-trained ImageNet weights with custom classification head (`Dropout(0.5)` + `Linear(2048, 3)`).
- **Normalization:** Standard ImageNet channel mean and standard deviation.
- **Target Classes:** *Caterpillar*, *Diabrotica speciosa*, and *Healthy*.
- **Explainability Layer:** Final bottleneck block (`model.layer4[-1]`) hooked into Grad-CAM.
            """)

    with c_right:
        with st.container(border=True):
            st.markdown("### Evaluation Artifacts")
            st.info(
                "💡 **Tip for Presentation:** Save your Confusion Matrix and"
                " Accuracy/Loss curves as PNGs and display them here using"
                " `st.image('confusion_matrix.png')`."
            )

# =========================================================
# TAB 3: SYSTEM PIPELINE
# =========================================================
with tab3:
    st.subheader("End-to-End UAV Pest Detection Workflow")
    with st.container(border=True):
        st.markdown("""
1. **UAV Image Acquisition:** High-resolution aerial images of soybean canopy captured via drone surveillance.
2. **Preprocessing & Normalization:** Leaf images resized to `224 × 224` pixels and normalized for deep feature extraction.
3. **Deep Feature Extraction (ResNet-50):** Residual convolutional blocks identify subtle foliar feeding patterns.
4. **Explainable AI (Grad-CAM):** Pure PyTorch hooks highlight spatial regions responsible for the classification decision, ensuring model transparency.
5. **Decision Support Output:** Delivers real-time confidence scores and actionable field recommendations.
        """)