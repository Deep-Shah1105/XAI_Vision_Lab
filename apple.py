
import os

# Windows/Anaconda can load multiple OpenMP runtimes when PyTorch, NumPy,
# scikit-image, SHAP, and other scientific packages are used together.
# Set this BEFORE importing scientific libraries so the Streamlit app can start.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import io
import numpy as np
import cv2
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries

from model import ResNet50Classifier
from gradcam import GradCAM, overlay_heatmap
from lime_explainer import LIMEExplainer
from kernel_shap_explainer import KernelSHAPExplainer
from lrp_explainer import LRPExplainer


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="XAI Vision Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.72;
        margin-top: 0.25rem;
        margin-bottom: 1.5rem;
    }

    .prediction-card {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.07);
        margin-bottom: 0.7rem;
    }

    .method-card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.20);
        min-height: 110px;
    }

    .method-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .method-tag {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        opacity: 0.7;
        text-transform: uppercase;
    }

    .small-text {
        font-size: 0.9rem;
        opacity: 0.72;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD MODEL ONCE
# ---------------------------------------------------------

@st.cache_resource
def load_classifier():
    return ResNet50Classifier()


@st.cache_resource
def load_lime(_classifier):
    return LIMEExplainer(_classifier)


classifier = load_classifier()
model = classifier.model
device = classifier.device


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_prediction(image):
    input_tensor = classifier.preprocess(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    target_class = int(output.argmax(dim=1).item())
    confidence = float(probabilities[0, target_class].item())

    top_probs, top_indices = torch.topk(probabilities, 5)

    top5 = []
    for p, idx in zip(top_probs[0], top_indices[0]):
        top5.append(
            {
                "class": classifier.categories[idx.item()],
                "probability": float(p.item()),
            }
        )

    return input_tensor, target_class, confidence, top5


def positive_relevance_map(relevance):
    rel = relevance[0].detach().cpu().numpy()
    rel = rel.sum(axis=0)

    rel = np.maximum(rel, 0)
    if rel.max() > 0:
        rel = rel / rel.max()

    return rel


def signed_relevance_map(relevance):
    rel = relevance[0].detach().cpu().numpy()
    rel = rel.sum(axis=0)

    max_abs = np.max(np.abs(rel))
    if max_abs > 0:
        rel = rel / max_abs

    return rel


def heatmap_overlay(image, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.45):
    image_np = np.array(image).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # LRP relevance is produced at the model input size (224x224),
    # while the uploaded image can have any resolution. Resize the
    # relevance map before combining it with the original image.
    heatmap = cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    result = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def signed_overlay(image, signed_map, alpha=0.60):
    image_np = np.array(image).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Resize the model-space relevance map to the uploaded image size.
    signed_map = cv2.resize(
        np.asarray(signed_map, dtype=np.float32),
        (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = ((signed_map + 1.0) / 2.0)
    normalized = np.clip(normalized, 0, 1)

    cmap = plt.get_cmap("RdBu_r")
    rgba = cmap(normalized)
    rgb = np.uint8(rgba[:, :, :3] * 255)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        rgb_bgr,
        alpha,
        0,
    )

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def build_lime_visual(image, explanation, target_class):
    # Standard LIME visualization:
    # important superpixels supporting/opposing the target class.
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=False,
        num_features=15,
        hide_rest=False,
    )

    temp = np.clip(temp, 0, 1)
    visual = mark_boundaries(temp, mask, color=(1, 1, 0), mode="thick")

    return visual


def build_kernel_shap_visual(image, shap_values, shap_explainer):
    values = np.asarray(shap_values)
    values = np.squeeze(values)

    segments = shap_explainer.segments
    number_of_segments = int(segments.max()) + 1

    if values.shape[0] != number_of_segments:
        raise ValueError(
            f"KernelSHAP returned {values.shape[0]} values "
            f"but the image has {number_of_segments} superpixels."
        )

    shap_map = np.zeros(segments.shape, dtype=np.float32)

    for segment_id in range(number_of_segments):
        shap_map[segments == segment_id] = values[segment_id]

    max_abs = np.max(np.abs(shap_map))

    if max_abs > 0:
        signed = shap_map / max_abs
    else:
        signed = shap_map

    return signed_overlay(image, signed)


def build_lrp_visuals(image, relevance):
    positive = positive_relevance_map(relevance)
    signed = signed_relevance_map(relevance)

    positive_visual = heatmap_overlay(
        image,
        positive,
        colormap=cv2.COLORMAP_JET,
        alpha=0.45,
    )

    signed_visual = signed_overlay(
        image,
        signed,
        alpha=0.60,
    )

    return positive_visual, signed_visual


def run_gradcam(image_tensor, target_class):
    # Main Grad-CAM view: deepest convolutional stage.
    target_layer = model.layer4[-1].conv3

    explainer = GradCAM(
        model,
        target_layer,
        device,
    )

    cam, _ = explainer.generate(
        image_tensor,
        target_class=target_class,
    )

    result = overlay_heatmap(
        Image.open(io.BytesIO(st.session_state["image_bytes"])).convert("RGB"),
        cam,
        alpha=0.45,
    )

    explainer.remove_hooks()

    return result


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 XAI Vision Lab</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'One ResNet-50 decision, explained four different ways.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Experiment")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    st.markdown("---")

    st.subheader("XAI settings")

    lime_samples = st.slider(
        "LIME samples",
        min_value=250,
        max_value=1000,
        value=500,
        step=250,
        help="More samples usually give a more stable LIME explanation, but take longer.",
    )

    shap_samples = st.slider(
        "KernelSHAP samples",
        min_value=200,
        max_value=750,
        value=500,
        step=100,
        help="More samples usually improve the approximation but increase runtime.",
    )

    shap_segments = st.slider(
        "KernelSHAP superpixels",
        min_value=50,
        max_value=100,
        value=75,
        step=5,
        help="The superpixels are the interpretable features given to KernelSHAP.",
    )

    run_button = st.button(
        "🚀 Run XAI Analysis",
        width=True,
        type="primary",
    )

    st.markdown("---")

    st.caption(f"Device: {device}")
    if torch.cuda.is_available():
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
    st.caption("Model: ResNet-50 / ImageNet")


# ---------------------------------------------------------
# IMAGE INPUT
# ---------------------------------------------------------

if uploaded_file is None:
    st.info("Upload an image from the sidebar to begin.")
    st.stop()

image_bytes = uploaded_file.getvalue()
st.session_state["image_bytes"] = image_bytes

image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Prediction is cheap, so show it immediately.
input_tensor, target_class, confidence, top5 = get_prediction(image)
class_name = classifier.categories[target_class]


# ---------------------------------------------------------
# PREDICTION HEADER
# ---------------------------------------------------------

col1, col2, col3 = st.columns([1.7, 1, 1])

with col1:
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="small-text">RESNET-50 PREDICTION</div>
            <h2 style="margin: 0.2rem 0;">{class_name}</h2>
            <div class="small-text">
                Explaining this exact top-1 prediction across all four methods.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric("Confidence", f"{confidence * 100:.2f}%")

with col3:
    st.metric("Input", f"{image.width} × {image.height}")


# ---------------------------------------------------------
# ORIGINAL + TOP-5
# ---------------------------------------------------------

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Original Image")
    st.image(image, width=True)

with right:
    st.subheader("Top-5 Predictions")

    for rank, item in enumerate(top5, start=1):
        st.write(
            f"**{rank}. {item['class']}** — "
            f"{item['probability'] * 100:.2f}%"
        )
        st.progress(item["probability"])


st.divider()


# ---------------------------------------------------------
# RUN EXPLANATIONS
# ---------------------------------------------------------

if run_button:
    st.session_state["run_complete"] = False

    progress = st.progress(0)
    status = st.empty()

    try:
        # -------------------------------------------------
        # 1. LIME
        # -------------------------------------------------
        status.info("1/4 — Generating LIME explanation...")
        lime_explainer = load_lime(classifier)

        lime_result = lime_explainer.explain(
            image,
            num_samples=lime_samples,
        )

        lime_visual = build_lime_visual(
            image,
            lime_result,
            lime_result.top_labels[0],
        )

        progress.progress(25)

        # -------------------------------------------------
        # 2. KernelSHAP
        # -------------------------------------------------
        status.info("2/4 — Generating KernelSHAP explanation...")
        shap_explainer = KernelSHAPExplainer(
            classifier,
            n_segments=shap_segments,
            compactness=10,
        )

        shap_values, _, shap_target_class = shap_explainer.explain(
            image,
            nsamples=shap_samples,
        )

        shap_visual = build_kernel_shap_visual(
            image,
            shap_values,
            shap_explainer,
        )

        progress.progress(50)

        # -------------------------------------------------
        # 3. Grad-CAM
        # -------------------------------------------------
        status.info("3/4 — Generating Grad-CAM explanation...")
        gradcam_visual = run_gradcam(
            input_tensor,
            target_class,
        )

        progress.progress(75)

        # -------------------------------------------------
        # 4. LRP
        # -------------------------------------------------
        status.info("4/4 — Generating LRP explanation...")
        lrp_explainer = LRPExplainer(classifier)

        _, relevance = lrp_explainer.explain(
            input_tensor,
            target_class,
        )

        lrp_positive, lrp_signed = build_lrp_visuals(
            image,
            relevance,
        )

        progress.progress(100)
        status.success("All four explanations generated.")

        st.session_state["lime_visual"] = lime_visual
        st.session_state["shap_visual"] = shap_visual
        st.session_state["gradcam_visual"] = gradcam_visual
        st.session_state["lrp_positive"] = lrp_positive
        st.session_state["lrp_signed"] = lrp_signed
        st.session_state["run_complete"] = True

    except Exception as exc:
        st.error("The XAI pipeline encountered an error.")
        st.exception(exc)
        st.stop()


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

if st.session_state.get("run_complete", False):

    st.divider()

    st.header("🔍 Explanation Comparison")

    # ---- Row 1: LIME / KernelSHAP
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME")
        st.caption(
            "Model-agnostic • local surrogate • superpixel importance"
        )
        st.image(
            st.session_state["lime_visual"],
            width=True,
        )

    with col2:
        st.subheader("KernelSHAP")
        st.caption(
            "Model-agnostic • Shapley-based • superpixel contributions"
        )
        st.image(
            st.session_state["shap_visual"],
            width=True,
        )

    # ---- Row 2: Grad-CAM / LRP
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grad-CAM")
        st.caption(
            "Model-specific • gradient-weighted activation localization • Layer 4"
        )
        st.image(
            st.session_state["gradcam_visual"],
            width=True,
        )

    with col2:
        st.subheader("LRP")
        st.caption(
            "Model-specific • relevance propagation • ResNet-aware canonicalization"
        )
        st.image(
            st.session_state["lrp_positive"],
            width=True,
        )

    # ---- LRP signed view
    with st.expander("LRP: Positive / Negative relevance"):
        st.image(
            st.session_state["lrp_signed"],
            width   =True,
        )

        st.markdown(
            """
            **Red:** evidence supporting the selected class.

            **Blue:** evidence opposing the selected class.

            **White:** weak or near-zero contribution.
            """
        )

    st.divider()

    # -----------------------------------------------------
    # METHOD CARDS
    # -----------------------------------------------------

    st.header("🧪 What each method is doing")

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (
            c1,
            "LIME",
            "MODEL-AGNOSTIC",
            "Perturbs image regions and learns a simple local model around the prediction.",
        ),
        (
            c2,
            "KernelSHAP",
            "MODEL-AGNOSTIC",
            "Treats superpixels as features and estimates their Shapley contributions.",
        ),
        (
            c3,
            "Grad-CAM",
            "MODEL-SPECIFIC",
            "Uses gradients flowing through deep convolutional feature maps to localize evidence.",
        ),
        (
            c4,
            "LRP",
            "MODEL-SPECIFIC",
            "Propagates relevance backward through the network to assign input-level importance.",
        ),
    ]

    for column, title, tag, description in cards:
        with column:
            st.markdown(
                f"""
                <div class="method-card">
                    <div class="method-tag">{tag}</div>
                    <div class="method-title">{title}</div>
                    <div class="small-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    st.divider()
    st.subheader("Ready to explain the decision?")
    st.caption(
        "Press **Run XAI Analysis** in the sidebar. "
        "LIME and KernelSHAP are the slower stages; Grad-CAM and LRP are usually much faster."
    )