
import os

# =========================================================
# WINDOWS / ANACONDA OPENMP WORKAROUND
# =========================================================
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# =========================================================
# IMPORTS
# =========================================================
import io
import hashlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from PIL import Image

from model import ResNet50Classifier
from gradcam import GradCAM, overlay_heatmap
from lime_explainer import LIMEExplainer
from kernel_shap_explainer import KernelSHAPExplainer
from lrp_explainer import LRPExplainer


# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="XAI Vision Lab",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# SIMPLE CSS
# =========================================================
st.markdown(
    """
    <style>
    .big-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 17px;
        opacity: 0.70;
        margin-top: 0;
    }

    .method-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        opacity: 0.65;
        text-transform: uppercase;
    }

    .slide-count {
        text-align: center;
        font-weight: 600;
        opacity: 0.70;
        padding-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_classifier():
    return ResNet50Classifier()


# IMPORTANT:
# Leading underscore prevents Streamlit from trying to hash
# the custom ResNet50Classifier object.
@st.cache_resource
def load_lime(_classifier):
    return LIMEExplainer(_classifier)


classifier = load_classifier()

model = classifier.model
device = classifier.device


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
DEFAULT_SESSION_STATE = {
    "run_complete": False,
    "lime_slides": [],
    "shap_visual": None,
    "gradcam_slides": [],
    "lrp_slides": [],
    "image_hash": None,
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# PREDICTION
# =========================================================
def get_prediction(image):

    input_tensor = classifier.preprocess(image)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    target_class = int(output.argmax(dim=1).item())
    confidence = float(probabilities[0, target_class].item())

    top_probs, top_indices = torch.topk(probabilities, 5)

    top5 = []

    for probability, index in zip(
        top_probs[0],
        top_indices[0]
    ):
        top5.append(
            {
                "class": classifier.categories[index.item()],
                "probability": float(probability.item()),
            }
        )

    return input_tensor, target_class, confidence, top5


# =========================================================
# GENERIC HEATMAP OVERLAY
# =========================================================
def resize_map_to_image(heatmap, image):

    image_np = np.asarray(image)
    height, width = image_np.shape[:2]

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    return cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )


def heatmap_overlay_safe(
    image,
    heatmap,
    colormap=cv2.COLORMAP_JET,
    alpha=0.45,
):
    """
    Safe heatmap overlay that always resizes the explanation
    to the uploaded image dimensions.
    """

    image_np = np.asarray(image).astype(np.uint8)

    heatmap = resize_map_to_image(
        heatmap,
        image_np
    )

    heatmap = np.clip(
        heatmap,
        0,
        1
    )

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        colormap
    )

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    result = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )


# =========================================================
# SIGNED MAP OVERLAY
# =========================================================
def signed_overlay(
    image,
    signed_map,
    alpha=0.60,
):
    """
    Signed relevance:
    positive -> red
    zero     -> white
    negative -> blue
    """

    image_np = np.asarray(image).astype(np.uint8)

    signed_map = resize_map_to_image(
        signed_map,
        image_np
    )

    max_abs = np.max(
        np.abs(signed_map)
    )

    if max_abs > 0:
        signed_map = signed_map / max_abs

    signed_map = np.clip(
        signed_map,
        -1,
        1
    )

    # Map -1..1 -> 0..1 for RdBu_r
    normalized = (
        signed_map + 1.0
    ) / 2.0

    cmap = plt.get_cmap("RdBu_r")

    rgba = cmap(normalized)

    rgb = np.uint8(
        rgba[:, :, :3] * 255
    )

    heat_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    result = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        heat_bgr,
        alpha,
        0
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )


# =========================================================
# LIME VISUALIZATION
# =========================================================
def build_lime_slides(
    image,
    explanation,
):
    """
    Create three LIME views using the original image:
      1. Supporting regions
      2. Opposing regions
      3. Positive + negative regions

    This avoids the black-image problem caused by treating
    LIME's returned 0..255 image as if it were 0..1.
    """

    image_np = np.asarray(
        image
    ).astype(np.uint8)

    segments = explanation.segments

    # LIME stores:
    # (superpixel_id, local surrogate weight)
    target_class = explanation.top_labels[0]

    local_exp = explanation.local_exp[target_class]

    number_of_segments = (
        int(segments.max()) + 1
    )

    weights = np.zeros(
        number_of_segments,
        dtype=np.float32
    )

    for segment_id, weight in local_exp:
        if 0 <= segment_id < number_of_segments:
            weights[segment_id] = weight

    positive_map = np.zeros_like(
        segments,
        dtype=np.float32
    )

    negative_map = np.zeros_like(
        segments,
        dtype=np.float32
    )

    for segment_id in range(
        number_of_segments
    ):

        positive_map[
            segments == segment_id
        ] = max(
            weights[segment_id],
            0
        )

        negative_map[
            segments == segment_id
        ] = max(
            -weights[segment_id],
            0
        )

    max_positive = positive_map.max()
    max_negative = negative_map.max()

    if max_positive > 0:
        positive_map /= max_positive

    if max_negative > 0:
        negative_map /= max_negative

    # -----------------------------------------------------
    # Slide 1: supporting evidence
    # -----------------------------------------------------
    positive_visual = image_np.copy()

    positive_color = np.zeros_like(
        image_np
    )

    # Green = supporting evidence
    positive_color[:, :, 1] = 255

    mask = positive_map[:, :, None]

    positive_visual = (
        positive_visual.astype(np.float32) * (1 - 0.48 * mask)
        + positive_color.astype(np.float32) * (0.48 * mask)
    ).clip(0, 255).astype(np.uint8)

    # Yellow boundaries
    positive_boundaries = mark_boundaries_rgb(
        positive_visual,
        segments,
        positive_map
    )

    # -----------------------------------------------------
    # Slide 2: opposing evidence
    # -----------------------------------------------------
    negative_visual = image_np.copy()

    negative_color = np.zeros_like(
        image_np
    )

    # Red = opposing evidence
    negative_color[:, :, 0] = 255

    mask = negative_map[:, :, None]

    negative_visual = (
        negative_visual.astype(np.float32) * (1 - 0.48 * mask)
        + negative_color.astype(np.float32) * (0.48 * mask)
    ).clip(0, 255).astype(np.uint8)

    negative_boundaries = mark_boundaries_rgb(
        negative_visual,
        segments,
        negative_map
    )

    # -----------------------------------------------------
    # Slide 3: positive + negative
    # -----------------------------------------------------
    combined_visual = image_np.astype(
        np.float32
    ).copy()

    # Green positive
    positive_weight = positive_map[:, :, None]

    green_overlay = np.zeros_like(
        combined_visual
    )

    green_overlay[:, :, 1] = 255

    combined_visual = (
        combined_visual * (1 - 0.40 * positive_weight)
        + green_overlay * (0.40 * positive_weight)
    )

    # Red negative
    negative_weight = negative_map[:, :, None]

    red_overlay = np.zeros_like(
        combined_visual
    )

    red_overlay[:, :, 0] = 255

    combined_visual = (
        combined_visual * (1 - 0.40 * negative_weight)
        + red_overlay * (0.40 * negative_weight)
    )

    combined_visual = (
        combined_visual
        .clip(0, 255)
        .astype(np.uint8)
    )

    combined_visual = mark_boundaries_rgb(
        combined_visual,
        segments,
        np.maximum(
            positive_map,
            negative_map
        )
    )

    return [
        (
            "Supporting evidence",
            "Green regions contributed positively to the selected class.",
            positive_boundaries
        ),
        (
            "Opposing evidence",
            "Red regions contributed negatively to the selected class.",
            negative_boundaries
        ),
        (
            "Positive + negative",
            "Green = support • Red = opposition • Original image remains visible.",
            combined_visual
        )
    ]


def mark_boundaries_rgb(
    image,
    segments,
    importance_map,
):
    """
    Draw superpixel boundaries without turning the image black.
    """

    result = image.astype(
        np.float32
    ) / 255.0

    # Boundaries between SLIC/LIME regions.
    boundary = np.zeros(
        segments.shape,
        dtype=bool
    )

    boundary[:-1, :] |= (
        segments[:-1, :] != segments[1:, :]
    )

    boundary[:, :-1] |= (
        segments[:, :-1] != segments[:, 1:]
    )

    # Only draw boundaries around regions with meaningful
    # LIME importance.
    threshold = 0.05

    boundary &= (
        importance_map > threshold
    )

    # Yellow boundary
    result[boundary] = (
        1.0,
        1.0,
        0.0
    )

    return (
        result
        .clip(0, 1)
        * 255
    ).astype(np.uint8)


# =========================================================
# KERNELSHAP VISUALIZATION
# =========================================================
def build_kernel_shap_visual(
    image,
    shap_values,
    shap_explainer,
):

    values = np.asarray(
        shap_values
    )

    values = np.squeeze(
        values
    )

    segments = (
        shap_explainer.segments
    )

    number_of_segments = (
        int(segments.max()) + 1
    )

    if values.shape[0] != number_of_segments:
        raise ValueError(
            f"KernelSHAP returned "
            f"{values.shape[0]} values "
            f"but the image has "
            f"{number_of_segments} superpixels."
        )

    shap_map = np.zeros(
        segments.shape,
        dtype=np.float32
    )

    for segment_id in range(
        number_of_segments
    ):
        shap_map[
            segments == segment_id
        ] = values[segment_id]

    return signed_overlay(
        image,
        shap_map
    )


# =========================================================
# GRAD-CAM — ALL FOUR RESNET STAGES
# =========================================================
def build_gradcam_slides(
    image,
    input_tensor,
    target_class,
):

    target_layers = {
        "Layer 1": model.layer1[-1].conv3,
        "Layer 2": model.layer2[-1].conv3,
        "Layer 3": model.layer3[-1].conv3,
        "Layer 4": model.layer4[-1].conv3,
    }

    slides = []

    for layer_name, target_layer in target_layers.items():

        explainer = GradCAM(
            model,
            target_layer,
            device,
        )

        cam, _ = explainer.generate(
            input_tensor,
            target_class=target_class,
        )

        visual = overlay_heatmap(
            image,
            cam,
            alpha=0.45,
        )

        explainer.remove_hooks()

        if layer_name == "Layer 1":
            description = (
                "Early visual features — edges, color transitions "
                "and basic textures."
            )

        elif layer_name == "Layer 2":
            description = (
                "Intermediate features — simple shapes and local patterns."
            )

        elif layer_name == "Layer 3":
            description = (
                "Higher-level features — increasingly meaningful "
                "object parts."
            )

        else:
            description = (
                "Deepest convolutional features — the most semantic "
                "view used for final-class reasoning."
            )

        slides.append(
            (
                layer_name,
                description,
                visual
            )
        )

    return slides


# =========================================================
# LRP — TWO VIEWS
# =========================================================
def build_lrp_slides(
    image,
    relevance,
):

    rel = (
        relevance[0]
        .detach()
        .cpu()
        .numpy()
    )

    # Sum RGB channels
    rel = rel.sum(
        axis=0
    )

    # Positive relevance
    positive = np.maximum(
        rel,
        0
    )

    if positive.max() > 0:
        positive /= positive.max()

    positive_visual = heatmap_overlay_safe(
        image,
        positive,
        colormap=cv2.COLORMAP_JET,
        alpha=0.45,
    )

    # Signed relevance
    signed_visual = signed_overlay(
        image,
        rel,
        alpha=0.60,
    )

    return [
        (
            "Positive relevance",
            "Red regions are pixels/features supporting the selected class.",
            positive_visual,
        ),
        (
            "Positive + negative relevance",
            "Red = supporting evidence • Blue = opposing evidence • "
            "White = near-zero relevance.",
            signed_visual,
        ),
    ]


# =========================================================
# CAROUSEL
# =========================================================
def render_carousel(
    method_key,
    title,
    slides,
):
    """
    Renders one explanation image at a time inside one box.

    The Previous / Next buttons make it behave like a slide deck
    instead of placing several large images vertically.
    """

    if not slides:
        return

    state_key = f"{method_key}_slide"

    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    current = st.session_state[state_key]

    # Keep state valid if number of slides changes.
    current %= len(slides)

    slide_title, slide_description, slide_image = slides[current]

    with st.expander(
        f"🔍 {title} — slide view",
        expanded=True
    ):

        # Slide title
        st.subheader(
            slide_title
        )

        st.caption(
            slide_description
        )

        # Single image shown in the box
        st.image(
            slide_image
        )

        # Navigation
        prev_col, count_col, next_col = st.columns(
            [1, 2, 1]
        )

        with prev_col:

            if st.button(
                "← Previous",
                key=f"{method_key}_prev",
            ):
                st.session_state[state_key] = (
                    current - 1
                ) % len(slides)

                st.rerun()

        with count_col:

            st.markdown(
                f"""
                <div class="slide-count">
                    Slide {current + 1} of {len(slides)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with next_col:

            if st.button(
                "Next →",
                key=f"{method_key}_next",
            ):
                st.session_state[state_key] = (
                    current + 1
                ) % len(slides)

                st.rerun()


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="big-title">🧠 XAI Vision Lab</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'One ResNet-50 decision, explained four different ways.'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.header(
        "Experiment"
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
    )

    st.divider()

    st.subheader(
        "XAI Settings"
    )

    lime_samples = st.slider(
        "LIME samples",
        250,
        1000,
        500,
        250,
    )

    shap_samples = st.slider(
        "KernelSHAP samples",
        200,
        750,
        500,
        100,
    )

    shap_segments = st.slider(
        "KernelSHAP superpixels",
        50,
        100,
        75,
        5,
    )

    run_button = st.button(
        "🚀 Run XAI Analysis",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.caption(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        st.caption(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    st.caption(
        "Model: ResNet-50 / ImageNet"
    )


# =========================================================
# WAIT FOR IMAGE
# =========================================================
if uploaded_file is None:

    st.info(
        "Upload an image from the sidebar to begin."
    )

    st.stop()


# =========================================================
# LOAD IMAGE
# =========================================================
image_bytes = uploaded_file.getvalue()

current_image_hash = hashlib.md5(
    image_bytes
).hexdigest()

if st.session_state.get("image_hash") != current_image_hash:
    st.session_state["image_hash"] = current_image_hash
    st.session_state["run_complete"] = False
    st.session_state["lime_slides"] = []
    st.session_state["shap_visual"] = None
    st.session_state["gradcam_slides"] = []
    st.session_state["lrp_slides"] = []

image = Image.open(
    io.BytesIO(
        image_bytes
    )
).convert("RGB")


# =========================================================
# PREDICTION
# =========================================================
(
    input_tensor,
    target_class,
    confidence,
    top5,
) = get_prediction(
    image
)

class_name = classifier.categories[
    target_class
]


# =========================================================
# PREDICTION HEADER
# =========================================================
c1, c2, c3 = st.columns(
    [1.7, 1, 1]
)

with c1:

    st.markdown(
        f"### ResNet-50 Prediction\n"
        f"## {class_name}"
    )

    st.caption(
        "The four XAI methods will explain this exact top-1 prediction."
    )

with c2:

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

with c3:

    st.metric(
        "Image Size",
        f"{image.width} × {image.height}"
    )


# =========================================================
# ORIGINAL + TOP 5
# =========================================================
left, right = st.columns(
    [1.25, 1]
)

with left:

    st.subheader(
        "Original Image"
    )

    st.image(
        image
    )

with right:

    st.subheader(
        "Top-5 Predictions"
    )

    for rank, item in enumerate(
        top5,
        start=1
    ):

        st.write(
            f"**{rank}. {item['class']}** — "
            f"{item['probability'] * 100:.2f}%"
        )

        st.progress(
            item["probability"]
        )


st.divider()


# =========================================================
# XAI ANALYSIS
# =========================================================
if run_button:

    # Reset all previous slide positions
    st.session_state["lime_slide"] = 0
    st.session_state["gradcam_slide"] = 0
    st.session_state["lrp_slide"] = 0

    st.session_state["run_complete"] = False

    progress = st.progress(
        0
    )

    status = st.empty()

    try:

        # =================================================
        # 1. LIME
        # =================================================
        status.info(
            "1/4 — Generating LIME explanation..."
        )

        lime_explainer = load_lime(
            classifier
        )

        lime_result = lime_explainer.explain(
            image,
            num_samples=lime_samples,
        )

        lime_slides = build_lime_slides(
            image,
            lime_result,
        )

        progress.progress(
            25
        )


        # =================================================
        # 2. KERNELSHAP
        # =================================================
        status.info(
            "2/4 — Generating KernelSHAP explanation..."
        )

        shap_explainer = KernelSHAPExplainer(
            classifier,
            n_segments=shap_segments,
            compactness=10,
        )

        (
            shap_values,
            _,
            _
        ) = shap_explainer.explain(
            image,
            nsamples=shap_samples,
        )

        shap_visual = build_kernel_shap_visual(
            image,
            shap_values,
            shap_explainer,
        )

        progress.progress(
            50
        )


        # =================================================
        # 3. GRAD-CAM
        # =================================================
        status.info(
            "3/4 — Generating Grad-CAM explanation..."
        )

        gradcam_slides = build_gradcam_slides(
            image,
            input_tensor,
            target_class,
        )

        progress.progress(
            75
        )


        # =================================================
        # 4. LRP
        # =================================================
        status.info(
            "4/4 — Generating LRP explanation..."
        )

        lrp_explainer = LRPExplainer(
            classifier
        )

        (
            _,
            relevance
        ) = lrp_explainer.explain(
            input_tensor,
            target_class,
        )

        lrp_slides = build_lrp_slides(
            image,
            relevance,
        )

        progress.progress(
            100
        )

        status.success(
            "All four explanations generated successfully!"
        )


        # =================================================
        # STORE RESULTS
        # =================================================
        st.session_state["lime_slides"] = lime_slides

        st.session_state["shap_visual"] = shap_visual

        st.session_state["gradcam_slides"] = gradcam_slides

        st.session_state["lrp_slides"] = lrp_slides

        st.session_state["run_complete"] = True


    except Exception as exc:

        st.error(
            "The XAI pipeline encountered an error."
        )

        st.exception(
            exc
        )

        st.stop()


# =========================================================
# EXPLANATION RESULTS
# =========================================================
results_ready = (
    st.session_state.get("run_complete", False)
    and bool(st.session_state.get("lime_slides"))
    and st.session_state.get("shap_visual") is not None
    and bool(st.session_state.get("gradcam_slides"))
    and bool(st.session_state.get("lrp_slides"))
)

if results_ready:

    st.divider()

    st.header(
        "🔬 Explanation Comparison"
    )

    st.caption(
        "Each explainer is presented as a slide deck so multiple "
        "views do not consume the entire page."
    )


    # =====================================================
    # LIME
    # =====================================================
    render_carousel(
        "lime",
        "LIME",
        st.session_state.get("lime_slides", []),
    )


    # =====================================================
    # KERNELSHAP
    # =====================================================
    with st.expander(
        "🔬 KernelSHAP",
        expanded=True
    ):

        st.subheader(
            "Superpixel contribution map"
        )

        st.caption(
            "Red indicates positive contribution to the selected class; "
            "blue indicates negative contribution."
        )

        st.image(
            st.session_state.get("shap_visual")
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================
    render_carousel(
        "gradcam",
        "Grad-CAM",
        st.session_state.get("gradcam_slides", []),
    )


    # =====================================================
    # LRP
    # =====================================================
    render_carousel(
        "lrp",
        "LRP",
        st.session_state.get("lrp_slides", []),
    )


    # =====================================================
    # METHOD EXPLANATIONS
    # =====================================================
    st.divider()

    st.header(
        "🧪 What Each Method Is Doing"
    )

    m1, m2 = st.columns(2)

    with m1:

        st.subheader(
            "LIME"
        )

        st.caption(
            "MODEL-AGNOSTIC"
        )

        st.write(
            "LIME temporarily hides or perturbs regions of the image "
            "and observes how the model's prediction changes. It then "
            "fits a simple local surrogate model to estimate which "
            "image regions were important."
        )

        st.write(
            "**Think of it as:** "
            "“What happens to the prediction when I remove this region?”"
        )


    with m2:

        st.subheader(
            "KernelSHAP"
        )

        st.caption(
            "MODEL-AGNOSTIC"
        )

        st.write(
            "KernelSHAP treats the image's superpixels as interpretable "
            "features. It evaluates different combinations of those "
            "features and estimates their Shapley-value contributions "
            "to the selected prediction."
        )

        st.write(
            "**Think of it as:** "
            "“How much credit does each region deserve for this prediction?”"
        )


    m3, m4 = st.columns(2)

    with m3:

        st.subheader(
            "Grad-CAM"
        )

        st.caption(
            "MODEL-SPECIFIC"
        )

        st.write(
            "Grad-CAM uses the gradients of the selected class with "
            "respect to convolutional feature maps. The resulting "
            "heatmap shows which spatial regions of a convolutional "
            "layer were most influential for that class."
        )

        st.write(
            "**Think of it as:** "
            "“Where in the image were the deep features looking?”"
        )


    with m4:

        st.subheader(
            "LRP"
        )

        st.caption(
            "MODEL-SPECIFIC"
        )

        st.write(
            "Layer-wise Relevance Propagation starts from the selected "
            "prediction and propagates relevance backward through the "
            "network. The relevance is ultimately assigned to the input "
            "pixels, showing which features supported or opposed the "
            "decision."
        )

        st.write(
            "**Think of it as:** "
            "“How can the model's decision be traced back to the pixels?”"
        )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================
    st.divider()

    st.subheader(
        "🧠 The Four Explanations at a Glance"
    )

    st.write(
        "**LIME** → local perturbation-based explanation"
    )

    st.write(
        "**KernelSHAP** → Shapley-based feature contribution"
    )

    st.write(
        "**Grad-CAM** → gradient-based deep feature localization"
    )

    st.write(
        "**LRP** → backward relevance propagation to the input"
    )


else:

    st.divider()

    st.subheader(
        "Ready to explain the decision?"
    )

    st.caption(
        "Choose an image and press **Run XAI Analysis** in the sidebar. "
        "LIME and KernelSHAP are usually the slowest stages."
    )
