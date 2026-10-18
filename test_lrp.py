from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ResNet50Classifier
from lrp_explainer import LRPExplainer


# ==================================================
# LOAD MODEL
# ==================================================

classifier = ResNet50Classifier()

model = classifier.model
device = classifier.device


# ==================================================
# LOAD IMAGE
# ==================================================

image = Image.open(
    "test_image.jpg"
).convert("RGB")


# ==================================================
# PREPROCESS IMAGE
# ==================================================

input_tensor = classifier.preprocess(
    image
).unsqueeze(0).to(device)


# ==================================================
# GET PREDICTION
# ==================================================

model.eval()

with torch.no_grad():

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    target_class = output.argmax(
        dim=1
    ).item()


class_name = classifier.categories[
    target_class
]

confidence = probabilities[
    0,
    target_class
].item()


print()
print("==============================")
print("           LRP")
print("==============================")

print()

print(
    f"Target class: {class_name}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print(
    f"Class index: {target_class}"
)


# ==================================================
# CREATE LRP EXPLAINER
# ==================================================

explainer = LRPExplainer(
    classifier
)


# ==================================================
# GENERATE LRP
# ==================================================

print()
print(
    "Generating LRP explanation..."
)

output, relevance = explainer.explain(
    input_tensor,
    target_class
)


# ==================================================
# INSPECT RESULT
# ==================================================

print()
print(
    "Raw relevance shape:",
    relevance.shape
)


# ==================================================
# MOVE TO CPU
# ==================================================

relevance = relevance.detach().cpu()


# ==================================================
# REMOVE BATCH DIMENSION
# ==================================================

relevance = relevance.squeeze(
    0
)


# ==================================================
# Convert 3 channels → 1 relevance map
# ==================================================

relevance_map = relevance.sum(
    dim=0
).numpy()


# ==================================================
# NORMALIZE FOR DISPLAY
# ==================================================

max_abs = np.max(
    np.abs(relevance_map)
)

if max_abs != 0:

    relevance_display = (
        relevance_map / max_abs
    )

else:

    relevance_display = relevance_map


# ==================================================
# VISUALIZATION
# ==================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# --------------------------------------------------
# Original
# --------------------------------------------------

axes[0].imshow(
    image
)

axes[0].set_title(
    f"Original Image\n"
    f"{class_name}\n"
    f"{confidence * 100:.2f}%"
)

axes[0].axis(
    "off"
)


# --------------------------------------------------
# Positive relevance
# --------------------------------------------------

positive = np.maximum(
    relevance_display,
    0
)

axes[1].imshow(
    image
)

axes[1].imshow(
    positive,
    cmap="Reds",
    alpha=0.65
)

axes[1].set_title(
    "LRP — Positive Relevance"
)

axes[1].axis(
    "off"
)


# --------------------------------------------------
# Full relevance
# --------------------------------------------------

heatmap = axes[2].imshow(
    relevance_display,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1
)

axes[2].set_title(
    "LRP — Positive / Negative"
)

axes[2].axis(
    "off"
)


# --------------------------------------------------
# Colorbar
# --------------------------------------------------

fig.colorbar(
    heatmap,
    ax=axes[2],
    fraction=0.046,
    pad=0.04,
    label="Normalized relevance"
)


# --------------------------------------------------
# Overall title
# --------------------------------------------------

plt.suptitle(
    "Layer-wise Relevance Propagation — ResNet-50",
    fontsize=18
)

plt.tight_layout()

plt.show()