from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import ResNet50Classifier
from kernel_shap_explainer import KernelSHAPExplainer


# ==================================================
# LOAD MODEL
# ==================================================

classifier = ResNet50Classifier()


# ==================================================
# LOAD IMAGE
# ==================================================

image = Image.open(
    "test_image.jpg"
).convert("RGB")


# ==================================================
# CREATE KERNELSHAP EXPLAINER
# ==================================================

explainer = KernelSHAPExplainer(
    classifier,

    # Ask SLIC for approximately 75 regions
    n_segments=75,

    compactness=10
)


# ==================================================
# GENERATE EXPLANATION
# ==================================================

print()
print("==============================")
print("       KernelSHAP XAI")
print("==============================")

shap_values, probabilities, target_class = (
    explainer.explain(
        image,

        # Number of sampled coalitions
        nsamples=500
    )
)


# ==================================================
# PREDICTION
# ==================================================

class_name = classifier.categories[
    target_class
]

confidence = probabilities[
    target_class
]


print()
print("==============================")
print("Prediction")
print("==============================")

print(
    f"Class: {class_name}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


# ==================================================
# EXTRACT SHAP VALUES
# ==================================================

shap_values = np.asarray(
    shap_values
)

print()
print(
    "Raw SHAP shape:",
    shap_values.shape
)


# Remove unnecessary dimensions
shap_values = np.squeeze(
    shap_values
)

print(
    "Processed SHAP shape:",
    shap_values.shape
)


# ==================================================
# CREATE REGION SHAP MAP
# ==================================================

segments = explainer.segments

shap_map = np.zeros(
    segments.shape,
    dtype=np.float32
)


number_of_segments = (
    int(segments.max()) + 1
)


# Safety check
if len(shap_values) != number_of_segments:

    raise ValueError(
        f"SHAP returned {len(shap_values)} "
        f"values, but there are "
        f"{number_of_segments} superpixels."
    )


for segment_id in range(
    number_of_segments
):

    shap_map[
        segments == segment_id
    ] = shap_values[
        segment_id
    ]


# ==================================================
# COLOR RANGE
# ==================================================

max_value = np.max(
    np.abs(shap_map)
)

if max_value == 0:

    max_value = 1


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
# Superpixels
# --------------------------------------------------

axes[1].imshow(
    image
)

axes[1].imshow(
    segments,
    alpha=0.45,
    cmap="nipy_spectral"
)

axes[1].set_title(
    f"SLIC Superpixels\n"
    f"{number_of_segments} regions"
)

axes[1].axis(
    "off"
)


# --------------------------------------------------
# KernelSHAP
# --------------------------------------------------

heatmap = axes[2].imshow(
    shap_map,

    cmap="RdBu_r",

    vmin=-max_value,

    vmax=max_value
)

axes[2].set_title(
    "KernelSHAP Contribution"
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
    label="SHAP contribution"
)


# --------------------------------------------------
# Overall title
# --------------------------------------------------

plt.suptitle(
    "KernelSHAP — Model-Agnostic Explanation",
    fontsize=18
)

plt.tight_layout()

plt.show()