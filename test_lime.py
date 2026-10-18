from PIL import Image
import matplotlib.pyplot as plt

from model import ResNet50Classifier
from lime_explainer import LIMEExplainer


# --------------------------------------------------
# Load model
# --------------------------------------------------

classifier = ResNet50Classifier()


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = Image.open(
    "test_image.jpg"
).convert("RGB")


# --------------------------------------------------
# Create LIME explainer
# --------------------------------------------------

explainer = LIMEExplainer(
    classifier
)


# --------------------------------------------------
# Generate explanation
# --------------------------------------------------

print(
    "Generating LIME explanation..."
)

print(
    "This may take a little while..."
)

explanation = explainer.explain(
    image,
    num_samples=1000
)


# --------------------------------------------------
# Get predicted class
# --------------------------------------------------

top_label = explanation.top_labels[0]


# --------------------------------------------------
# Positive + negative explanation
# --------------------------------------------------

positive_image, positive_mask = (
    explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
)


both_image, both_mask = (
    explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
)


# --------------------------------------------------
# Display
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(image)

axes[0].set_title(
    "Original Image"
)

axes[0].axis("off")


axes[1].imshow(positive_image)

axes[1].set_title(
    "LIME - Positive Regions"
)

axes[1].axis("off")


axes[2].imshow(both_image)

axes[2].set_title(
    "LIME - Positive + Negative"
)

axes[2].axis("off")


plt.tight_layout()

plt.show()