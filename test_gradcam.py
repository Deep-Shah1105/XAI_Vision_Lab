from PIL import Image
import matplotlib.pyplot as plt
import torch

from model import ResNet50Classifier
from gradcam import GradCAM, overlay_heatmap


# --------------------------------------------------
# Load model
# --------------------------------------------------

classifier = ResNet50Classifier()

model = classifier.model
device = classifier.device


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = Image.open(
    "test_image.jpg"
).convert("RGB")


# --------------------------------------------------
# Preprocess
# --------------------------------------------------

input_tensor = classifier.preprocess(
    image
).unsqueeze(0).to(device)


# --------------------------------------------------
# ResNet-50 target layers
#
# We use the final convolution of each
# major ResNet stage.
# --------------------------------------------------

target_layers = {
    "Layer 1": model.layer1[-1].conv3,
    "Layer 2": model.layer2[-1].conv3,
    "Layer 3": model.layer3[-1].conv3,
    "Layer 4": model.layer4[-1].conv3,
}


# --------------------------------------------------
# Create figure
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


# --------------------------------------------------
# Original image
# --------------------------------------------------

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")


# --------------------------------------------------
# Generate prediction first
# --------------------------------------------------

model.eval()

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(
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
print("Prediction")
print("==============================")
print(
    f"{class_name}: "
    f"{confidence * 100:.2f}%"
)


# --------------------------------------------------
# Generate Grad-CAM for every stage
# --------------------------------------------------

for index, (layer_name, target_layer) in enumerate(
    target_layers.items()
):

    print(
        f"Generating Grad-CAM for {layer_name}..."
    )

    # Create Grad-CAM object
    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
        device=device
    )

    # Generate CAM specifically for
    # the predicted class
    cam, _ = gradcam.generate(
        input_tensor,
        target_class=target_class
    )

    # Overlay CAM on original image
    result = overlay_heatmap(
        image,
        cam
    )

    # Map stage index to subplot
    if index < 2:
        row = 0
        col = index + 1
    else:
        row = 1
        col = index - 2

    axes[row, col].imshow(result)

    axes[row, col].set_title(
        f"{layer_name}\n"
        f"{class_name}"
    )

    axes[row, col].axis("off")

    # Remove hooks
    gradcam.remove_hooks()


# --------------------------------------------------
# Empty final subplot
# --------------------------------------------------

axes[1, 2].axis("off")


# --------------------------------------------------
# Overall title
# --------------------------------------------------

fig.suptitle(
    "Grad-CAM Across ResNet-50",
    fontsize=18
)

plt.tight_layout()

plt.show()