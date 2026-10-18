import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model, target_layer, device):

        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):

        # Make sure gradients are enabled
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # If no class is specified, use predicted class
        if target_class is None:

            target_class = output.argmax(dim=1).item()

        # Get score for target class
        target_score = output[:, target_class]

        # Backward pass
        target_score.backward()

        # Get stored activations and gradients
        activations = self.activations
        gradients = self.gradients

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1)

        # ReLU
        cam = F.relu(cam)

        # Remove batch dimension
        cam = cam[0]

        # Convert to numpy
        cam = cam.detach().cpu().numpy()

        # Normalize
        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        return cam, target_class

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_heatmap(
    image,
    cam,
    alpha=0.45
):

    # image is expected to be a PIL image
    image = np.array(image)

    # Convert RGB → BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Resize CAM to image size
    cam = cv2.resize(
        cam,
        (
            image_bgr.shape[1],
            image_bgr.shape[0]
        )
    )

    # Convert CAM to 0-255
    heatmap = np.uint8(
        255 * cam
    )

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Overlay
    result = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    # Convert BGR → RGB
    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    return result