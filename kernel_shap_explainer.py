import numpy as np
import torch
import shap

from PIL import Image
from skimage.segmentation import slic


class KernelSHAPExplainer:

    def __init__(
        self,
        classifier,
        n_segments=75,
        compactness=10
    ):

        self.classifier = classifier

        self.model = classifier.model

        self.device = classifier.device

        self.preprocess = classifier.preprocess

        self.n_segments = n_segments

        self.compactness = compactness

        self.image = None
        self.segments = None
        self.baseline = None
        self.target_class = None

    # ==================================================
    # RESNET PREDICTION
    # ==================================================

    def predict_images(self, images):

        """
        Convert a batch of RGB NumPy images into
        tensors and obtain ImageNet probabilities.
        """

        batch = []

        for image in images:

            # Make sure pixel values are valid
            image = np.clip(
                image,
                0,
                255
            ).astype(np.uint8)

            # NumPy → PIL
            image = Image.fromarray(
                image
            )

            # PIL → Tensor
            tensor = self.preprocess(
                image
            )

            batch.append(tensor)

        # Stack into batch
        batch = torch.stack(
            batch
        )

        # Move to GPU
        batch = batch.to(
            self.device
        )

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(
                batch
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        return probabilities.cpu().numpy()

    # ==================================================
    # CREATE SUPERPIXELS
    # ==================================================

    def create_superpixels(self, image):

        """
        Divide the image into visually meaningful
        regions using SLIC.
        """

        image_array = np.array(
            image
        ).astype(np.float32) / 255.0

        segments = slic(
            image_array,

            n_segments=self.n_segments,

            compactness=self.compactness,

            sigma=1,

            start_label=0
        )

        return segments

    # ==================================================
    # CREATE BASELINE
    # ==================================================

    def create_baseline(self, image):

        """
        Create a blurred baseline image.

        A blurred baseline is preferable to black
        because black regions can introduce an
        artificial feature into ImageNet models.
        """

        image_array = np.array(
            image
        ).astype(np.float32)

        # Convert to tensor
        image_tensor = torch.from_numpy(
            image_array
        ).permute(
            2,
            0,
            1
        ).unsqueeze(0)

        # Average pooling creates a blurred image
        baseline = torch.nn.functional.avg_pool2d(
            image_tensor,
            kernel_size=31,
            stride=1,
            padding=15
        )

        baseline = baseline.squeeze(
            0
        ).permute(
            1,
            2,
            0
        ).numpy()

        return baseline

    # ==================================================
    # APPLY SUPERPIXEL MASK
    # ==================================================

    def apply_mask(self, mask):

        """
        mask = 1:
            Keep original superpixel.

        mask = 0:
            Replace superpixel with baseline.
        """

        result = self.baseline.copy()

        number_of_segments = (
            int(self.segments.max()) + 1
        )

        for segment_id in range(
            number_of_segments
        ):

            if mask[segment_id] == 1:

                result[
                    self.segments == segment_id
                ] = self.image[
                    self.segments == segment_id
                ]

        return result

    # ==================================================
    # KERNELSHAP MODEL FUNCTION
    # ==================================================

    def shap_model(self, masks):

        """
        KernelSHAP sends binary feature masks.

        Each mask represents one combination of
        superpixels being present/absent.

        IMPORTANT:
        We return ONLY the probability of the
        target class.

        Therefore KernelSHAP explains:

            "Why did ResNet predict THIS class?"

        rather than explaining all 1000 classes.
        """

        images = []

        for mask in masks:

            image = self.apply_mask(
                mask
            )

            images.append(
                image
            )

        # Get predictions from ResNet
        probabilities = self.predict_images(
            images
        )

        # Extract ONLY target-class probability
        target_probabilities = probabilities[
            :,
            self.target_class
        ]

        # SHAP expects:
        #
        # number_of_samples × number_of_outputs
        #
        return target_probabilities.reshape(
            -1,
            1
        )

    # ==================================================
    # MAIN EXPLANATION
    # ==================================================

    def explain(
        self,
        image,
        nsamples=500
    ):

        # ------------------------------------------------
        # Store image
        # ------------------------------------------------

        self.image = np.array(
            image
        ).astype(np.uint8)

        # ------------------------------------------------
        # Generate superpixels
        # ------------------------------------------------

        self.segments = self.create_superpixels(
            image
        )

        # ------------------------------------------------
        # Create baseline
        # ------------------------------------------------

        self.baseline = self.create_baseline(
            image
        )

        # ------------------------------------------------
        # Get original prediction
        # ------------------------------------------------

        probabilities = self.predict_images(
            [self.image]
        )

        # Highest probability class
        self.target_class = int(
            np.argmax(
                probabilities[0]
            )
        )

        class_name = self.classifier.categories[
            self.target_class
        ]

        confidence = probabilities[
            0,
            self.target_class
        ]

        # ------------------------------------------------
        # Determine actual number of superpixels
        # ------------------------------------------------

        number_of_segments = (
            int(self.segments.max()) + 1
        )

        print()
        print(
            f"Actual superpixels: "
            f"{number_of_segments}"
        )

        print(
            f"Target class: "
            f"{class_name}"
        )

        print(
            f"Target probability: "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"KernelSHAP samples: "
            f"{nsamples}"
        )

        # ------------------------------------------------
        # IMPORTANT:
        # Create the KernelSHAP background
        # ------------------------------------------------

        background = np.zeros(
            (
                1,
                number_of_segments
            ),
            dtype=np.float32
        )

        # ------------------------------------------------
        # Create KernelSHAP explainer
        # ------------------------------------------------

        explainer = shap.KernelExplainer(
            self.shap_model,

            background,

            link="identity"
        )

        # ------------------------------------------------
        # Explain the ORIGINAL image
        #
        # All ones means:
        #
        # superpixel 0 = present
        # superpixel 1 = present
        # ...
        # ------------------------------------------------

        full_image_mask = np.ones(
            (
                1,
                number_of_segments
            ),
            dtype=np.float32
        )

        print()
        print(
            "Generating KernelSHAP explanation..."
        )

        print(
            "This may take a while..."
        )

        shap_values = explainer.shap_values(
            full_image_mask,

            nsamples=nsamples
        )

        return (
            shap_values,
            probabilities[0],
            self.target_class
        )