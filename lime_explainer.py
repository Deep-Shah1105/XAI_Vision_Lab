import numpy as np
import torch

from PIL import Image
from lime import lime_image

class LIMEExplainer:

    def __init__(self, classifier):

        self.classifier = classifier

        self.model = classifier.model

        self.device = classifier.device

        self.preprocess = classifier.preprocess

        self.explainer = lime_image.LimeImageExplainer()

    def predict(self, images):

        """
        LIME provides images as NumPy arrays.

        Convert each NumPy array to a PIL Image before
        passing it through torchvision preprocessing.
        """

        batch = []

        for image in images:

         # NumPy array → PIL Image
            image = Image.fromarray(
                image.astype(np.uint8)
            )

            # PIL Image → Tensor
            tensor = self.preprocess(image)

            batch.append(tensor)

        # Create batch
        batch = torch.stack(batch)

        # Move to GPU
        batch = batch.to(self.device)

        # Inference mode
        self.model.eval()

        with torch.no_grad():

            outputs = self.model(batch)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        # LIME expects NumPy probabilities
        return probabilities.cpu().numpy()

    def explain(
        self,
        image,
        num_samples=1000
    ):

        """
        Generate a LIME explanation.
        """

        image_array = np.array(image)

        explanation = self.explainer.explain_instance(

            image_array,

            self.predict,

            top_labels=5,

            hide_color=0,

            num_samples=num_samples
        )

        return explanation