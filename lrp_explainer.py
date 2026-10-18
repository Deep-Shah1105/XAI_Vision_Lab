import torch

from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer
from zennit.attribution import Gradient


class LRPExplainer:

    def __init__(self, classifier):

        self.classifier = classifier

        self.model = classifier.model

        self.device = classifier.device

        # --------------------------------------------------
        # ResNet-specific canonizer
        #
        # This is the important part.
        #
        # It handles:
        #   - BatchNorm
        #   - residual connections
        #   - skip connections
        # --------------------------------------------------

        self.canonizer = ResNetCanonizer()

        # --------------------------------------------------
        # EpsilonPlusFlat
        #
        # This provides the LRP propagation rules.
        # --------------------------------------------------

        self.composite = EpsilonPlusFlat(
            canonizers=[
                self.canonizer
            ]
        )

    def explain(
        self,
        input_tensor,
        target_class
    ):

        """
        Generate an LRP attribution for the
        specified target class.
        """

        # LRP requires gradients
        input_tensor = input_tensor.clone().detach()

        input_tensor.requires_grad_()

        # --------------------------------------------------
        # Create one-hot target vector
        # --------------------------------------------------

        target = torch.zeros(
            (
                input_tensor.shape[0],
                1000
            ),
            device=self.device
        )

        target[
            0,
            target_class
        ] = 1.0

        # --------------------------------------------------
        # Run Zennit's propagation
        # --------------------------------------------------

        with Gradient(
            model=self.model,
            composite=self.composite
        ) as attributor:

            output, relevance = attributor(
                input_tensor,
                target
            )

        return output, relevance