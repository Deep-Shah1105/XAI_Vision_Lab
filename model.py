import torch
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Classifier:

    def __init__(self):

        # Select GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        # Load pretrained ResNet-50
        weights = ResNet50_Weights.DEFAULT

        self.model = resnet50(weights=weights)

        # Evaluation mode
        self.model.eval()

        # Move model to GPU
        self.model.to(self.device)

        # Image preprocessing provided by torchvision
        self.preprocess = weights.transforms()

        # ImageNet class labels
        self.categories = weights.meta["categories"]

    def predict(self, image):

        # Preprocess image
        input_tensor = self.preprocess(image)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # Move image to GPU
        input_tensor = input_tensor.to(self.device)

        # Disable gradient calculation for normal prediction
        with torch.no_grad():

            outputs = self.model(input_tensor)

            probabilities = torch.nn.functional.softmax(
                outputs,
                dim=1
            )

        # Get top 5 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities,
            5
        )

        results = []

        for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
        ):

            results.append({
                "class": self.categories[index.item()],
                "probability": probability.item()
            })

        return results