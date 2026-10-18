from PIL import Image
from model import ResNet50Classifier


# Load model
classifier = ResNet50Classifier()

# Load image
image = Image.open("test_image.jpg").convert("RGB")

# Predict
results = classifier.predict(image)

print("\nTop 5 Predictions")
print("=================")

for result in results:

    print(
        f"{result['class']:30s} "
        f"{result['probability'] * 100:.2f}%"
    )