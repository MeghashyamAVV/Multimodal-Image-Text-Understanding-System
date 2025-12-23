from multimodal_pipeline import predict
from captioning import generate_caption

image_path = input("Enter image path: ")

text_input = input(
    "Enter comma-separated text options: "
)
labels = [t.strip() for t in text_input.split(",")]

probs = predict(image_path, labels)
best_idx = probs.argmax().item()

print("\nPrediction:", labels[best_idx])
print("Confidence:", probs[0][best_idx].item())

caption = generate_caption(image_path)
print("Image Explanation:", caption)
