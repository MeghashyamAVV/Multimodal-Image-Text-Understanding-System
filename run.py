from multimodal_pipeline import predict

image_path = input("Enter image path: ")

text_input = input(
    "Enter comma-separated text options (e.g., a dog, a cat, a car): "
)
text_candidates = [t.strip() for t in text_input.split(",")]

probs = predict(image_path, text_candidates)

best_idx = probs.argmax().item()
confidence = probs[0][best_idx].item()

print("\nPrediction:")
print(f"Label: {text_candidates[best_idx]}")
print(f"Confidence: {confidence:.2f}")