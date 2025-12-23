from sklearn.metrics import accuracy_score
from multimodal_pipeline import predict

# Example
test_data = [
    ("data/images/dog.jpg", ["a dog", "a cat"], "a dog"),
    ("data/images/cat.jpg", ["a dog", "a cat"], "a cat"),
]

y_true = []
y_pred = []

for img, labels, gt in test_data:
    probs = predict(img, labels)
    pred_label = labels[probs.argmax().item()]

    y_true.append(gt)
    y_pred.append(pred_label)

acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")