import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

#CLIP model
MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

def predict(image_path, text_candidates):
    """
    image_path: path to image
    text_candidates: list of text descriptions
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=text_candidates,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)

    return probs