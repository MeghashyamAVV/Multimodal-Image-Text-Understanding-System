from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

#BLIP model (for image captioning)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.eval()

def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

if __name__ == "__main__":
    caption = generate_caption("data/images/dog.jpg")
    print("Generated caption:", caption)