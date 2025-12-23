# Multimodal Image-Text Understanding System

Built a multimodal AI system that jointly understands images and text to perform classification and explanation tasks using vision–language models.

## Key Highlights

- Image–text understanding using pre-trained vision–language models  
- Supports image classification using text prompts  
- Generates natural language explanations for images  
- Interactive terminal-based inference  
- Demonstrates cross-modal reasoning capabilities  

## Tech Stack

- Python  
- PyTorch  
- Hugging Face Transformers  
- CLIP (Vision–Language Model)  
- BLIP (Image Captioning)  
- Vision Transformers (ViT)  

## How It Works

1. Images are processed using a Vision Transformer encoder  
2. Text prompts are encoded using a transformer-based text encoder  
3. Image and text embeddings are aligned in a shared embedding space  
4. CLIP performs image–text matching for classification  
5. BLIP generates natural language captions as explanations  

## Install & Run

# Install dependencies
```bash
pip install torch torchvision transformers pillow sentence-transformers scikit-learn
```

# Run the interactive demo
```bash
python run.py
```
Provide an image path and text prompts in the terminal

## Evaluation

The system is evaluated using basic classification accuracy and confidence scores based on image–text similarity.

## Summary

1. Demonstrates applied multimodal learning
2. Shows understanding of vision–language models
3. Combines classification and explanation in one pipeline
4. Reflects real-world multimodal AI systems used in practice
