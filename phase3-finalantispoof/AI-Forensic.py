# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("image-classification", model="ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL")
# r=pipe("L120.png")
# print("resss",r)



# Load model directly
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# processor = AutoImageProcessor.from_pretrained("ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL")
# model = AutoModelForImageClassification.from_pretrained("ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL")



from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch



from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = "ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL"

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# # Load model and feature extractor
# model_name = "ideepankarsharma2003/AI_ImageClassification_MidjourneyV6_SDXL"
# model = AutoModelForImageClassification.from_pretrained(model_name)
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
# inputs = image_processor(images=image, return_tensors="pt")

# # Load and preprocess image
image = Image.open("L120.png")
inputs = image_processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(-1).item()

# Label Mapping
id2label = {0: "ai_gen", 1: "human"}
print("Predicted label:", id2label[predicted_label])
