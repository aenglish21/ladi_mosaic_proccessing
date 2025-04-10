from transformers import AutoImageProcessor, AutoModelForImageClassification 
import torch
import requests
from PIL import Image
from io import BytesIO
import os

# Define the image URL and the proxy
image_url = "http://s3.amazonaws.com/fema-cap-imagery/Images/CAP_-_VT_Flooding_Jul_2023/Source/23-1-5464/A0001_AerialOblique/_CAP0347.JPG"
proxy = "http://dbproxy.erau.edu:3128/"

# Set up proxies
proxies = {
    "http": proxy,
    "https": proxy
}

# Download the image using the proxy
response = requests.get(image_url, proxies=proxies)
response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

# Open the image
img = Image.open(BytesIO(response.content))

# Load the model and feature extractor
model_path = "/home/gridsan/groups/CAP_shared/finetuned_models/model_DyG9LBaB/run_20240208-110739_8vZVm0y/epoch_049"
feature_extractor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

# Preprocess the image
inputs = feature_extractor(img, return_tensors="pt")

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted label
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
