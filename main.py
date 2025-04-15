# Install Ultralytics (YOLOv8)
# !pip install ultralytics
# !pip install open_clip_torch
from ultralytics import YOLO

# Load YOLOv8 pretrained model (you can also use a custom one)
model = YOLO("yoloe-11l-seg.pt")  # yolov8n = nano, yolov8m = medium, yolov8x = large

from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt
import open_clip
import torch

uploaded = files.upload()
image_path = list(uploaded.keys())[0]
device = "cuda" if torch.cuda.is_available() else "cpu"

# Run object detection
names = ["cabinets", "refrigerator", "ceiling", "floor", "oven", "roof", "dishwasher", "sink", "wall", "window", "door", "countertop"]
model.set_classes(names, model.get_text_pe(names))
results = model(image_path)

# Display results
results[0].show()

# Get list of detected object names
detected_items = list({model.names[int(cls)] for cls in results[0].boxes.cls})
print("Detected items:", detected_items)
image = Image.open(image_path).convert("RGB")
boxes = results[0].boxes.xyxy.cpu().numpy()
classes = results[0].boxes.cls.cpu().numpy()
names1 = model.names

# clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')
clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')

def assess_renovation(pil_crop, label):
    text_prompts = [
        f"a renovated {label}",
        f"a worn-out {label}"
    ]
    text_tokens = tokenizer(text_prompts).to(device)

    image_input = preprocess(pil_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0).softmax(dim=0).cpu().numpy()

    return text_prompts, similarity

print("\n Object Renovation Assessment:")

for box, cls in zip(boxes, classes):
    label = names1[int(cls)]
    x1, y1, x2, y2 = map(int, box)
    cropped = image.crop((x1, y1, x2, y2))
    
    prompts, scores = assess_renovation(cropped, label)
    
    print(f"\n {label.upper()} (box: {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    for prompt, score in zip(prompts, scores):
        print(f"   {prompt}: {score:.2f}")

    if scores[1]  > scores[0]:
        print("Suggest renovation")
    else:
        print("Looks updated")
