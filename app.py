import os
import torch
import timm
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 🔹 Class Names
class_names = ['Bacterialblight', 'Brownspot', 'Leafsmut']

# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load ViT Model
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, 3)
model.load_state_dict(torch.load("vit_leaf_model.pth", map_location=device))
model.to(device)
model.eval()

# 🔹 Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 🔹 Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_path = filepath

            # Open Image
            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # Prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)
