import torch
from torchvision import transforms, models
from PIL import Image

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']

def predict_image(model, class_names, img_path):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(x)
        label = preds.argmax(1).item()
    return class_names[label]

if __name__ == "__main__":
    model_path = "/content/drive/MyDrive/saved_models/best_model.pth"
    model, class_names = load_model(model_path)
    image_path = "/content/drive/MyDrive/raw-img/sheep/eb3db40a21f4043ed1584d05fb1d4e9fe777ead218ac104497f5c978a6e8b0b1_640.jpg"

    result = predict_image(model, class_names, image_path)
    print(f"Predicted class: {result}")
