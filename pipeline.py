import os
from typing import List, Set, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import numpy as np

IMAGE_MODEL_PATH = "/content/drive/MyDrive/saved_models/best_model.pth"
NER_MODEL_DIR = "/content/drive/MyDrive/ner_model"
DATA_DIR = "/content/drive/MyDrive/raw-img"

TRANSLATE = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "scoiattolo": "squirrel", "ragno": "spider"
}

# ---------------------------
#  Image model: load & predict
# ---------------------------

def load_image_model(model_path: str, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Image model not found: {model_path}")

    chk = torch.load(model_path, map_location=device)
    class_names = list(chk.get("class_names", []))
    if not class_names:
        raise ValueError("No class_names found in checkpoint")

    # init architecture same as training
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(chk["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return model, class_names, transform, device

def predict_image_label(model, class_names: List[str], transform, device, image_path: str) -> Tuple[str, float]:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    pred_raw = class_names[top_idx]
    pred_norm = TRANSLATE.get(pred_raw.lower(), pred_raw.lower())
    return pred_norm, top_prob

# ---------------------------
#  NER model: load & predict (extract animal tokens)
# ---------------------------

def load_ner_model(ner_dir: str):
    if not os.path.isdir(ner_dir):
        raise FileNotFoundError(f"NER model directory not found: {ner_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(ner_dir)
    model = DistilBertForTokenClassification.from_pretrained(ner_dir)
    model.eval()
    return tokenizer, model

def extract_animals_from_text(tokenizer, model, text: str) -> Set[str]:
    inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1, seq_len, num_labels)
        preds = torch.argmax(logits, dim=2)[0].cpu().numpy()  # shape (seq_len,)
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    if id2label is None:
        id2label = {0: "O", 1: "B-ANIMAL"}

    word_ids = inputs.word_ids(batch_index=0)  # mapping token -> word index
    found = []
    prev_w = None
    for token_idx, widx in enumerate(word_ids):
        if widx is None:
            continue
        if widx != prev_w:
            label_id = int(preds[token_idx])
            label = id2label.get(label_id, str(label_id))
            if "ANIMAL" in label or label.upper().endswith("ANIMAL"):
                word = text.split()[widx]
                found.append(word)
            prev_w = widx
    normalized = set()
    for w in found:
        w_low = w.lower()
        norm = TRANSLATE.get(w_low, w_low)
        normalized.add(norm)
    return normalized

# ---------------------------
#  Pipeline logic: compare image label and NER
# ---------------------------

def pipeline_decide(text: str, image_path: str,
                    image_model, image_class_names, image_tf, device,
                    ner_tokenizer, ner_model,
                    image_conf_threshold: float = 0.0) -> Tuple[bool, dict]:

    # 1) image predict
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_pred, image_conf = predict_image_label(image_model, image_class_names, image_tf, device, image_path)

    # 2) NER extract animals
    extracted = extract_animals_from_text(ner_tokenizer, ner_model, text)

    # 3) decision: if any extracted animal equals image_pred
    match = False
    if image_conf < image_conf_threshold:
        # optional: if confidence too low, treat as False (or change policy)
        match = False
    else:
        if image_pred in extracted:
            match = True
        else:
            # sometimes NER finds nothing but text contains the animal name untagged -> fallback match by substring
            for w in text.split():
                if TRANSLATE.get(w.lower(), w.lower()) == image_pred:
                    match = True
                    break

    details = {
        "image_pred": image_pred,
        "image_conf": image_conf,
        "ner_extracted": list(extracted),
        "match": match
    }
    return match, details

# ---------------------------
#  Utilities: batch test on whole dataset (image-only)
# ---------------------------

def evaluate_image_model_on_dataset(image_model, class_names, transform, device, data_root: str, batch_size: int = 32):
    ds = datasets.ImageFolder(data_root, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0
    image_model.eval()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            outputs = image_model(x)
            preds = outputs.argmax(1).cpu().numpy()
            correct += (preds == y.numpy()).sum()
            total += y.size(0)
    return correct / total if total > 0 else 0.0

# ---------------------------
#  MAIN (demo)
# ---------------------------

def main_demo(text: str = None, image: str = None):
    image_model, image_class_names, image_tf, device = load_image_model(IMAGE_MODEL_PATH)
    ner_tokenizer, ner_model = load_ner_model(NER_MODEL_DIR)

    if image is None:
        for cls in image_class_names:
            cls_dir = os.path.join(DATA_DIR, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        image = os.path.join(cls_dir, f)
                        break
            if image:
                break
        if image is None:
            print("No example image found inside DATA_DIR, please provide image path.")
            return None

    if text is None:
        print("Please provide text input.")
        return None

    match, details = pipeline_decide(
        text=text,
        image_path=image,
        image_model=image_model,
        image_class_names=image_class_names,
        image_tf=image_tf,
        device=device,
        ner_tokenizer=ner_tokenizer,
        ner_model=ner_model
    )

    # Result
    print("\nImage:", image)
    print("Text:", text)
    print("Details:", details)
    print("Match:", match)
    return match

if __name__ == "__main__":
    main_demo(
        text="a horse running",
        image="/content/drive/MyDrive/raw-img/horse/OIP-z1W6sgECKNKjQl6jZJWpXgHaFj.jpeg"
    )
