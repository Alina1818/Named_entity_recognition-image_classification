import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ITALIAN_TO_ENGLISH = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant",
    "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat",
    "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel",
    "ragno": "spider"
}
ANIMAL_NAMES = set(ITALIAN_TO_ENGLISH.values())

def main():
    model_dir = "/content/drive/MyDrive/ner_model"

    test_sentences = [
        ["The", "dog", "is", "running", "in", "the", "garden"],
        ["A", "cat", "sits", "on", "the", "roof"],
        ["She", "feeds", "the", "cow", "and", "the", "sheep"],
        ["A", "gatto", "slept", "on", "the", "sofa"],
        ["The", "pecora", "is", "white"],
        ["A", "cane", "barked", "loudly"],
        ["The", "cavallo", "was", "running"],
        ["Elephants", "and", "dogs", "are", "different", "sizes"],
        ["Butterflies", "like", "flowers"],
        ["The", "spider", "spins", "a", "web"],
    ]

    test_labels = [
        ["O", "B-ANIMAL", "O", "O", "O", "O", "O"],
        ["O", "B-ANIMAL", "O", "O", "O", "O"],
        ["O", "O", "O", "B-ANIMAL", "O", "O", "B-ANIMAL"],
        ["O", "B-ANIMAL", "O", "O", "O", "O"],
        ["O", "B-ANIMAL", "O", "O"],
        ["O", "B-ANIMAL", "O", "O"],
        ["O", "B-ANIMAL", "O", "O"],
        ["B-ANIMAL", "O", "B-ANIMAL", "O", "O", "O", "O"],
        ["B-ANIMAL", "O", "O"],
        ["O", "B-ANIMAL", "O", "O", "O"]
    ]

    label_to_id = {"O": 0, "B-ANIMAL": 1}

    # --- Load model and tokenizer ---
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def normalize(tok):
        t = tok.lower()
        t = ITALIAN_TO_ENGLISH.get(t, t)
        if t.endswith("s") and t[:-1] in ANIMAL_NAMES:
            return t[:-1]
        return t

    all_preds, all_labels = [], []

    for tokens, labels in zip(test_sentences, test_labels):
        tokens_translated = [normalize(tok) for tok in tokens]

        inputs = tokenizer(
            tokens_translated,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

        word_ids = inputs.word_ids(batch_index=0)
        filtered_preds, filtered_labels = [], []
        prev_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                filtered_preds.append(preds[idx])
                if word_idx < len(labels):
                    filtered_labels.append(label_to_id[labels[word_idx]])
                prev_word_idx = word_idx

        all_preds.extend(filtered_preds)
        all_labels.extend(filtered_labels)

        decoded_pred = ["B-ANIMAL" if p == 1 else "O" for p in filtered_preds]


    # --- Compute metrics ---
    if all_labels:
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        print(f"Final Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
    else:
        print("No valid tokens for evaluation.")

if __name__ == "__main__":
    main()
