# Multimodal Animal Detection Pipeline

**Description:**
This project implements a multimodal pipeline to verify if the animal mentioned in a text matches the animal in an image. It uses two models:

1. **Image model** – EfficientNet-B0 for animal image classification.
2. **NER model** – DistilBERT for extracting animal names from text.

The pipeline returns `True` if the animal in the text matches the animal in the image, and `False` otherwise.

---


## Possible Improvements

* Softmax threshold for low-confidence ner predictions.
* Support multiple animals in text and image simultaneously.
* Web interface or GUI demo.
* Better normalization of animal names (singular/plural, synonyms).
