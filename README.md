# 🧠 Sentiment Analysis for Mental Health Detection

This repository contains the code and documentation for a **BERT-based model** fine-tuned to classify text into four mental health categories:
- Anxiety  
- Depression  
- Normal  
- Suicidal Ideation  

---

## 🚀 Live Demo & Model
This project is deployed and accessible via **Hugging Face**.

- **Live Demo (Gradio):** Try the model in your browser  
- **Hugging Face Model:** View the model card & weights  

---

## 🛑 Disclaimer
This model is for **educational and research purposes only**.  
It is **not a substitute** for professional medical or psychological advice, diagnosis, or treatment.

If you or someone you know is in crisis, please contact local mental health helplines or emergency services immediately.  
This model is a tool, **not a medical professional**, and its predictions are **not a diagnosis**.

---

## 💻 Example Usage

You can easily load and use this model directly from the **Hugging Face Hub**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "ourafla/mental-health-bert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define labels
labels = ["Anxiety", "Depression", "Normal", "Suicidal"]

# Run inference
text = "I've been feeling really anxious and tired lately."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# Get probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Print the results
print(f"Text: '{text}'")
print("--- Predictions ---")
print({label: round(float(prob), 4) for label, prob in zip(labels, probs[0])})
```

---

## ⚙️ Model Training

The model was fine-tuned on a **custom-curated English dataset** of social posts and anonymized text.

| Parameter | Value |
|------------|--------|
| Dataset Split | 80% Train / 10% Validation / 10% Test |
| Base Model | `bert-base-uncased` |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 4 |
| Max Sequence Length | 128 |
| Optimizer | AdamW with LR-scheduler |
| Precision | fp16 (mixed) |

---

## 📊 Evaluation Results

The model achieves **balanced performance** across all classes, with a slight overlap observed between *Anxiety* and *Depression*.

| Metric | Score |
|---------|--------|
| Accuracy | 0.93 |
| Macro F1 | 0.91 |
| Precision | 0.92 |
| Recall | 0.90 |

---

## 🐳 Deployment

This model can be easily containerized and deployed as a web service.

```bash
# 1. Build the Docker image
docker build -t mental-health-api .

# 2. Run the container
docker run -p 8080:8080 mental-health-api
```

Once running, visit [http://localhost:8080](http://localhost:8080) to verify the API health (assuming a health-check endpoint is configured in your `main.py`/`app.py`).

---

## 🌎 Environmental Impact

| Resource | Detail |
|-----------|---------|
| GPU | 1 × NVIDIA T4 (16 GB) |
| Training Duration | ~1 hour |
| Cloud Provider | Google Cloud |
| CO₂ Emissions | ≈ 60 g CO₂eq |

---

## 📜 Citation

If you use this model in your work, please cite it:

```bibtex
@misc{ourafla2025mentalbert,
  author = {Ourafla},
  title  = {Mental Health BERT Fine-Tuned Classifier},
  year   = {2025},
  howpublished = {\url{https://huggingface.co/ourafla/mental-health-bert-finetuned}}
}
```

---

## 👤 Contact

**Author:** Ourafla  
**Hugging Face:** [huggingface.co/ourafla](https://huggingface.co/ourafla)  
**GitHub:** [github.com/auraflaa](https://github.com/auraflaa)
