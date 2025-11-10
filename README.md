# 🧠 Sentiment Analysis for Mental Health Detection

This repository contains the code and documentation for a **BERT-based model** fine-tuned to classify text into four mental health categories:

* Anxiety
* Depression
* Normal
* Suicidal Ideation

---

## 🚀 Live Demo & Model

This project is deployed and accessible via Hugging Face. The live frontend for the demo is hosted separately — see the **Live Site** and **Frontend repository** links below.

* **Live Site (Production):** [https://mentalhealthsurvey.vercel.app/](https://mentalhealthsurvey.vercel.app/)

* **Frontend (Source Code):** [https://github.com/auraflaa/Sentiment-Analysis-For-Mental-Health-Detection-Frontend](https://github.com/auraflaa/Sentiment-Analysis-For-Mental-Health-Detection-Frontend)

* **Resources & Assets (Google Drive):** [https://drive.google.com/drive/folders/1cmoBkGWXl0z6FBM6VODNSI8-LNJbNGdB?usp=sharing](https://drive.google.com/drive/folders/1cmoBkGWXl0z6FBM6VODNSI8-LNJbNGdB?usp=sharing)

* **Live Demo (Gradio):** Try the model in your browser
  [https://huggingface.co/spaces/ourafla/Mental-Health-Detection/tree/main](https://huggingface.co/spaces/ourafla/Mental-Health-Detection/tree/main)

* **Hugging Face Model:** View the model card & weights
  [https://huggingface.co/ourafla/mental-health-bert-finetuned](https://huggingface.co/ourafla/mental-health-bert-finetuned)

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

The model was fine-tuned on a **custom-curated English dataset from multiple sources**.

### Dataset Split

* Training: ~49,382 samples
* Validation: ~5,487 samples
* Test: 992 samples (balanced: 248 per class)

### Hyperparameters

| Parameter           | Value                             |
| ------------------- | --------------------------------- |
| Base Model          | `mental/mental-bert-base-uncased` |
| Batch Size          | 16                                |
| Learning Rate       | 2e-5                              |
| Epochs              | 5                                 |
| Max Sequence Length | 128                               |
| Optimizer           | AdamW with linear warmup          |
| Precision           | fp16 (mixed)                      |

---

## 📊 Evaluation Results

The model was evaluated on a **balanced test set of 992 samples** (248 per class).

### Summary Metrics

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 89.72% |
| Macro F1        | 89.54% |
| Macro Precision | 89.56% |
| Macro Recall    | 89.72% |

### Per-Class Performance

* **Normal:** 96% F1
* **Suicidal:** 94% F1
* **Anxiety:** 87% F1
* **Depression:** 82% F1

📌 *Confusion is highest between Anxiety and Depression due to linguistic overlap.*

---

## 🐳 Deployment

This model can be easily containerized and deployed as a web service.

```bash
# Build the Docker image
docker build -t mental-health-api .

# Run the container
docker run -p 8080:8080 mental-health-api
```

Once running, open:
**[http://localhost:8080](http://localhost:8080)**

(Assuming you configured a `/health` endpoint)

---

## 🌎 Environmental Impact

| Resource          | Details                  |
| ----------------- | ------------------------ |
| GPU               | 1 × NVIDIA T4 (16GB)     |
| Training Duration | ~1.3 hours (~80 minutes) |
| Cloud Provider    | Google Cloud (Colab)     |

---

## 📜 Citation

If you use this model in your work, please cite:

```bibtex
@software{mental_health_classifier_2025,
  author = {Mukherjee, Priyangshu},
  title = {Mental Health Text Classifier (MentalBERT Fine-tuned)},
  year = {2025},
  note = {Fine-tuned model hosted on Hugging Face as ourafla/mental-health-bert-finetuned},
  howpublished = {\url{https://huggingface.co/ourafla/mental-health-bert-finetuned}}
}
```

And the base model:

```bibtex
@inproceedings{ji2022mentalbert,
  title = {{MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare}},
  author = {Shaoxiong Ji and Tianlin Zhang and Luna Ansari and Jie Fu and Prayag Tiwari and Erik Cambria},
  year = {2022},
  booktitle = {Proceedings of LREC}
}
```

---

## 👤 Contact

**Author:** Priyangshu Mukherjee
**Hugging Face:** [https://huggingface.co/ourafla](https://huggingface.co/ourafla)
**GitHub:** [https://github.com/auraflaa](https://github.com/auraflaa)
