# 🧠 Sentiment Analysis for Mental Health Detection

This repository contains the code and documentation for a **BERT-based model** fine-tuned to classify text into four mental health categories:

- Anxiety  
- Depression  
- Normal  
- Suicidal Ideation  

The project demonstrates **end-to-end ML ownership**: dataset curation, model training, evaluation, deployment, and public distribution across **Hugging Face and Kaggle**.

---

## 🚀 Live Demo, Models & Datasets

### 🔗 Live Applications

- **Production Web App:**  
  https://mentalhealthsurvey.vercel.app/

- **Frontend Source Code:**  
  https://github.com/auraflaa/Sentiment-Analysis-For-Mental-Health-Detection-Frontend

- **Interactive Demo (Gradio on Hugging Face Spaces):**  
  https://huggingface.co/spaces/ourafla/Mental-Health-Detection/tree/main

---

### 🤖 Model Artifacts

The fine-tuned model is published on **both Hugging Face and Kaggle** to maximize accessibility and reproducibility.

- **Hugging Face Model (Primary):**  
  https://huggingface.co/ourafla/mental-health-bert-finetuned  

- **Kaggle Model:**  
  https://www.kaggle.com/models/priyangshumukherjee/mental-health-bert-fine-tunes  

Both model versions contain the same trained weights and label configuration.

---

### 📂 Dataset Artifacts

The training data is a **custom-curated, multi-source English mental health dataset**, published on both platforms.

- **Hugging Face Dataset (Primary):**  
  https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset  

- **Kaggle Dataset:**  
  https://www.kaggle.com/datasets/priyangshumukherjee/mental-health-text-classification-dataset  

The dataset includes cleaned text samples with four target classes and fixed train/validation/test splits.

---

### 📁 Resources & Assets

- **Google Drive (reports, visuals, auxiliary files):**  
  https://drive.google.com/drive/folders/1cmoBkGWXl0z6FBM6VODNSI8-LNJbNGdB?usp=sharing  

---

## 🛑 Disclaimer

This project is intended **strictly for educational and research purposes**.

It is **not a medical diagnostic tool** and must not be used as a substitute for professional mental health care.

If you or someone you know is experiencing a mental health crisis, please seek help from qualified professionals or local emergency services.

---

## 💻 Example Usage (Hugging Face Hub)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "ourafla/mental-health-bert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["Anxiety", "Depression", "Normal", "Suicidal"]

text = "I've been feeling really anxious and tired lately."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

probs = torch.nn.functional.softmax(logits, dim=-1)

print({label: round(float(prob), 4) for label, prob in zip(labels, probs[0])})
````

---

## ⚙️ Model Training

### Dataset Composition

* **Training:** ~49,382 samples
* **Validation:** ~5,487 samples
* **Test:** 992 samples (balanced: 248 per class)

### Training Configuration

| Parameter           | Value                             |
| ------------------- | --------------------------------- |
| Base Model          | `mental/mental-bert-base-uncased` |
| Batch Size          | 16                                |
| Learning Rate       | 2e-5                              |
| Epochs              | 5                                 |
| Max Sequence Length | 128                               |
| Optimizer           | AdamW + Linear Warmup             |
| Precision           | FP16 (mixed precision)            |

---

## 📊 Evaluation Results

Evaluated on a **strictly held-out, balanced test set**.

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 89.72% |
| Macro F1        | 89.54% |
| Macro Precision | 89.56% |
| Macro Recall    | 89.72% |

### Per-Class F1 Scores

* **Normal:** 96%
* **Suicidal:** 94%
* **Anxiety:** 87%
* **Depression:** 82%

*Most confusion occurs between Anxiety and Depression due to linguistic overlap.*

---

## 📈 Model Evaluation & Error Analysis

To better understand the model’s behaviour beyond aggregate metrics, an additional evaluation notebook is provided on Kaggle. This analysis focuses on class-wise errors, confusion patterns, and probability calibration, with particular attention to uncertainty in linguistically overlapping categories such as Anxiety and Depression.

The intent of this evaluation is not to claim clinical reliability, but to transparently examine where the model performs well and where it remains limited.

* **Kaggle Evaluation Notebook:**
  [https://www.kaggle.com/code/priyangshumukherjee/mental-health-bert-fine-tuned-evaluation](https://www.kaggle.com/code/priyangshumukherjee/mental-health-bert-fine-tuned-evaluation)

---

## 🐳 Deployment

The model can be containerized and deployed as an API service.

```bash
docker build -t mental-health-api .
docker run -p 8080:8080 mental-health-api
```

---

## 🌎 Environmental Impact

| Resource      | Details              |
| ------------- | -------------------- |
| GPU           | NVIDIA T4 (16GB)     |
| Training Time | ~1.3 hours           |
| Platform      | Google Cloud / Colab |

---

## 📜 Citation

```bibtex
@software{mental_health_classifier_2025,
  author = {Mukherjee, Priyangshu},
  title = {Mental Health Text Classifier (MentalBERT Fine-tuned)},
  year = {2025},
  note = {Model published on Hugging Face and Kaggle},
  howpublished = {\url{https://huggingface.co/ourafla/mental-health-bert-finetuned}}
}
```

Base model:

```bibtex
@inproceedings{ji2022mentalbert,
  title = {MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare},
  author = {Ji, Shaoxiong and Zhang, Tianlin and Ansari, Luna and Fu, Jie and Tiwari, Prayag and Cambria, Erik},
  booktitle = {LREC},
  year = {2022}
}
```

---

## 👤 Author

**Priyangshu Mukherjee**

* Hugging Face: [https://huggingface.co/ourafla](https://huggingface.co/ourafla)
* GitHub: [https://github.com/auraflaa](https://github.com/auraflaa)
* Kaggle: [https://www.kaggle.com/priyangshumukherjee](https://www.kaggle.com/priyangshumukherjee)

