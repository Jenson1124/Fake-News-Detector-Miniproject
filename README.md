# 🔍 Fake News Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.48%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning web app that classifies news as **Fake** or **Real** using TF-IDF + Linear SVM, with LIME word-level explanations and a **live news tab** powered by NewsAPI — supporting both 🇮🇳 Indian and 🇺🇸 US sources.

---

## ✨ Features

| Feature | Details |
|---|---|
| **3 input methods** | Paste text · Upload .txt · Scrape URL |
| **📡 Live News tab** | Fetch & analyse real headlines from NewsAPI |
| **🇮🇳 Indian news support** | Top India headlines · NDTV, The Hindu, TOI, India Today |
| **🌐 Global search** | Search any topic in English or Hindi |
| **Country flags** | Each article tagged with source country |
| **LIME explanations** | Word-level highlighting (green = Real, red = Fake) |
| **Model comparison** | LR vs Naive Bayes vs Linear SVM |
| **Export** | Download live news results as CSV |
| **Session history** | Tracks all analyses in current session |

---

## 🚀 Quick Start

```bash
git clone <your-repo>
cd fake-news-detector
pip install -r requirements.txt
python train_model.py        # trains on US dataset (~2 min)
streamlit run app.py
```

---

## 🇮🇳 Adding Indian News Dataset (improves Indian accuracy)

1. Download from Kaggle:  
   https://www.kaggle.com/datasets/vikasukani/fake-news-detection-dataset-in-india

2. Place CSV files in `dataset/indian/`

3. Retrain:
```bash
python train_model.py
# Model automatically combines US + Indian data
```

Check dataset status:
```bash
python download_indian_dataset.py
```

---

## 📡 Live News Tab

- **India headlines** — Top stories from Indian sources via NewsAPI
- **Global headlines** — Top US/international stories  
- **Search any topic** — e.g. "Modi", "election", "economy", "climate"
- Results show confidence bars, probability breakdown, source flags
- Filter by verdict · Sort by confidence/date/source
- Export results as CSV

---

## 🧠 Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| **Linear SVM 🏆** | **99.48%** | **99.45%** |
| Logistic Regression | 98.74% | 98.69% |
| Naive Bayes | 96.04% | 95.84% |

---

## 📁 File Structure

```
fake-news-detector/
├── app.py                      # Streamlit web app
├── train_model.py              # Training pipeline (US + India combined)
├── download_dataset.py         # US dataset helper
├── download_indian_dataset.py  # Indian dataset instructions
├── requirements.txt
├── FakeNewsDetector_EDA.ipynb  # EDA notebook
├── .streamlit/config.toml
├── README.md
├── dataset/
│   ├── True.csv                # Kaggle US real news
│   ├── Fake.csv                # Kaggle US fake news
│   └── indian/                 # Place Indian CSVs here
└── model.pkl / vectorizer.pkl / metrics.json
```

---

## ⚠️ Known Limitations

- Trained on US political news 2016–2017 — works best on that style
- Short headlines (<15 words) give lower confidence — paste full articles
- Analyses **writing style**, not factual accuracy
- Add Indian dataset to improve accuracy on Indian news

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `TF-IDF` · `Linear SVM` · `LIME` · `Streamlit` · `NewsAPI` · `newspaper3k` · `NLTK`

🔗 **Live demo:** 

https://fake-news-detector-miniproject-lz6sy6epkqqvouzqpn4ngn.streamlit.app/

## Author

Jenson Antony Jenianto


