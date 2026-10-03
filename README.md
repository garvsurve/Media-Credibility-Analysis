# 🔍 Media Credibility Analysis — Fake News Detection

A machine learning project that classifies news articles as **real** or **fake** using Natural Language Processing and a Multinomial Naive Bayes classifier.

---

## 📌 Overview

Misinformation is a growing concern in the digital age. This project builds a text classification pipeline that can distinguish between authentic and fabricated news articles using TF-IDF features and a Naive Bayes model.

---

## 📊 Dataset

| File | Description | Size |
|---|---|---|
| `True.csv` | Verified real news articles | ~54 MB |
| `Fake.csv` | Fabricated/fake news articles | ~63 MB |

Each CSV contains columns: `title`, `text`, `subject`, `date`.

---

## ⚙️ Methodology

1. **Labeling** — Real news labeled `0`, fake news labeled `1`
2. **Text Preprocessing**
   - Lowercasing
   - Regex-based special character removal
   - Stopword removal (NLTK)
   - WordNet lemmatization
3. **Feature Extraction** — TF-IDF vectorization (up to 50,000 features, unigrams + bigrams)
4. **Model** — Multinomial Naive Bayes
5. **Evaluation** — Classification report and accuracy score on train/test sets

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🚀 Usage

### Run the script

```bash
python fake_news_detection.py
```

### Or use the notebook

Open `Fake News Detection.ipynb` in Jupyter Notebook or Google Colab for an interactive walkthrough with EDA and results.

---

## 🐛 Bugs Fixed

The following bugs were found and fixed in `fake_news_detection.py`:

| # | Line | Bug | Fix |
|---|---|---|---|
| 1 | 16 | Hardcoded path `./Desktop/ProjectGurukul/Fake News Detection/` | Changed to `./` (relative to project root) |
| 2 | 68 | `clean_data()` used undefined variable `row` | Changed to `text` (the function parameter) |
| 3 | 70 | `clean_data()` used undefined variable `news` | Changed to `token` (the processed word list) |
| 4 | 92 | `vectorizer.fit_transform(train_data)` — `train_data` not yet defined | Changed to `train_X` |
| 5 | 98-99 | `get_feature_names()` deprecated in scikit-learn ≥1.0 | Changed to `get_feature_names_out()` |

---

## 📁 Project Structure

```
Media Credibility Analysis/
├── Fake News Detection.ipynb   # Interactive notebook
├── fake_news_detection.py      # Standalone Python script
├── True.csv                    # Real news dataset
├── Fake.csv                    # Fake news dataset
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📜 License

This project is for educational and research purposes.
