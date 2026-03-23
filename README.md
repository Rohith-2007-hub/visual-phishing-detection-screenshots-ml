# Visual Phishing Detection Using Screenshots

## 🔍 Overview

This project presents a hybrid machine learning framework for detecting phishing webpages using screenshot images. The system operates independently of URLs and HTML, making it effective for zero-day phishing detection.

## ⚙️ Key Features

* Screenshot-based phishing detection
* Multi-crop slicing (Top, Middle, Bottom)
* EfficientNetB3 for deep feature extraction
* Handcrafted features (texture, color, edge, frequency, ELA)
* CatBoost classifier
* Total features per image: **4935**

## 🧠 System Pipeline

1. Input screenshot
2. Preprocessing
3. Multi-crop segmentation
4. Feature extraction
5. Feature fusion
6. Classification

## 📊 Results

* Accuracy: **91.49%**
* F1-score: **0.91**

## 📁 Project Structure

* `src/` → core modules
* `main.py` → execution file
* `requirements.txt` → dependencies

## 📦 Dataset

The dataset consists of phishing and legitimate webpage screenshots.

👉 (Add your Kaggle dataset link here)

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

## 📌 Author

T.Rohith Reddy
