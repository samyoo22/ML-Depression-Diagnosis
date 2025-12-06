# ML Depression Diagnosis

This repository contains the code and experiments for a machine learning and deep learning–based system that diagnoses depression from social media text, with a particular focus on Reddit posts from depression-related communities.

## 📌 Overview

Depression is typically diagnosed through subjective clinical interviews and self-reports, which can be inconsistent and difficult to scale. This project explores whether we can support early screening of depression with an objective, data-driven pipeline built on NLP and modern ML/DL models.

Using a cleaned **Reddit depression dataset** with binary labels (`is_depression` ∈ {0, 1}), we:

- Preprocess raw posts (stopword removal, tokenization, text cleaning)
- Analyze linguistic patterns (n-grams, sentiment analysis)
- Vectorize text (TF–IDF, Word2Vec + dimensionality reduction)
- Train and compare multiple **machine learning** and **deep learning** models for depression detection

## 🧵 Dataset & Preprocessing

- Source: Reddit posts from depression-related and non-depression subreddits (7,731 samples with binary labels).
- Labels:
  - `1` → post indicative of depression
  - `0` → non-depression post
- Preprocessing steps:
  - Removal of English stopwords and domain-specific tokens (`http`, `www`, `com`, etc.)
  - Tokenization using NLTK
  - Lowercasing and removal of non-alphanumeric characters
- Exploratory analysis:
  - N-gram (unigram, bigram, trigram) analysis of depression vs. non-depression posts
  - Shallow sentiment analysis using TextBlob

## 🧠 Models

### Classical Machine Learning

We evaluate several ML models on TF–IDF / vectorized representations:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- GradientBoostingClassifier

Among these, **GradientBoostingClassifier** achieved the highest performance, capturing non-linear patterns in the depression-related text.

### Deep Learning

We further compare a range of neural architectures:

- GRU
- Fully Connected Neural Network (FCNN)
- CNN
- LSTM
- Transformer
- **BERT**

BERT, with its bidirectional transformer-based attention, clearly outperformed the other deep learning models and achieved the **best overall accuracy** on the dataset.

## 🔍 Key Findings

- Vectorization (TF–IDF, Word2Vec) and careful preprocessing are crucial for turning noisy, unstructured Reddit text into meaningful features for ML/DL models.
- Among classical ML models, **GradientBoostingClassifier** delivers the strongest performance on this task.
- Among deep learning models, **BERT** achieves the highest accuracy (~97%), significantly outperforming simpler sequence models such as GRU.
- More advanced, context-aware architectures (Transformers, BERT) are better suited for nuanced mental health–related language than simpler models or sentiment-only baselines.

## 📈 Experiments & Visualizations

The repository includes:

- Confusion matrices and ROC curves for each ML model
- Training curves (accuracy over epochs) for each deep learning model
- Model comparison plots (average accuracy per model)

These visualizations highlight how model choice and architecture complexity affect performance on depression detection.

## 🚀 Future Work

Potential extensions include:

- Ensemble methods that combine strong ML and DL models
- Training on larger and more diverse multilingual datasets
- Applying explainable AI (XAI) techniques to make predictions interpretable for clinicians
- Validating models on real-world clinical datasets beyond Reddit

## ⚠️ Ethical Note

This project is **research-focused** and is **not** a clinical diagnostic tool.  
Model outputs should **never** be used as a substitute for professional mental health evaluation or treatment.

## 🇰🇷 간단 요약 (Korean Summary)

이 프로젝트는 **우울증 관련 Reddit 글**을 이용해서,  
머신러닝·딥러닝 모델들이 우울증 여부를 얼마나 잘 구분할 수 있는지 비교한 연구 프로젝트입니다.

- 텍스트 전처리 및 벡터화(TF–IDF, Word2Vec)를 수행하고,
- Decision Tree, Random Forest, GradientBoosting 등 전통적인 ML 모델과
- GRU, LSTM, CNN, Transformer, **BERT** 같은 딥러닝 모델을 모두 실험했습니다.

실험 결과,
- 전통 ML 중에서는 **GradientBoostingClassifier**가 가장 높은 성능을 보였고,
- 딥러닝 모델 중에서는 **BERT**가 약 97% 정확도로 가장 우수한 성능을 기록했습니다.

이 레포지토리는 코드, 실험 결과, 시각화(혼동 행렬, ROC 커브, 정확도 비교 그래프) 등을 포함하고 있으며,  
정신건강 진단에 ML/DL을 적용할 때 어떤 모델을 선택해야 하는지에 대한 인사이트를 제공합니다.
