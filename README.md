# Query-Focused Summarization for Medical Questions

This repository presents a machine learning project focusing on query-focused summarization in the medical domain. The objective is to identify relevant sentences from medical publications that answer specific medical questions. The project utilizes deep learning models, including Siamese neural networks, LSTMs, and more, to perform this task effectively.

---
## Overview

Medical query-focused summarization is a challenging task due to the complexity of natural language and the need for precise identification of relevant content. This project builds on data derived from the [BioASQ Challenge](http://www.bioasq.org/), processing a dataset of medical questions and associated sentences labeled for relevance.

### Objectives:
- Determine which sentences from medical publications are part of the answer to a given question.
- Implement and evaluate multiple deep learning models for relevance detection.

---

## Project Workflow

### 1. Data Preparation
- **Dataset**: BioASQ-derived, preprocessed CSV format with questions, sentences, and binary relevance labels.
- **Preprocessing**:
  - Generate balanced triplets (anchor, positive, and negative samples).
  - Convert text data to TF-IDF vectors.

### 2. Model Development
- **Models Implemented**:
  - Siamese Neural Networks
  - LSTM-based Networks
  - Custom distance layers for relevance scoring.
- **Optimization**:
  - Evaluate configurations of hidden layers and LSTM units.
  - Use F1-score as the primary evaluation metric.

---

## Implemented Models

### 1. Simple Siamese Neural Network
- **Architecture**: Input layers, three dense hidden layers, and a custom distance layer.
- **Performance**:
  - F1 Score: 0.5333 for the best configuration `(128, 64, 64)`.

### 2. LSTM-Based Siamese Neural Network
- **Architecture**:
  - Embedding layer (35 dimensions).
  - LSTM layer with optimized units.
  - Three hidden layers with ReLU activation.
- **Performance**:
  - Best F1 Score: 0.6392 with `36 LSTM units` and `(32, 32, 32)` hidden layers.

### 3. Custom Summarization Function
- Extracts the top `n` most relevant sentences for each question based on model predictions.

---

## Setup and Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- Keras
- Pandas, NumPy, Scikit-learn

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/jasminehuynh11/Query-Focused-Summarization-for-Medical-Questions.git
