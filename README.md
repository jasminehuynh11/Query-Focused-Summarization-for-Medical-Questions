# Query-Focused Summarization for Medical Questions

This repository presents a machine learning project focusing on query-focused summarization in the medical domain. The objective is to identify relevant sentences from medical publications that answer specific medical questions. The project utilizes deep learning models, including Siamese neural networks, LSTMs, and Transformer-based models using BERT, to perform this task effectively.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Workflow](#project-workflow)
3. [Implemented Models](#implemented-models)
4. [Setup and Usage](#setup-and-usage)
5. [Results and Analysis](#results-and-analysis)
6. [Future Enhancements](#future-enhancements)

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
  - Convert text data into embeddings using BERT.

### 2. Model Development
- **Models Implemented**:
  - Siamese Neural Networks
  - LSTM-based Networks
  - Transformer-based Models with BERT as a feature extractor
- **Optimization**:
  - Evaluate configurations of hidden layers, LSTM units, transformer encoder/decoder layers, and BERT parameters.
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

### 3. Transformer-Based Model with BERT
- **Architecture**:
  - BERT as a feature extractor for token embeddings.
  - Two transformer encoder layers (hidden dimension 768, 8 attention heads).
  - Two transformer decoder layers (hidden dimension 768, 8 attention heads).
  - One hidden layer (512 units) and a binary classification output layer.
- **Training**:
  - Dataset is formatted as `[CLS] question [SEP] answer [SEP]`.
  - Trained using binary crossentropy loss and Adam optimizer with a learning rate of `1e-5`.
- **Performance**:
  - Precision: 0.2727
  - Recall: 1.0000
  - F1 Score: 0.4286
- **Analysis**:
  - The model achieved perfect recall, identifying all relevant sentences, but at the cost of low precision, resulting in a moderate F1 score. Extended training and tuning are required to balance performance.

### 4. Custom Summarization Function
- Extracts the top `n` most relevant sentences for each question based on model predictions.

---

## Setup and Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- Hugging Face Transformers
- Pandas, NumPy, Scikit-learn

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/jasminehuynh11/Query-Focused-Summarization-for-Medical-Questions.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Query-Focused-Summarization-for-Medical-Questions
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Prepare Data
- Place your dataset in the `data/` directory.

### Train Models
- Run the provided scripts to train Siamese, LSTM, and Transformer-based models.

### Evaluate Models
- Use the `summarizer` function to test on unseen data.

---

## Results and Analysis

### Comparative Performance

| Model                    | F1 Score |
|--------------------------|----------|
| Siamese NN (Simple)      | 0.5333   |
| LSTM Siamese NN (Best)   | 0.6392   |
| Transformer-Based Model  | 0.4286   |

### Key Insights

- **BERT-based Transformer Models**: Demonstrate the ability to capture complex semantic relationships but require more training and tuning for balanced performance.
- **LSTM-based Models**: Currently outperform the Transformer in terms of precision and F1 score, likely due to the smaller dataset and computational constraints.
- **Precision vs. Recall Trade-off**: Transformer models achieve high recall but struggle with precision.

---

## Future Enhancements

### Transformer Model Tuning
- Extend training epochs for the Transformer model to improve performance.
- Experiment with different thresholds to balance precision and recall.
- Tune hyperparameters, such as the number of Transformer layers and learning rate.

### Data Augmentation
- Increase the dataset size or use augmentation techniques to improve model generalization.

### Ensemble Methods
- Combine LSTM and Transformer-based models to leverage the strengths of both.

---

## Deployment
- Deploy the summarization system as a web application using Flask or Streamlit for real-world usability.
