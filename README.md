# Machine Learning Internship Task2-Sentiment Analisys with NLP

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : BOYAPATI RAMA KRISHNA

*INTERN ID* : CT04DK494

*DOMAIN* : MACHINE LEARNING

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH


# 📊 Sentiment Analysis of Amazon Customer Reviews Using NLP, TF-IDF, and Logistic Regression

## 🧠 Project Overview

This project focuses on building a **Sentiment Analysis** model using Natural Language Processing (NLP) techniques applied to a dataset of **50,000 Amazon customer reviews**, sourced from Hugging Face's Datasets Hub. The primary goal is to classify customer reviews as **positive** or **negative** using a traditional machine learning pipeline.

The project covers a full end-to-end NLP workflow, from raw data preprocessing to feature extraction, model training, evaluation, and performance analysis — all implemented using Python in a Jupyter Notebook environment (Google Colab).

---

## 📦 Dataset Description

- **Dataset Name**: Amazon Polarity Reviews
- **Source**: Hugging Face Datasets ([link](https://huggingface.co/datasets/amazon_polarity))
- **Total Records Used**: 50,000 (subset for efficient training)
- **Features**:
  - `title`: Title of the product review
  - `content`: Body/content of the review
  - `label`: Binary label (1 = positive, 0 = negative)

The dataset represents a balanced mix of product reviews across a variety of categories, making it ideal for binary sentiment classification tasks.

---

## 🧪 Project Workflow

### 1️⃣ Data Preprocessing
- Loaded the dataset using Hugging Face Datasets API and pandas.
- Combined `title` and `content` for a more comprehensive representation of each review.
- Applied text normalization techniques:
  - Lowercasing
  - Removal of punctuation and special characters
  - Tokenization
  - Stopwords removal (using NLTK)
  - Lemmatization

### 2️⃣ Feature Extraction
- Utilized **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) from `sklearn.feature_extraction.text` to convert cleaned text into numeric feature vectors.
- Restricted feature dimensions with parameters like `max_features`, `ngram_range`, and `min_df` to optimize performance and reduce overfitting.

### 3️⃣ Model Building
- Trained a **Logistic Regression** classifier using `sklearn.linear_model.LogisticRegression`.
- Performed train-test split (80-20) for model validation.
- Fit the model on the TF-IDF vectors of the training data.

### 4️⃣ Evaluation Metrics
- Evaluated the model on test data using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report
- Visualized model performance with seaborn heatmaps and matplotlib charts.

---

## 📈 Results & Observations

- The Logistic Regression model achieved a test accuracy of **~85%**, demonstrating strong performance on the sentiment classification task.
- The confusion matrix showed balanced precision and recall across both positive and negative classes.
- TF-IDF effectively captured key word-level features contributing to review sentiment.
- Simpler models like Logistic Regression are highly interpretable and perform well when data is clean and preprocessed properly.
![Image](https://github.com/user-attachments/assets/8f7aab0a-ecd1-428c-b902-2599ecd236ee)

![Image](https://github.com/user-attachments/assets/42f3986c-5447-4200-837b-cc79eeb342f6)

![Image](https://github.com/user-attachments/assets/eb1c2862-0c8f-494f-bf2e-aef581058996)

![Image](https://github.com/user-attachments/assets/6a843c3e-da7a-4fe8-a458-1b85a3b24429)

![Image](https://github.com/user-attachments/assets/cb30a016-b5c8-4830-bd2c-c77bd89119f8)

---

## 🛠️ Tools & Technologies Used

- **Python** (3.x)
- **Google Colab**
- **Pandas**, **NumPy** – for data manipulation
- **Matplotlib**, **Seaborn** – for data visualization
- **NLTK** – for NLP preprocessing
- **Scikit-learn** – for TF-IDF, model training and evaluation
- **Hugging Face Datasets** – for sourcing and loading the dataset

---

## 🔍 Key Learnings

- Understanding the power of traditional NLP pipelines using TF-IDF.
- Importance of text preprocessing in enhancing model performance.
- Practical application of Logistic Regression in binary text classification.
- Exposure to real-world sentiment data and its nuances.
- Learned to create visualizations for performance insights and debugging.

---
