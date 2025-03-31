# Financial Sentiment Analysis – A Comparative Study

This project presents a comparative study on the efficacy of sentiment analysis in the financial domain. The goal of this study was to evaluate how accurately machine learning models can classify financial news articles as positive, neutral, or negative, and explore how sentiment analysis can aid in understanding market behaviour, predicting trends, and identifying financial risks.

## Overview

As financial markets are highly responsive to news and public sentiment, the ability to detect emotional tone in financial content has become increasingly important. This study implements and compares the performance of three machine learning algorithms:

- K-Nearest Neighbour (KNN)
- Support Vector Machine (SVM)
- Decision Tree

These models were trained and tested on a dataset of 4,847 financial news headlines, with the goal of evaluating their accuracy and practicality for use in real-world financial sentiment analysis.

## Methodology

- **Dataset**: The dataset consists of 4,847 financial news articles with sentiment labels (positive, neutral, negative), sourced from Kaggle. The dataset includes two columns: `content` (the news article) and `sentiment` (the corresponding sentiment label). The dataset is publicly available and can be accessed [here](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news).

- **Pre-processing**: Tokenisation, stop word removal, lemmatisation, punctuation cleaning, Bag-of-Words, and TF-IDF feature extraction were applied to prepare the data for model training.

- **Evaluation**: Model performance was assessed using accuracy scores, confusion matrices, and performance visualisations.

- **Optimisation**: Bayesian optimisation was used to tune hyper-parameters for the KNN model, specifically the number of neighbours (k).

## Results

The models achieved the following accuracy scores on the dataset:

| Model                         | Accuracy |
|-------------------------------|----------|
| K-Nearest Neighbour (KNN)     | 86.34%   |
| Decision Tree                 | 73.02%   |
| Support Vector Machine (SVM)  | 70.22%   |

KNN achieved the highest accuracy but required more training and prediction time. Decision Tree offered a balanced trade-off, while SVM performed the least effectively, indicating a need for further tuning or advanced feature engineering.

## Ethical Considerations

- **Market Manipulation**: Misuse of sentiment analysis tools could contribute to the spread of misleading financial information, potentially affecting investor decisions and market trends.

- **Data Privacy**: The study used publicly available content. Nonetheless, ethical data handling practices were followed to ensure respect for data privacy.

## Future Work

- Apply k-fold cross-validation to improve model robustness and reduce overfitting, especially for KNN.
- Explore hybrid models that integrate machine learning with deep learning techniques, such as LSTM.
- Extend sentiment analysis to real-time data streams from financial APIs or social media.
- Incorporate fake news detection techniques to enhance result reliability.

## Setup Instructions

To run this project locally, ensure the following are installed:

- **MATLAB** (any version supporting the functions used in the code)
- **Toolboxes**: `Statistics and Machine Learning Toolbox`, `Text Analytics Toolbox`

Ensure the dataset is placed in the correct directory before running the scripts.

## Acknowledgements

This project was developed as part of the **Intelligent Systems KF5042** module at Northumbria University, focusing on sentiment analysis in financial contexts using machine learning techniques.
