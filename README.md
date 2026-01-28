# Building a Text Classification Pipeline – Word Embedding Exploration

## Overview
This project applies classical NLP techniques to sentiment classification using the IMDb movie reviews dataset. It compares generative and discriminative models using both sparse (TF-IDF) and dense (Word2Vec) feature representations.

## Dataset
- **Source:** IMDb Movie Reviews (via `keras.datasets.imdb`)
- **Classes:** Positive (1) and Negative (0)
- **Size:** 25,000 training + 25,000 testing samples

## Pipeline Summary
1. **Data Acquisition & Exploration**
   - Loaded IMDb dataset with top 10,000 words.
   - Verified class balance and dataset size.

2. **Pre-processing Pipeline**
   - Decoded integer sequences to text.
   - Padded sequences to fixed length (200 tokens).

3. **Feature Engineering**
   - Sparse: Bag-of-Words, TF-IDF (via `CountVectorizer`, `Tn sample reviews)

4. **Modelling & Evaluation**
   - Models: Multinomial Naive Bayes, Logistic Regression
   - Evaluation: Accuracy, Precision, Recall, F1-score
   - Example results
