# Handwritten Digit Classifier using Logistic Regression

This project demonstrates a **machine learning classification task** using **Logistic Regression** to recognize handwritten digits from the **scikit-learn digits dataset**.  
The dataset contains **1,797 grayscale images** (8×8 pixels each) of handwritten digits (0–9).  

The goal of this project is to:
1. Train a Logistic Regression model to classify digits.
2. Evaluate its accuracy on unseen data.
3. Visualize classification results using a **confusion matrix heatmap**.

---

## Features
- Loads the built-in `digits` dataset from **scikit-learn**.
- Splits the data into training and test sets.
- Trains a **Logistic Regression** model for classification.
- Displays model accuracy.
- Visualizes results with a **Seaborn confusion matrix heatmap**.

---

## Requirements
Install the required Python libraries before running the script:
```bash
pip install matplotlib seaborn scikit-learn
