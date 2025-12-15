# ECE_6143_ML_Project
# Mushroom Toxicity Classification

This is a machine learning project for **ECE 6143 Machine Learning**.  
The goal is to predict whether a mushroom is **edible** or **poisonous** based on its physical features (e.g., odor, color, shape).

All features in this dataset are **categorical**, and we use **One-Hot Encoding** to prepare them for machine learning models.

---

## Team Members

* **Jiangyue Chu    (Netid: jc13605)**
* **Xinyu Wang      (Netid: xw3713)**

---

## Dataset

We use the **Mushroom Classification** dataset sourced from Kaggle:

**Original Source:** [Kaggle - Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification?resource=download)

- **Samples:** 8,124  
- **Features:** 22 categorical features (shape, surface, color, odor, etc.)  
- **Target labels:**
  - `e` = edible (mapped to 0)  
  - `p` = poisonous (mapped to 1)

> **Note:** For ease of reproducibility in Google Colab, our code loads the dataset directly from a GitHub raw link mirror, but the data content is identical to the Kaggle version.

```python
# Code snippet for loading data
df = pd.read_csv("https://raw.githubusercontent.com/jc13605-0721/ECE_6143_ML_Project/main/mushrooms.csv")
```

---

## Project Workflow

### 1. Data Processing
- **Label Encoding:** Map target `e → 0` and `p → 1`.
- **Splitting:** Split the dataset into training (80%) and testing (20%) sets.
- **Preprocessing:** Apply **One-Hot Encoding** to all categorical features using `ColumnTransformer` and `Pipeline`.

### 2. Models Implemented
We implemented and compared four different classifiers:

1. **Logistic Regression** 
2. **K-Nearest Neighbors**
3. **Decision Tree**
4. **Random Forest**

### 3. Evaluation Metrics
For each model, we computed:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC Score
- Confusion Matrix

---

## Hyperparameter Tuning

We used **GridSearchCV (5-fold cross-validation)** to find the best parameters for our tree-based models:

- **Decision Tree:** Tuned `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **Random Forest:** Tuned `n_estimators`, `max_depth`, and splits.

---

## Model Interpretability & Insights

One of the key goals of this project was to understand *why* a mushroom is poisonous.

### 1. Feature Importance (Random Forest)
We extracted the top 20 most important features. The results show that **Odor** is the single most critical predictor.
- **Odor = Foul:** Highly indicative of a poisonous mushroom.
- **Odor = None:** Highly indicative of an edible mushroom.

### 2. Decision Tree Visualization
We visualized the top 3 levels of the Decision Tree. The root node splits directly on **Odor**, confirming that smell is the primary rule for safety.

---

## Results Summary

Our models achieved exceptional performance on the test set:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **Logistic Regression** | **~99.9%** | **~0.99** | **1.00** |
| **KNN** | **100%** | **1.00** | **1.00** |
| **Decision Tree** | **100%** | **1.00** | **1.00** |
| **Random Forest** | **100%** | **1.00** | **1.00** |

**Key Finding:** The dataset features (especially Odor) are highly predictive, allowing tree-based models to achieve **perfect classification** on the test set.

---

## How to Run the Project

### 1. Prerequisites
Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Running the Code
You can run the project in Jupyter Notebook or Google Colab:

1. Open `mushroom.ipynb`.
2. Run all cells sequentially.
3. The script will automatically load data, train models, and generate visualization plots.

---

## Project Structure

```text
├── mushroom.ipynb
├── mushrooms.csv  
└── README.md
```
