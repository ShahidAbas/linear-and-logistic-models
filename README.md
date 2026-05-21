# Linear and Logistic Models

This repository explores the application of statistical learning across both regression and classification tasks. The project bridges two core domains: analyzing vehicle characteristics (such as Fuel Type, Engine Size, and Cylinders) using the 1995-2014 Fuel Consumption Ratings dataset from the Open Government of Canada to predict continuous environmental impacts, and applying logistic classification frameworks to predict discrete categorical outcomes like customer churn behavior. Together, these implementations demonstrate a complete end-to-end data science workflow—from cleaning real-world datasets to handling acute class imbalances and evaluating predictive probability accuracy.

---
##  Implemented Models & Detailed Framework

### 1. Simple Linear Regression *Completed*
* **Purpose:** Models the direct, straight-line relationship between a single independent predictor variable and a continuous target ($CO_2$ Emissions).
* **Core Workflow:** * Isolated **Fuel Consumption** as the primary driver to map its direct mathematical correlation with tailpipe emissions.
  * Split the dataset into training and testing subsets to ensure unbiased evaluation.
* **Evaluation Metrics:** Evaluated model fit using **Mean Squared Error (MSE)** and the **$R^2$ Score (Coefficient of Determination)** to measure the percentage of emission variance perfectly captured by a single feature.

### 2. Multiple Linear Regression *Completed*
* **Purpose:** Multiplies predictive capability by incorporating a comprehensive suite of engine specifications simultaneously.
* **Core Workflow:** * Engineered a multi-variable pipeline incorporating **Engine Size**, **Cylinders**, and **Fuel Type** to construct a multidimensional hyperplane of prediction.
  * Managed feature scaling and addressed potential multicollinearity between closely related engine design metrics.
* **Evaluation Metrics:** Utilized **Adjusted $R^2$** to ensure adding extra technical variables genuinely improved predictive power rather than artificially inflating model confidence.

### 3. Logistic Regression (Classification & Optimization) *Completed*
* **Purpose:** Applied as a robust binary classification tool to predict specific categorical target outcomes (e.g., categorizing high/low efficiency states or customer churn risk).
* **Advanced Optimization Implemented:**
  * **Hyperparameter Tuning:** Tuned the regularization strength to an optimized $C=0.01$ to prevent model overfitting.
  * **Class Imbalance Handling:** Implemented `class_weight='balanced'` to prevent majority-class bias, successfully driving the **Log Loss down from ~0.625 to an optimized ~0.592**.
* **Thorough Evaluation Visualizations:**
  * **Confusion Matrix Heatmap:** To track exact True Positives, True Negatives, and Type I/II errors.
  * **ROC Curve & AUC Score:** Achieved a strong **AUC of 0.78**, proving high predictive power in class separation.
  * **Probability Distribution Plot:** Rendered using Seaborn KDE plots to visualize how confidently the model separates probabilities between the true classes.

---

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / VS Code
