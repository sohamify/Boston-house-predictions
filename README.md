# Boston Housing Price Prediction - A Comprehensive Linear Regression Pipeline

This project presents a robust, end-to-end machine learning solution for predicting median house prices in Boston, Massachusetts. The solution is built on a Linear Regression model, but it goes far beyond a simple implementation. It includes meticulous data preprocessing, advanced feature engineering, model regularization, and the creation of a full prediction pipeline for easy deployment.

The goal is to not only build an accurate predictive model but also to demonstrate a professional and reproducible machine learning workflow.

## Table of Contents

1.  **Project Overview**
2.  **Dataset**
3.  **Project Workflow**
    * Step 1: Data Loading & Initial Inspection
    * Step 2: Exploratory Data Analysis (EDA)
    * Step 3: Feature Engineering & Preprocessing
    * Step 4: Model Building & Evaluation
    * Step 5: Final Model Selection & Export
4.  **Underlying Concepts & Mathematics**
5.  **How to Run the Project**
6.  **Dependencies**

---

## 1. Project Overview

The project follows a standard machine learning pipeline to predict `MEDV` (Median value of owner-occupied homes) using 13 different features. The key steps and highlights are:

* **Exploratory Data Analysis (EDA):** Visualizing feature distributions, identifying outliers, and analyzing correlations to understand the data's characteristics.
* **Feature Engineering:** A crucial step that includes handling outliers, transforming skewed features, creating polynomial terms for non-linear relationships, and adding interaction features based on domain knowledge.
* **Multicollinearity Handling:** Using the Variance Inflation Factor (VIF) to detect and mitigate multicollinearity among features, which can destabilize linear models.
* **Model Building:** Training and comparing three models:
    * **Ordinary Least Squares (OLS) Linear Regression** as a baseline.
    * **Ridge Regression (L2 regularization)** to reduce overfitting.
    * **Lasso Regression (L1 regularization)** to both regularize and perform automatic feature selection.
* **Hyperparameter Tuning:** Using `GridSearchCV` with cross-validation to find the optimal regularization strength (`alpha`) for Ridge and Lasso.
* **Prediction Pipeline:** Encapsulating all preprocessing steps and the final model into a single, reusable `Pipeline` object.
* **Model Export:** Saving the entire trained pipeline to a file (`.joblib`), making it ready for deployment and future use without retraining.

---

## 2. Dataset

The dataset is a classic machine learning benchmark, comprising 506 samples with 13 features describing various aspects of residential areas in Boston. The target variable is `MEDV`.

| Feature | Description |
| :--- | :--- |
| `CRIM` | per capita crime rate by town |
| `ZN` | proportion of residential land zoned for lots over $25,000 ft^2$ |
| `INDUS` | proportion of non-retail business acres per town |
| `CHAS` | Charles River dummy variable (1 if tract bounds river; 0 otherwise) |
| `NOX` | nitric oxides concentration (parts per 10 million) |
| `RM` | average number of rooms per dwelling |
| `AGE` | proportion of owner-occupied units built prior to 1940 |
| `DIS` | weighted distances to five Boston employment centres |
| `RAD` | index of accessibility to radial highways |
| `TAX` | full-value property-tax rate per $10,000 |
| `PTRATIO` | pupil-teacher ratio by town |
| `B` | $1000(Bk - 0.63)^2$ where $Bk$ is the proportion of black residents by town |
| `LSTAT` | % lower status of the population |
| `MEDV` | **Target:** Median value of owner-occupied homes in $1000s |

---

## 3. Project Workflow

### Step 1: Data Loading & Initial Inspection

The process begins by loading the `boston.csv` file into a Pandas DataFrame. We use `df.head()`, `df.info()`, and `df.describe()` to get a first look at the data's structure, identify potential missing values, and understand the basic statistical distribution of each feature. This step confirms the data is clean, with no missing values, but highlights the need for further analysis on feature distributions and outliers.

### Step 2: Exploratory Data Analysis (EDA)

This is a critical step for understanding the data's underlying patterns.

* **Univariate Analysis:**
    * Histograms and box plots revealed that many features (`CRIM`, `ZN`, `LSTAT`, etc.) are highly **skewed** and contain significant **outliers**.
    * This finding suggests that these features need to be transformed to a more normal distribution for the Linear Regression model to perform optimally.
* **Bivariate Analysis:**
    * Scatter plots against the target (`MEDV`) showed strong relationships, both linear and non-linear. For example, `LSTAT` and `RM` showed clear non-linear patterns, suggesting the need for polynomial features.
* **Correlation & Multicollinearity:**
    * A correlation heatmap identified highly correlated features. Notably, `RAD` and `TAX` showed a correlation of **0.91**, which is a strong indicator of multicollinearity. This will be addressed in the next step.

### Step 3: Feature Engineering & Preprocessing

This is where the raw data is transformed into a format that a Linear Regression model can effectively learn from.

* **Outlier Handling (Capping):**
    * Extreme values (outliers) in features like `CRIM` and `LSTAT` can disproportionately influence the model's coefficients.
    * We used a capping strategy, setting values below the 1st percentile to the 1st percentile value and values above the 99th percentile to the 99th percentile value. This is a robust method that preserves data integrity while mitigating the impact of outliers.
* **Skewness Transformation (Yeo-Johnson):**
    * The Power Transformer with the Yeo-Johnson method was applied to highly skewed features. This transformation attempts to make the feature distributions more Gaussian-like, which is a key assumption for many linear models.
* **Polynomial Feature Creation:**
    * The relationships between `LSTAT`, `RM`, and `MEDV` were visibly non-linear. To allow a linear model to capture these patterns, we created second-degree polynomial features (e.g., `LSTAT^2`, `RM^2`).
* **Interaction Feature Creation:**
    * Based on a hypothesis that the impact of pollution (`NOX`) might be dependent on the distance to employment centers (`DIS`), an interaction feature `DIS_NOX_interaction` was created by multiplying the two features. This allows the model to learn a more complex, non-additive relationship.
* **Multicollinearity Handling (VIF):**
    * **Multicollinearity** occurs when two or more predictor variables in a regression model are highly correlated, leading to unstable coefficient estimates.
    * The **Variance Inflation Factor (VIF)** is a metric used to quantify the severity of multicollinearity. A VIF value greater than 5 or 10 is often considered a sign of high multicollinearity.
    * Our analysis showed `RAD` and `TAX` had VIFs well above 10. We made a strategic decision to remove `RAD`, which resulted in a significant reduction in VIF for `TAX` and other features, stabilizing the model.
* **Feature Scaling (StandardScaler):**
    * All features were standardized using `StandardScaler`, which transforms the data to have a mean of 0 and a standard deviation of 1.
    * This step is crucial for regularization algorithms like Ridge and Lasso, as it ensures all features contribute equally to the penalty term and prevents features with larger scales from dominating the model.

### Step 4: Model Building & Evaluation

We compare three types of linear models to find the best fit.

* **Linear Regression (OLS):** The basic model establishes a baseline performance.
* **Ridge Regression:** This model adds an L2 penalty to the cost function. This penalty shrinks the coefficients towards zero but does not set them to zero. It is effective at reducing overfitting and is particularly useful when multicollinearity is present.
    * **Mathematics:** Cost Function + $\lambda \sum_{j=1}^{p} \beta_j^2$
* **Lasso Regression:** This model adds an L1 penalty to the cost function. This penalty has the unique property of being able to shrink some coefficients to exactly zero. This makes Lasso a powerful tool for automatic **feature selection**.
    * **Mathematics:** Cost Function + $\lambda \sum_{j=1}^{p} |\beta_j|$

For both Ridge and Lasso, we use `GridSearchCV` to find the optimal regularization parameter `alpha` by performing cross-validation on a grid of possible values.

The performance of all models is evaluated on the unseen test set using key metrics:
* **Mean Absolute Error (MAE):** The average absolute difference between actual and predicted values. It's a robust metric in the same units as the target variable.
* **Root Mean Squared Error (RMSE):** The square root of the average of the squared errors. It's more sensitive to large errors than MAE.
* **R-squared ($R^2$):** A value between 0 and 1 that represents the proportion of the variance in the target variable that is predictable from the features. A higher value indicates a better fit.

Visualizations like the **Actual vs. Predicted Plot** and the **Residual Plot** are used to visually confirm the model's performance. The residual plot, showing a random scatter of points around the zero line, confirms that the model has captured the underlying patterns without systematic errors.

### Step 5: Final Model Selection & Export

Based on the evaluation metrics, the Lasso model was chosen as the final model due to its strong performance and its ability to perform feature selection, resulting in a more interpretable model.

A `Pipeline` was then constructed, which combines all the preprocessing steps and the final Lasso model into a single object. This pipeline is the final deployable artifact.

```python
# Example of the final prediction pipeline
full_pipeline = Pipeline(steps=[
    ('feature_engineer', CustomFeatureEngineer(...)),
    ('scaler', StandardScaler()),
    ('regressor', best_lasso_model)
])
