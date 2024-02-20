# Anderson_Cancer_Center
This Python script performs data analysis on the breast cancer dataset using Principal Component Analysis (PCA) and implements logistic regression for prediction. The goal is to identify essential variables for securing donor funding at the Anderson Cancer Center.
**Import Necessary Libraries:**

numpy: For numerical operations and array manipulation.
pandas: For data manipulation and analysis.
matplotlib.pyplot: For data visualization.
sklearn.datasets: For loading the breast cancer dataset.
sklearn.preprocessing: For data standardization.
sklearn.decomposition: For PCA implementation.
sklearn.model_selection: For splitting data into training and testing sets.
sklearn.linear_model: For logistic regression modeling.
sklearn.metrics: For evaluating model performance.

**Load the Breast Cancer Dataset:**

Loads the breast cancer dataset from sklearn.datasets.
Creates a DataFrame X with features and a Series y with the target variable (malignant/benign).

**Standardize the Data:**

Standardizes the features in X using StandardScaler to have a mean of 0 and a standard deviation of 1.

**Perform PCA:**

Initializes a PCA object with n_components=2 to reduce the data to 2 principal components.
Fits PCA to the standardized data and transforms it into 2-dimensional components.

**Create DataFrame with Reduced Components:**

Creates a DataFrame df_pca with the 2 principal components and the target variable.

**Visualize the Reduced Dataset:**

Creates a scatter plot to visualize the distribution of samples in the reduced space based on their target values.

**Split Data into Training and Testing Sets:**

Splits the data into training and testing sets for model evaluation, using 80% for training and 20% for testing.

**(Bonus) Implement Logistic Regression for Prediction:**

Trains a logistic regression model on the training set.
Predicts on the test set.
Evaluate the model's accuracy using the accuracy_score metric.

**Key Points:**

PCA helps identify essential variables and visualize high-dimensional data in lower dimensions.
Logistic regression can be used for prediction tasks.
Data standardization is often crucial before applying PCA.
Splitting data into training and testing sets is essential for model evaluation.
Examining model accuracy and other metrics aids in understanding model performance.

