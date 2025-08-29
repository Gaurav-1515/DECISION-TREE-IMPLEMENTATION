# DECISION-TREE-IMPLEMENTATION

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : GAURAV PANDEY

*INTERN ID* : CT04DY1426

*DOMAIN* : MACHINE LEARNING

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

## 📄 Project Description: Decision Tree Classifier on Mall Customers Dataset

In this project, I have implemented and visualized a **Decision Tree Classifier** using the `scikit-learn` library to analyze and predict outcomes on customer data. The work was carried out in **Visual Studio Code (VS Code)**, where I imported the required libraries, processed the dataset, trained the model, and visualized the results. The dataset used for this project was taken from Kaggle: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). This dataset contains details of 200 mall customers including their `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1–100)`.

### 1. Data Loading and Understanding

The dataset was first loaded into a pandas DataFrame for inspection and cleaning. Each record represents one customer. The important attributes are:

* **CustomerID**: Unique identifier for each customer
* **Gender**: Categorical attribute (Male/Female)
* **Age**: Age of the customer
* **Annual Income (k\$)**: Annual income in thousands of dollars
* **Spending Score (1–100)**: A value assigned by the mall based on customer behavior and spending patterns

The target variable for this project was **Spending Score**, while the features included age, income, gender, and customer ID.

### 2. Data Preparation

Since the dataset included categorical data (`Gender`), preprocessing was required. Using **One-Hot Encoding**, the `Gender` column was converted into numeric form, resulting in a new feature `Gender_Male`. This allowed the model to interpret gender as a binary feature. After preprocessing, the **feature set (X)** contained `CustomerID`, `Age`, `Annual Income (k$)`, and `Gender_Male`. The target variable (y) remained `Spending Score (1–100)`.

To ensure that the model could be properly trained and tested, the dataset was divided into **training and testing subsets** using the `train_test_split` function from scikit-learn. Typically, 70% of the data was used for training and 30% for testing. This ensured that the model could be trained on one portion of the dataset and evaluated on unseen data for generalization.

### 3. Model Training with Decision Tree Classifier

The **Decision Tree Classifier** from `scikit-learn` was used to train the model. A decision tree works by splitting data into branches based on feature conditions, eventually leading to leaf nodes that represent predictions. During training, the model learned how features such as `Age` and `Annual Income` influence the `Spending Score`.

The model was fitted using the training data `(X_train, y_train)`. Once trained, it was used to predict the values of the test set `(X_test)`. These predictions (`y_pred`) allowed us to evaluate how well the decision tree could predict spending behavior for new, unseen customers.

### 4. Visualization of the Decision Tree

One of the main advantages of using a decision tree is **interpretability**. The trained tree was visualized using `plot_tree` from scikit-learn. The visualization displayed how the model splits customers based on features such as age and income to predict spending scores. Each node showed the decision condition, number of samples, and predicted value. This made it easier to understand the logic behind the model’s predictions.

### 5. Analysis and Insights

The results demonstrated how demographic and income details play a role in determining customers’ spending patterns. Customers with higher income and younger age groups often had higher spending scores, while certain branches of the tree revealed patterns for moderate and low spenders. Although the decision tree can capture complex patterns, it can also overfit the training data, which is something to be aware of. In practice, techniques such as pruning or using ensemble methods like Random Forests can be applied for better accuracy.

### Conclusion

This project successfully demonstrated how a **Decision Tree Classifier** can be applied to real-world customer data to classify and predict outcomes. By using scikit-learn in VS Code, I was able to preprocess the data, train the model, visualize the decision tree, and analyze customer spending behavior. The Kaggle dataset provided a practical scenario for applying machine learning techniques, and the decision tree model offered an intuitive way to understand customer segmentation. Overall, the project highlights the importance of machine learning in business decision-making, particularly in customer behavior prediction and segmentation.

#OUTPUT
<img width="1464" height="1135" alt="Image" src="https://github.com/user-attachments/assets/53a1bd50-e3f5-443c-9bc1-37e186db7095" />

<img width="1919" height="1141" alt="Image" src="https://github.com/user-attachments/assets/e94f3dd3-bc65-4781-b3e8-79997d2b0591" />
