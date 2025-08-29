from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv("Mall_Customers.csv")
df1 = pd.get_dummies(df1, columns=['Gender'], drop_first=True) # drop_first=True to avoid multicollinearity
def categorize(score):
    if score <= 40:
        return "Low"
    elif score <= 70:
        return "Medium"
    else:
        return "High"
df1["Score_Category"] = df1["Spending Score (1-100)"].apply(categorize)
x = df1[["CustomerID", "Age", "Annual Income (k$)", "Gender_Male"]] # Include the new 'Gender_Male' column
y = df1[["Score_Category"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
x_test['Predicted Spending Score'] = y_pred
output_df = x_test.merge(df1[['CustomerID', 'Gender_Male']], on='CustomerID', how='left')
output_df['Gender'] = output_df['Gender_Male_y'].apply(lambda x: 'Male' if x else 'Female')
print(output_df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Predicted Spending Score']])
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=["CustomerID", "Age", "Annual Income (k$)", "Gender_Male"], filled=True)
plt.show()

