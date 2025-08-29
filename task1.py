from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
df1 = pd.read_csv("Mall_Customers.csv")
df1 = pd.get_dummies(df1, columns=['Gender'], drop_first=True) # drop_first=True to avoid multicollinearity
x = df1[["CustomerID", "Age", "Annual Income (k$)", "Gender_Male"]] # Include the new 'Gender_Male' column
y = df1[["Spending Score (1-100)"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
x_test['Predicted Spending Score'] = y_pred
output_df = x_test.merge(df1[['CustomerID', 'Gender_Male']], on='CustomerID', how='left')
output_df['Gender'] = output_df['Gender_Male_y'].apply(lambda x: 'Male' if x else 'Female')
print(output_df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Predicted Spending Score']])