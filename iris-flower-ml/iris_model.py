from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the data
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# EDA
sns.pairplot(df, hue="species")
plt.show()

# Train-test split
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
# Example flower measurements: [sepal_length, sepal_width, petal_length, petal_width]
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

# Get species name
predicted_species = iris.target_names[prediction[0]]
print("Predicted species:", predicted_species)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)
print("Predicted species:", iris.target_names[prediction[0]])

