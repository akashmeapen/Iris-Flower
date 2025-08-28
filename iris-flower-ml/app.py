import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data and train
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

st.title("Iris Flower Classifier")

sepal_length = st.slider("Sepal Length", 4.0, 8.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5)
petal_length = st.slider("Petal Length", 1.0, 7.0)
petal_width = st.slider("Petal Width", 0.1, 2.5)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
st.write("Predicted Species:", iris.target_names[prediction[0]])
