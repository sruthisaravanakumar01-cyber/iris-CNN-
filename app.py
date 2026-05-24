import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load model and scaler
model = tf.keras.models.load_model("iris_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🌸 Iris Flower Classification App")

st.write("Enter flower measurements:")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    result = np.argmax(prediction)

    labels = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Prediction: {labels[result]}")
