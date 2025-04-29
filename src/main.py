# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st

# 1. Load Data
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# 2. Preprocess Data
def preprocess_data(data):
    # Example: Drop irrelevant columns
    data = data.drop(columns=["ID", "Timestamp"], errors="ignore")

    # Example: Fill missing values
    data = data.fillna(method="ffill")

    # Encoding categorical variables if any
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype("category").cat.codes

    return data

# 3. Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_test, y_test, y_pred

# 4. Visualize Data
def visualize_data(data):
    st.subheader("Accident Analysis")
    st.bar_chart(data["Severity"].value_counts())

    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# 5. Predict Accident Severity
def predict_severity(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# 6. Streamlit App Interface
def main():
    st.title("AI-Driven Traffic Accident Analysis and Prediction")
    st.write("Enhancing Road Safety through Data Insights")

    uploaded_file = st.file_uploader("Upload Traffic Accident Dataset (.csv)", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.success("Data Loaded Successfully!")

        st.subheader("Raw Data")
        st.dataframe(data.head())

        data = preprocess_data(data)

        visualize_data(data)

        if st.button("Train Model"):
            X = data.drop("Severity", axis=1)
            y = data["Severity"]

            model, accuracy, X_test, y_test, y_pred = train_model(X, y)

            st.success(f"Model Trained with Accuracy: {accuracy*100:.2f}%")
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.subheader("Predict Accident Severity")
            user_input = []
            for feature in X.columns:
                value = st.number_input(f"Enter {feature}", value=0.0)
                user_input.append(value)

            if st.button("Predict"):
                prediction = predict_severity(model, user_input)
                st.success(f"Predicted Accident Severity Level: {prediction}")

if __name__ == "__main__":
    main()
