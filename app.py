import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("Exploratory Data Analysis (EDA) App")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    
    # Show the first few rows of the dataset
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Basic Information about the data
    st.write("Basic Information:")
    st.write(df.info())

    # Summary statistics
    st.write("Summary Statistics:")
    st.write(df.describe())

    # Display correlation matrix
    st.write("Correlation Heatmap:")
    correlation = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

    # Histograms for each numeric column
    st.write("Histograms for Numeric Columns:")
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        st.subheader(f"Histogram for {col}")
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        st.pyplot()
