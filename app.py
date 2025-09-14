import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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

    # Handle missing values: Fill with 0 or drop rows/columns
    df_clean = df.fillna(0)  # Alternatively, use df.dropna() to drop missing data

    # Display correlation matrix only for numeric columns
    st.write("Correlation Heatmap:")
    numeric_df = df_clean.select_dtypes(include=["float64", "int64"])  # Select only numeric columns

    if numeric_df.empty:
        st.write("No numeric columns available for correlation.")
    else:
        # Correlation Heatmap with masking for better visualization
        correlation = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", mask=np.triu(correlation))
        st.pyplot()

    # Histograms for each numeric column
    st.write("Histograms for Numeric Columns:")
    num_cols = numeric_df.columns
    if num_cols.empty:
        st.write("No numeric columns available for histograms.")
    else:
        for col in num_cols:
            st.subheader(f"Histogram for {col}")
            plt.figure(figsize=(6, 4))
            sns.histplot(df_clean[col], kde=True)
            st.pyplot()

    # Add Boxplot and Violin plot to show distributions
    st.write("Boxplot and Violin Plot for Numeric Columns:")
    for col in num_cols:
        st.subheader(f"Boxplot for {col}")
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df_clean[col])
        st.pyplot()
        
        st.subheader(f"Violin Plot for {col}")
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=df_clean[col])
        st.pyplot()

    # Pairplot for feature relationships
    st.write("Pairplot for Feature Relationships:")
    if len(num_cols) > 1:
        pairplot = sns.pairplot(df_clean[num_cols])
        st.pyplot(pairplot)
    else:
        st.write("Not enough numeric columns for pairplot.")

    # Interactive Plot (using Plotly)
    st.write("Interactive Scatter Plot:")
    if len(num_cols) >= 2:
        x_axis = st.selectbox("Select X-axis", num_cols)
        y_axis = st.selectbox("Select Y-axis", num_cols)
        fig = px.scatter(df_clean, x=x_axis, y=y_axis, title=f"Scatter Plot of {x_axis} vs {y_axis}")
        st.plotly_chart(fig)

    # Missing Data Visualization (using heatmap)
    st.write("Missing Data Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="Blues")
    st.pyplot()

    # Feature Importance using Random Forest (if applicable)
    st.write("Feature Importance using Random Forest:")
    if len(numeric_df.columns) > 1:  # At least two columns needed
        target = st.selectbox("Select target column (for feature importance)", numeric_df.columns)
        X = numeric_df.drop(target, axis=1)
        y = numeric_df[target]
        
        # Train Random Forest to get feature importance
        model = RandomForestRegressor()
        model.fit(X, y)
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        st.write("Feature Importances:")
        st.bar_chart(feature_importance)

    # KDE Plot for smooth distribution visualization
    st.write("KDE Plot for Distribution:")
    for col in num_cols:
        st.subheader(f"KDE Plot for {col}")
        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_clean[col], shade=True)
        st.pyplot()

    # Time Series Plot (if applicable)
    st.write("Time Series Plot (if Timestamp column is available):")
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.line_chart(df.set_index('timestamp')['value'])  # Assuming 'value' is the column to plot


