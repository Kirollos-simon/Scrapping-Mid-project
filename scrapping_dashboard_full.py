
# SCRAPPING DASHBOARD WITH ML INSIGHTS
# Author: Kirollos Simon
# Final Project for Scrap Risk Analysis

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
@st.cache_data
def load_data():
    file_path = 'cleaned_scrapping_provision_from_modified1.csv'
    return pd.read_csv(file_path)

data = load_data()

# Title and Sidebar
st.title("📦 Scrapping Provision Dashboard with ML Insights")
page = st.sidebar.radio("Navigation", ["Summary Dashboard", "Detailed Analysis", "Risk & Financial Insights", "Additional Insights", "ML Insights"])

# Summary Dashboard
if page == "Summary Dashboard":
    st.header("Page 1: Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Quantity", f"{data['Quantity'].sum():,.2f}")
    with col2:
        st.metric("Total Value (EGP)", f"{data['Value in EGP'].sum():,.2f}")
    with col3:
        st.metric("Unique Products", len(data["Product"].unique()))

    if "Segement" in data.columns:
        st.subheader("Total Value by Segement")
        segement_value = data.groupby("Segement")["Value in EGP"].sum().reset_index()
        fig_segement = px.bar(segement_value, x="Segement", y="Value in EGP", color="Segement",
                              title="Total Value by Segement", labels={"Value in EGP": "Total Value (EGP)"},
                              color_continuous_scale="Viridis", text="Value in EGP")
        fig_segement.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_segement.update_layout(template="plotly_white")
        st.plotly_chart(fig_segement)

    if "Plant" in data.columns and "Segement" in data.columns:
        st.subheader("Total Value by Plant and Segement")
        plant_segment_value = data.groupby(["Plant", "Segement"])["Value in EGP"].sum().reset_index()
        fig_plant_segment = px.bar(plant_segment_value, x="Plant", y="Value in EGP", color="Segement",
                                   title="Total Value by Plant and Segement",
                                   labels={"Value in EGP": "Total Value (EGP)"},
                                   barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
        fig_plant_segment.update_layout(template="plotly_white")
        st.plotly_chart(fig_plant_segment)

# Detailed Analysis
elif page == "Detailed Analysis":
    st.header("Page 2: Detailed Analysis")
    month_filter = st.selectbox("Select Month", options=["All"] + list(data["Scrapping Month"].unique()))
    plant_filter = st.selectbox("Select Plant", options=["All"] + list(data["Plant"].unique()))
    filtered_data = data
    if month_filter != "All":
        filtered_data = filtered_data[filtered_data["Scrapping Month"] == month_filter]
    if plant_filter != "All":
        filtered_data = filtered_data[filtered_data["Plant"] == plant_filter]

    st.subheader("Total Value per Classification (Plotly)")
    classification_value = filtered_data.groupby("Classification")["Value in EGP"].sum().reset_index()
    fig_classification = px.bar(classification_value, x="Classification", y="Value in EGP", color="Classification",
                                title="Total Value per Classification", color_continuous_scale="Viridis")
    fig_classification.update_layout(template="plotly_white")
    st.plotly_chart(fig_classification)

    st.subheader("Product Details by Classification")
    product_table = filtered_data.groupby(["Product", "Classification"])["Value in EGP"].sum().reset_index()
    product_table = product_table.sort_values(by="Value in EGP", ascending=False)
    grand_total_value = product_table["Value in EGP"].sum()
    product_table["% of Grand Total"] = (product_table["Value in EGP"] / grand_total_value) * 100
    product_table["% of Grand Total"] = product_table["% of Grand Total"].apply(lambda x: f"{x:.2f}%")

    def color_high_value(val):
        return "color: red" if val > product_table["Value in EGP"].quantile(0.9) else ""

    styled_table = product_table.style.applymap(color_high_value, subset=["Value in EGP"])
    st.dataframe(styled_table)

# Risk & Financial Insights
elif page == "Risk & Financial Insights":
    st.header("Page 3: Risk and Financial Insights")
    plant_filter = st.selectbox("Select Plant", options=["All"] + list(data["Plant"].unique()))
    filtered_data = data if plant_filter == "All" else data[data["Plant"] == plant_filter]

    st.subheader("Risk Level Distribution (Plotly)")
    fig = px.pie(filtered_data, names="Risk Level", title="Risk Level Distribution")
    st.plotly_chart(fig)

    st.subheader("Value vs. Quantity by Risk Level (Plotly)")
    fig = px.scatter(filtered_data, x="Quantity", y="Value in EGP", color="Risk Level",
                     title="Value vs. Quantity by Risk Level")
    st.plotly_chart(fig)

    st.subheader("Value by Risk Level (Matplotlib)")
    plt.figure(figsize=(10, 6))
    filtered_data.boxplot(column="Value in EGP", by="Risk Level", grid=False, patch_artist=True,
                          boxprops=dict(facecolor="lightblue"))
    plt.title("Value Distribution by Risk Level")
    plt.suptitle("")
    st.pyplot(plt)

# Additional Insights
elif page == "Additional Insights":
    st.header("Page 4: Three-Level Hierarchical Scrapping Insights")
    st.subheader("Total Value by Level 1")
    level1_value = data.groupby("Level 1")["Value in EGP"].sum().reset_index()
    fig_level1 = px.bar(level1_value, x="Level 1", y="Value in EGP", color="Level 1", title="Total Value by Level 1",
                        color_discrete_sequence=px.colors.qualitative.Set3)
    fig_level1.update_layout(template="plotly_white")
    st.plotly_chart(fig_level1)

    st.subheader("Value Distribution across Levels (Sunburst Chart)")
    sunburst_data = data.groupby(["Level 1", "Level 2", "Level 3"])["Value in EGP"].sum().reset_index()
    fig_sunburst = px.sunburst(sunburst_data, path=["Level 1", "Level 2", "Level 3"],
                                values="Value in EGP", color="Level 1", title="Value Distribution across Levels",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_sunburst)

    st.subheader("Detailed Classification Insights")
    detailed_classification = data.groupby(["Classification", "Level 1", "Level 2", "Level 3"])[["Quantity", "Value in EGP"]].sum().reset_index()
    st.dataframe(detailed_classification)

# ML Insights
elif page == "ML Insights":
    st.header("🤖 Machine Learning Insights: Risk Prediction")

    data['Product'] = data['Product'].fillna('Unknown')
    data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
    data['Quantity'] = data['Quantity'].fillna(data['Quantity'].mean())
    data['Value in EGP'] = pd.to_numeric(data['Value in EGP'], errors='coerce')
    data['Risk Level'] = data['Risk Level'].fillna('Unknown')

    le = LabelEncoder()
    data['Risk_Level_Label'] = le.fit_transform(data['Risk Level'])

    data_encoded = pd.get_dummies(data[['Plant', 'Segement', 'Classification']], drop_first=True)
    features = pd.concat([data[['Quantity', 'Value in EGP']], data_encoded], axis=1)
    target = data['Risk_Level_Label']
    features = features.fillna(features.mean(numeric_only=True))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    st.subheader("Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    st.success(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
    st.text("Classification Report (Random Forest)")
    st.text(classification_report(y_test, rf_preds))

    try:
        import xgboost as xgb
        st.subheader("XGBoost Classifier")
        xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        st.success(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.2f}")
        st.text("Classification Report (XGBoost)")
        st.text(classification_report(y_test, xgb_preds))
    except ImportError:
        st.warning("XGBoost is not installed. Run `pip install xgboost`.")

    st.subheader("Feature Importance (Random Forest)")
    importances = rf.feature_importances_
    feat_names = features.columns
    feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:10]
    fig, ax = plt.subplots()
    feat_imp.plot(kind='barh', color='green', ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
    st.info("This page uses classification models to predict scrap risk level based on quantity, value, and categorical features.")
