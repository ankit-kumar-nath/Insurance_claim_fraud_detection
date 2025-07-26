import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import io

# --- Load Model and Artifacts ---
best_model = joblib.load("random_forest_model.joblib")
pipeline = joblib.load("preprocessing_pipeline.joblib")
encoders = joblib.load("label_encoders.joblib")
features = joblib.load("model_features.joblib")

# --- UI ---
st.set_page_config(page_title="Auto Insurance Fraud Detection", layout="centered")
st.title("🚗 AI-Powered Auto Insurance Fraud Detection")

st.markdown("""
Upload a CSV file of auto insurance claims to detect potential fraud, or manually enter key parameters to get a prediction.
""")

uploaded_file = st.file_uploader("Upload Claims CSV", type=["csv"])

def preprocess(df):
    df = df.copy()
    date_cols = ['Bind_Date1', 'Policy_Start_Date', 'Policy_Expiry_Date',
                 'Accident_Date', 'Claims_Date', 'DL_Expiry_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if all(col in df.columns for col in date_cols):
        df["Policy_Duration_Days"] = (df["Policy_Expiry_Date"] - df["Policy_Start_Date"]).dt.days
        df["Days_To_Accident"] = (df["Accident_Date"] - df["Bind_Date1"]).dt.days
        df["Days_To_Claim"] = (df["Claims_Date"] - df["Accident_Date"]).dt.days
        df["DL_Valid_On_Accident"] = (df["DL_Expiry_Date"] > df["Accident_Date"]).astype(int)
        df["Accident_Month"] = df["Accident_Date"].dt.month
        df["Accident_Weekday"] = df["Accident_Date"].dt.weekday
        df["Is_Weekend_Accident"] = df["Accident_Weekday"].isin([5, 6]).astype(int)

    df["Claim_to_Cost_Ratio"] = df["Total_Claim"] / (df["Vehicle_Cost"] + 1e-6)
    df["Claim_to_Premium_Ratio"] = df["Total_Claim"] / (df["Policy_Premium"] + 1e-6)
    df["Net_Capital"] = df["Capital_Gains"] - df["Capital_Loss"]
    df["Is_High_Capital_Gain"] = (df["Capital_Gains"] > 10000).astype(int)
    df["Is_Underinsured"] = (df["Total_Claim"] > df["Umbrella_Limit"]).astype(int)
    df["Is_Young_Driver"] = (df["Age_Insured"] < 25).astype(int)
    df["Vehicle_Age"] = df["Accident_Date"].dt.year - df["Auto_Year"] if "Accident_Date" in df.columns and "Auto_Year" in df.columns else 0
    df["Is_Old_Vehicle"] = (df["Vehicle_Age"] > 10).astype(int)
    df["Is_High_Mileage"] = (df["Annual_Mileage"] > 15000).astype(int)
    dark_colors = ['black', 'navy blue', 'dark blue', 'dark gray', 'grey']
    df["Dark_Vehicle_Color"] = df["Vehicle_Color"].str.lower().isin(dark_colors).astype(int)
    df["No_Witnesses_Flag"] = (df["Witnesses"] == 0).astype(int)
    df["No_Police_Report_Flag"] = (df["Police_Report"].fillna("NO").str.upper() != "YES").astype(int)
    df["Suspicious_Claim_Flag"] = ((df["No_Witnesses_Flag"] == 1) &
                                   (df["No_Police_Report_Flag"] == 1) &
                                   (df["Total_Claim"] > 15000)).astype(int)

    df["Premium_per_Month"] = df["Policy_Premium"] / (df["Policy_Duration_Days"] + 1e-6)
    df["Vehicle_Value_Per_Year"] = df["Vehicle_Cost"] / (df["Vehicle_Age"] + 1e-6)
    df["Claim_Frequency"] = df["Total_Claim"] / (df["Annual_Mileage"] + 1e-6)
    df["Young_Underinsured"] = ((df["Is_Young_Driver"] == 1) & (df["Is_Underinsured"] == 1)).astype(int)

    df = df.drop(columns=['Claim_ID', 'Vehicle_Registration', 'Check_Point', 'Policy_Num'], errors='ignore')

    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except:
                df[col] = 0

    df = df.drop(columns=[col for col in df.columns if np.issubdtype(df[col].dtype, np.datetime64)], errors='ignore')
    df = df[[col for col in features if col in df.columns]]
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ File loaded successfully. Showing preview:")
    st.dataframe(df.head())

    # --- Visualizations ---
    st.subheader("📊 Data Visualizations")
    if "Fraud_Ind" in df.columns:
        fraud_counts = df["Fraud_Ind"].value_counts().rename(index={'Y': 'Fraud', 'N': 'Not Fraud'})
        fig1, ax1 = plt.subplots()
        ax1.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
        ax1.axis('equal')
        st.pyplot(fig1)

    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64] and df[col].nunique() > 1:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        elif df[col].dtype == object and df[col].nunique() <= 20:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Value Counts of {col}")
            st.pyplot(fig)

    df_proc = preprocess(df)
    df_proc = pipeline.transform(df_proc)
    predictions = best_model.predict(df_proc)

    df_result = df.copy()
    df_result["Predicted_Fraud"] = predictions
    df_result["Predicted_Fraud"] = df_result["Predicted_Fraud"].map({1: "Fraud", 0: "Not Fraud"})

    st.success("✅ Predictions completed.")
    st.dataframe(df_result)

    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("📅 Download CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

    # Display Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Predictions')

    st.download_button(
    "📊 Download Excel",
    data=output.getvalue(),
    file_name="fraud_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

else:
    st.subheader("📟 Manual Input: Predict Fraud for a Single Claim")

    manual_inputs = {
        'Total_Claim': st.number_input("Total Claim", min_value=0.0),
        'Policy_Premium': st.number_input("Policy Premium", min_value=0.0),
        'Vehicle_Cost': st.number_input("Vehicle Cost", min_value=0.0),
        'Annual_Mileage': st.number_input("Annual Mileage", min_value=0.0),
        'Age_Insured': st.slider("Age of Insured", min_value=18, max_value=100, value=35),
        'Vehicle_Age': st.slider("Vehicle Age", min_value=0, max_value=30, value=5),
        'Claim_to_Cost_Ratio': st.number_input("Claim to Cost Ratio", min_value=0.0),
        'Claim_to_Premium_Ratio': st.number_input("Claim to Premium Ratio", min_value=0.0),
        'Suspicious_Claim_Flag': st.selectbox("Suspicious Claim Flag", [0, 1])
    }

    if st.button("🔍 Predict Fraud Status"):
        input_df = pd.DataFrame([manual_inputs])

        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[features]
        input_df_transformed = pipeline.transform(input_df)
        prediction = best_model.predict(input_df_transformed)[0]
        result_label = "Fraud" if prediction == 1 else "Not Fraud"
        st.success(f"🚨 Prediction Result: **{result_label}**")
