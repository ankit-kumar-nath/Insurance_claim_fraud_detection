# 🚗 AI-Powered Auto Insurance Fraud Detection

This Streamlit-based web app helps detect potential fraudulent auto insurance claims using machine learning. It supports both **batch predictions via CSV uploads** and **manual single-record fraud checks**.


## 📌 Features

- ✅ Upload and process insurance claim CSV files
- 📊 Auto-generate visualizations for numerical and categorical data
- 🧠 Predict fraud using a trained Random Forest classifier
- 🧾 Manual input section for individual fraud prediction
- 📥 Download results as CSV and Excel (XLSX)
- 🔍 In-depth feature engineering for smarter predictions


## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend/Modeling**: Scikit-learn, XGBoost
- **Visualization**: Seaborn, Matplotlib
- **Data Handling**: Pandas, NumPy
- **Exporting**: `xlsxwriter`, `csv`, `io.BytesIO`
---

## 🧪 Setup Instructions

### 🔧 Prerequisites

- Python 3.7+
- pip

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/auto-insurance-fraud.git
cd auto-insurance-fraud
  
# Install dependencies
pip install -r requirements.txt
