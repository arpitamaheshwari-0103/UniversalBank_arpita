
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

st.set_page_config(page_title="Universal Bank Dashboard", layout="wide")

st.title("📊 Universal Bank Personal Loan Dashboard")

# Load data
df = pd.read_csv("UniversalBank.csv")

st.subheader("Dataset Overview")
st.write(df.head())

# Drop ID
df = df.drop(columns=["ID"])

# Features and target
X = df.drop(columns=["Personal Loan"])
y = df["Personal Loan"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

st.subheader("📈 Model Performance")

fig_roc = plt.figure()

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, prec, rec, f1])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

st.pyplot(fig_roc)

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
st.dataframe(results_df)

# Confusion Matrix
st.subheader("Confusion Matrix (Random Forest)")
rf = models["Random Forest"]
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

fig_cm = plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig_cm)

# Upload new data
st.subheader("📂 Upload Test Data for Prediction")

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    preds = rf.predict(new_data)
    new_data["Predicted Personal Loan"] = preds
    st.write(new_data.head())

    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
