import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

st.set_page_config(page_title="Dashboard K-Means & Naive Bayes", page_icon="🩺")
st.title("Dashboard K-Means Clustering & Naive Bayes untuk Penyakit Jantung")

# Load data
df = pd.read_csv("heart.csv")

# Preprocessing
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

categorical_cols = X.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------
# 1. Eksplorasi Data
# ---------------------
st.subheader("1. Eksplorasi Data")
st.write("Contoh data:")
st.dataframe(df.head())
st.write("Statistik deskriptif:")
st.dataframe(df.describe())

with st.expander("Visualisasi Barplot untuk Kolom Kategorikal"):
    for col in df.select_dtypes(include=["object"]).columns:
        if 2 < df[col].nunique() < 6:
            fig, ax = plt.subplots()
            df[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Distribusi {col}')
            st.pyplot(fig)

# ---------------------
# 2. K-Means Clustering
# ---------------------
st.subheader("2. K-Means Clustering")
k = st.slider("Pilih jumlah klaster:", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig_kmeans, ax_kmeans = plt.subplots()
scatter = ax_kmeans.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels, cmap='viridis')
ax_kmeans.set_title(f'Visualisasi PCA - {k} Klaster')
st.pyplot(fig_kmeans)

# ---------------------
# 3. Naive Bayes Model
# ---------------------
st.subheader("3. Naive Bayes Classification")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{acc:.2f}")
with col2:
    st.metric("Precision", f"{prec:.2f}")
with col3:
    st.metric("Recall", f"{rec:.2f}")
with col4:
    st.metric("F1 Score", f"{f1:.2f}")

with st.expander("Lihat Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

# ---------------------
# 4. Prediksi Baru
# ---------------------
st.subheader("4. Prediksi Data Baru")
with st.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 50)
    sex = st.selectbox("Sex", ["M", "F"])
    cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])
    submit = st.form_submit_button("Prediksi")

def map_inputs():
    cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
    restecg_map = {"Normal": 1, "ST": 2, "LVH": 0}
    sex_map = {"M": 1, "F": 0}
    exang_map = {"Y": 1, "N": 0}
    slope_map = {"Up": 2, "Flat": 1, "Down": 0}

    return pd.DataFrame([{
        "Age": age,
        "Sex": sex_map[sex],
        "ChestPainType": cp_map[cp],
        "RestingBP": trestbps,
        "Cholesterol": chol,
        "FastingBS": fbs,
        "RestingECG": restecg_map[restecg],
        "MaxHR": thalach,
        "ExerciseAngina": exang_map[exang],
        "Oldpeak": oldpeak,
        "ST_Slope": slope_map[slope]
    }])

if submit:
    input_df = map_inputs()
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    st.write(f"Hasil Prediksi: **{'Berisiko' if prediction == 1 else 'Tidak Berisiko'}** terkena penyakit jantung")
