import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Hostel Classifier", layout="wide")

st.title("🏠 Hostel Type Classification Dashboard")

# Sidebar
st.sidebar.header("⚙️ Settings")

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
model_choice = st.sidebar.selectbox("Choose Model", ["KNN", "SVM"])

if model_choice == "KNN":
    k_value = st.sidebar.slider("K Value", 1, 15, 5)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("hostel_dataset_1000.csv")

df = load_data()

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

with col2:
    st.subheader("📐 Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

# EDA
st.subheader("📈 Hostel Type Distribution")

fig, ax = plt.subplots()
sns.countplot(x='Hostel_Type', data=df, ax=ax)
st.pyplot(fig)

# ---------------- PREPROCESSING ----------------

le = LabelEncoder()
df['Hostel_Type'] = le.fit_transform(df['Hostel_Type'])

X = df.drop(['Hostel_Type', 'Price'], axis=1)
y = df['Hostel_Type']

# Scaling (IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# ---------------- MODEL ----------------

st.subheader("🤖 Model Performance")

if model_choice == "KNN":
    model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
else:
    model = SVC(kernel='rbf', C=1.0, gamma='scale')

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

# Metrics
col3, col4 = st.columns(2)

with col3:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col4:
    st.write("Model Used:", model_choice)

# ---------------- PREDICTION ----------------

st.subheader("🔍 Make Prediction")

input_data = []

for col in X.columns:
    val = st.number_input(f"{col}", value=float(X[col].mean()))
    input_data.append(val)

if st.button("Predict"):
    # SCALE INPUT (VERY IMPORTANT)
    input_scaled = scaler.transform([input_data])

    result = model.predict(input_scaled)
    label = le.inverse_transform(result)

    st.success(f"Predicted Hostel Type: {label[0]}")