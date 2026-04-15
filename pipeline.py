# import pandas as pd
# df = pd.read_csv("hostel_dataset.csv")

# print(df.head()) #shows first 5 rows
# print(df.shape) #no. of rows and columns

# #Exploratory Data Analysis(EDA)
# print(df.info())
# print(df.isnull().sum())
# print(df['Hostel_Type'].value_counts())

# #Graphs
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.countplot(x = 'Hostel_Type', data = df)
# plt.show()

# #Data Preprocessing
# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# df['Hostel_Type'] = le.fit_transform(df['Hostel_Type'])

# #Feature Selection
# X = df.drop('Hostel_Type', axis = 1)
# y = df['Hostel_Type']

# #Train Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# #Train KNN Model
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# knn_pred = knn.predict(X_test)

# #Train SVM model
# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(X_train, y_train)

# svm_pred = svm.predict(X_test)

# #Evaluating Models
# from sklearn.metrics import accuracy_score

# print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
# print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# #K-Fold Validation
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(knn, X, y, cv = 5)
# print("KNN Cross Validation Score:", scores.mean())




import streamlit as st
import pandas as pd

st.title("Hostel Type Classification")

df = pd.read_csv("hostel_dataset.csv")

st.write("Dataset Preview")
st.write(df.head())

st.write("Shape of Dataset:", df.shape)

#eda
import seaborn as sns
import matplotlib.pyplot as plt

st.write("Hostel Type Distribution")

fig, ax = plt.subplots()
sns.countplot(x='Hostel_Type', data=df, ax=ax)
st.pyplot(fig)

#knn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Encode target
le = LabelEncoder()
df['Hostel_Type'] = le.fit_transform(df['Hostel_Type'])

X = df.drop('Hostel_Type', axis=1)
y = df['Hostel_Type']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

st.write("KNN Accuracy:", accuracy_score(y_test, pred))


#svm
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

st.write("SVM Accuracy:", accuracy_score(y_test, svm_pred))


#slidebar
# model_choice = st.sidebar.selectbox("Choose Model", ["KNN", "SVM"])

#buttons
# if st.button("Train Model"):
#     st.write("Training started...")