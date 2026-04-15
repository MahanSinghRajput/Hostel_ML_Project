Hostel Type Classification Dashboard
A Streamlit-based machine learning web app that predicts the hostel type from user-provided features using either K-Nearest Neighbors or Support Vector Machine.

Overview
This project provides an interactive dashboard for exploring a hostel dataset, visualizing the distribution of hostel types, training a classification model, and making predictions in real time. Users can choose between KNN and SVM, adjust test size, and enter feature values to get a predicted hostel type.

Features
Interactive Streamlit dashboard.
Dataset preview and basic information display.
Hostel type distribution chart.
Model selection between KNN and SVM.
Adjustable test split size.
Adjustable k value for KNN.
Accuracy display for the selected model.
Real-time hostel type prediction from user input.

Project Structure
bash
project-folder/
├── pipe.py
├── hostel_dataset_1000.csv
└── README.md

Technologies Used
Python

Streamlit

Pandas

Seaborn

Matplotlib

Scikit-learn

How It Works
The dataset is loaded from hostel_dataset_1000.csv.

The app shows a preview of the data and dataset details.

A count plot is generated for the Hostel_Type column.

The target column is encoded using LabelEncoder.

Features are standardized using StandardScaler.

The data is split into training and testing sets.

A model is trained based on the selected option:

KNN: uses the selected number of neighbors.

SVM: uses an RBF kernel.

Accuracy is calculated and shown.

Users enter feature values and click Predict to get the hostel type prediction.

Installation
1. Clone the repository

bash
git clone https://github.com/your-username/hostel-classifier.git
cd hostel-classifier
2. Create a virtual environment

bash
python -m venv venv
3. Activate the environment

Windows:

bash
venv\Scripts\activate
Mac/Linux:

bash
source venv/bin/activate
4. Install dependencies

bash
pip install -r requirements.txt
Running the App
bash
streamlit run app.py
Then open the local URL shown in the terminal, usually:

bash
http://localhost:8501
Usage
Open the app in your browser.

Choose a model from the sidebar.

Adjust the test size if needed.

If using KNN, select the value of k.

Review the dataset preview and hostel type distribution.

Enter feature values in the prediction section.

Click Predict to see the predicted hostel type.

Model Details
K-Nearest Neighbors

KNN classifies a sample based on the labels of its closest neighbors. In this project, distance-weighted voting is used, which gives closer neighbors more influence.

Support Vector Machine

SVM finds a decision boundary that separates classes in the feature space. The RBF kernel allows the model to capture nonlinear relationships.

Notes
The dataset should contain a Hostel_Type column.

All feature columns should be numeric for scaling to work properly.

The app uses standardization, which is important for KNN and SVM.

For better practice, scaling should ideally be fit on training data only.

Possible Improvements
Add confusion matrix and classification report.

Handle categorical feature columns.

Save trained models for reuse.

Add feature importance or explainability tools.

Improve preprocessing with a pipeline.

Add file upload support for custom datasets.

Requirements
Example requirements.txt:

text
streamlit
pandas
seaborn
matplotlib
scikit-learn
License
This project can be licensed under MIT, Apache 2.0, or any license you prefer.

Author
Developed for hostel type classification using Streamlit and machine learning.

Would you like me to turn this into a more polished GitHub README with badges, sections, and a professional tone?

