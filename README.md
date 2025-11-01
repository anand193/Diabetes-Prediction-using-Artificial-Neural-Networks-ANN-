# Diabetes-Prediction-using-Artificial-Neural-Networks-ANN-
A deep learning project that predicts the likelihood of diabetes using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The model is trained on medical diagnostic data to classify whether a patient is diabetic or not

🧠 Diabetes Prediction using Artificial Neural Networks (ANN)
📌 Project Overview

This project aims to predict whether a patient is likely to have diabetes based on key medical diagnostic features.
The model is built using an Artificial Neural Network (ANN) implemented in TensorFlow/Keras, trained on the Pima Indians Diabetes Dataset.
A Streamlit web application is included to provide an interactive user interface for making real-time predictions.

📊 Dataset Information

Source: Pima Indians Diabetes Database – Kaggle

Feature	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skinfold thickness (mm)
Insulin	2-Hour serum insulin (mu U/ml)
BMI	Body Mass Index
DiabetesPedigreeFunction	Diabetes heredity function
Age	Age in years
Outcome	1 = Diabetic, 0 = Non-diabetic
🧩 Model Architecture

Framework: TensorFlow / Keras

Model Type: Feedforward Neural Network (ANN)

Architecture:

Input Layer (8 features)

2 Hidden Layers (ReLU activation)

Output Layer (Sigmoid activation)

Optimizer: Adam

Loss Function: Binary Crossentropy

Evaluation Metric: Accuracy

⚙️ Model Performance
Metric	Training Set	Test Set
Accuracy	~85%	~80%
Loss	Low Binary Crossentropy	Low Binary Crossentropy

(Exact results may vary depending on random state and data split.)

🧮 Technologies Used

Programming Language: Python

Libraries: TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Streamlit

Model Saving: Pickle / Joblib for preprocessing pipeline, .h5 for ANN model

🧠 How It Works

The user inputs medical parameters into the Streamlit interface.

The input data is preprocessed using a StandardScaler (saved from training).

The ANN model predicts the probability of diabetes.

Based on a threshold of 0.5, the model outputs:

"High likelihood of diabetes", or

"Low likelihood of diabetes"
