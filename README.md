🧠 Diabetes Prediction using Artificial Neural Networks (ANN)
📌 Project Overview

This project aims to predict whether a patient is likely to have diabetes based on key medical diagnostic features.
The model is built using an Artificial Neural Network (ANN) implemented in TensorFlow/Keras, trained on the Pima Indians Diabetes Dataset.
A Streamlit web application is included to provide an interactive user interface for making real-time predictions.

--- 

## 📊 Dataset Information

Source: Pima Indians Diabetes Database – Kaggle

Feature	Description
- Pregnancies	Number of times pregnant
- Glucose	Plasma glucose concentration
- BloodPressure	Diastolic blood pressure (mm Hg)
- SkinThickness	Triceps skinfold thickness (mm)
- Insulin	2-Hour serum insulin (mu U/ml)
- BMI	Body Mass Index
- DiabetesPedigreeFunction	Diabetes heredity function
- Age	Age in years
- Outcome	1 = Diabetic, 0 = Non-diabetic

--- 

## 🧩 Model Architecture

- Framework: TensorFlow / Keras

- Model Type: Feedforward Neural Network (ANN)

Architecture:

- Input Layer (8 features)

- 1 Hidden Layers (ReLU activation)

- Output Layer (Sigmoid activation)

- Optimizer: Adam

- Loss Function: Binary Crossentropy

- Evaluation Metric: Accuracy

--- 

## ⚙️ Model Performance
- Training Set	- 	~80%
- Test Set -	~75%

--- 

## 🧮 Technologies Used

 -Programming Language: Python

- Libraries: TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Streamlit

- Model Saving: Pickle / Joblib for preprocessing pipeline, .h5 for ANN model

---

## 🔬 Future Improvements

- Improve accuracy through hyperparameter tuning or deeper network architecture

- Deploy on Streamlit Cloud or Hugging Face Spaces

- Enhance UI with data visualization and feedback metrics

## 👨‍💻 Author

Anand Mehto
Aspiring Data Scientist | Python | Machine Learning | Deep Learning

🔗 linkedin.com/in/anandmehto
