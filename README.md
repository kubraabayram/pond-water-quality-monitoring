# 🌊 Pond Water Quality Monitoring System

This project is an end-to-end system designed to monitor and analyze water quality parameters critical for aquaculture. The system evaluates key environmental indicators such as **pH, TDS, and temperature** and classifies water quality into three categories: **Poor, Medium, and Good**.

The solution combines both **rule-based logic** and **machine learning models** to provide accurate and real-time water quality assessment.

---

## 🧠 Decision Mechanism & Model Training

At the core of the system, a hybrid decision-making approach is implemented.

- A **rule-based scoring mechanism** was designed to evaluate pH, TDS, and temperature based on predefined threshold values.
- To address class imbalance in the dataset (especially the underrepresented "Poor" category), **synthetic data generation** was applied to improve model performance.

Multiple machine learning models were trained and evaluated:

- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Logistic Regression  

After performance comparison, the **Random Forest model** was selected as the primary prediction engine due to its consistency and reliability. The trained model was serialized using **joblib** and integrated into the backend system.

---

## 🚀 FastAPI Backend & System Integration

A **FastAPI-based RESTful API** was developed to enable communication between the system components.

### Key Features:

- **Prediction Endpoint (`/predict`)**  
  Processes incoming sensor data using both rule-based logic and the machine learning model to generate water quality predictions and actionable insights.

- **Data Management (CRUD Operations)**  
  Supports creating, reading, updating, and deleting historical sensor data records.

- **Data Standardization**  
  Input data is normalized using **StandardScaler**, ensuring consistency with the model training process.

---

## 📊 Results & Evaluation

During the training phase, models were evaluated using key performance metrics:

- Accuracy  
- Precision  
- Recall  

Additionally, a **confusion matrix** was used to analyze classification performance across different categories, validating the reliability of the system.
