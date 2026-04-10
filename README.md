# Neuroheadstate-EEG-Analysis
EEG-based eye state classification system using FastAPI and Machine Learning

This project classifies eye states (Open/Closed) and Neuro-States from 14-channel EEG data using Machine Learning and provides a real-time API.

🚀 Features
Real-time Prediction: Low-latency FastAPI RESTful API for instant state classification.

High Accuracy: Achieved 96.8% accuracy using the Label Spreading algorithm.

Signal Validation: Validated using the Berger Effect (Alpha Rhythm) and Power Spectral Density (PSD) analysis.

Interactive Dashboard: Built with JavaScript and HTML for real-time sensor monitoring.

🛠 Tech Stack
Language: Python

Framework: FastAPI

Libraries: Scikit-learn, Pandas, NumPy, Joblib, Pydantic

Frontend: JavaScript, HTML

Data Visualization: Included a PCA-based dimensionality reduction script to visualize the 14-dimensional EEG data distribution in a 2D space.
