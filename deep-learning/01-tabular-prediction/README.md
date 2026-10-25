# Predictions

*This project is based on the first assignment from the "Deep Learning in Text Analytics" course by Professor Stefan Lessmann, Humboldt-Universität zu Berlin.*

##  Project Goal
To build and compare several machine learning models to find the most accurate one for predicting a continuous target variable from a tabular dataset.

##  Models & Technologies

### Models Benchmarked
This project implements and evaluates four distinct regression models:
* **Ridge Regression:** A linear model with L2 regularization to handle multicollinearity.
* **Random Forest:** An ensemble tree-based model.
* **Baseline Neural Network:** A simple, non-tuned deep learning model.
* **Tuned Neural Network:** A deep learning model optimized for performance.

### Technologies Used
The project was built using the following core libraries:
* **Data Manipulation:** pandas & NumPy
* **Modeling:** scikit-learn & TensorFlow (with Keras)
* **Hyperparameter Tuning:** Optuna
* **Visualization:** Matplotlib
