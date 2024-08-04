**PROJECT OVERVIEW**


**Introduction**
This project aims to analyze and predict university admission chances using neural network techniques. The project includes data loading, feature engineering, model training, and evaluation, with visualizations to aid understanding.


**Directory Structure**
.git: Contains Git version control data.
.gitattributes: Git attributes file.
main.py: The main script to run the project.
output_charts: Directory for storing generated charts and visualizations.
requirements.txt: Lists the Python dependencies.
src: Source code directory containing modules for various tasks.


**Key Components**
main.py
The central script that coordinates the following tasks:

**Data Loading:**
Loads the admission data from src/dataset/Admission.csv.

**Feature Engineering:**
Engineers new features from the dataset.
Scales numeric features for better model performance.
Creates interaction terms between selected features.

**Data Partitioning:**
Separates the dataset into training and testing sets.

**Model Training:**
Trains a neural network model on the training data.

**Model Evaluation:**
Evaluates the trained model on the testing data.
Generates Mean Squared Error (MSE) and R-squared (R2) metrics.
Stores evaluation charts in the output_charts directory.


**requirements.txt**
Specifies the Python libraries required for the project:

pandas
numpy
seaborn
matplotlib
scikit-learn


**Conclusion**
This project provides a comprehensive analysis and prediction of university admission chances using neural network techniques. It includes various visualizations to aid in understanding the results and determining the model's performance. ​
