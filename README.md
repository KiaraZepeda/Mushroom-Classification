# Mushroom Classification
Machine Learning Project

# Overview
This project focuses on predicting the genus of mushrooms using machine learning techniques. The dataset used contains nine of the most common Northern European mushroom genuses. The goal is to demonstrate the application of Convolutional Neural Networks (CNNs) and Deep Neural Networks (DNNs) for classification tasks.

# Problem Statement
Mushrooms are a diverse and rich species with some edible varieties and others that are poisonous. Accurately classifying mushrooms based on their physical features can play a critical role in safety, especially in the context of foraging for edible mushrooms. This project aims to test the accuracy of predictive models for the intention of foraging. 

# Dataset
The dataset used in this project is from the Mushroom Classification Dataset available on Kaggle (https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images/data). The dataset includes nine folders with 300-1500 images each, almost 2 GB of storage.  

# Objective
The main objective of this project is to build and compare two classification models that can predict the genus of mushrooms based on images. By applying machine learning algorithms such as Convolutional Neural Networks (CNN) and Deep Neural Networks (DNN), the models will classify each image into one of the five genus classes.

# Methods Used
Data Preprocessing:
The folders were reduced to five and the number of images within each were randomly reduced to 500.  

Model Development:

Non-Supervised Learning Model:

K-Means Clustering: K-Means is a clustering algorithm was used to groups data points into clusters based on their similarity. 

Supervised (or Self-Supervised) Learning Models:

Convolutional Neural Networks (CNNs): CNNs were utilized for image-like feature extraction from structured data.

Deep Neural Networks (DNNs): A multi-layer neural network was used to learn complex patterns in the data and predict the correct class label.

Model Evaluation: The models were evaluated using metrics such as accuracy, precision, recall, and F1-score to ensure reliable performance. In addition, ROC curves were plotted to visually assess the trade-off between true positive rate (sensitivity) and false positive rate. The AUC (Area Under the Curve) was calculated to provide a single-number summary of the model's ability to distinguish between classes.

Model Performance: The performance of the models was assessed on a test dataset to evaluate their ability to generalize to unseen data.

# Results
When comparing the CNN and DNN model, the CNN model perfomed better. It had a higher test accuracy, and AUC. It also had a lower test loss.

CNN

Test Loss: 1.5037,
Test Accuracy: 0.3480,
AUC = 0.68

DNN

Test Loss: 1.6045,
Test accuracy: 0.2440,
AUC = 0.54

This makes sense since CNNs are more efficient for image-related modeling. The clustering gernerally contained similar amounts from all five genuses.
# Conclusion
This project demonstrates the use of machine learning algorithms, particularly deep learning models, for the classification of mushrooms based on their features. By leveraging both CNNs and DNNs, this project shows how sophisticated models can be applied to real-world datasets, such as species classification.

# Future Work
Improvement of Model: We can explore other advanced machine learning models like Random Forest and XGBoost to improve performance as well as futher hyperparameter and parameter tuning.

Exploring Other Features: Clearer and maybe even 360 images of each mushroom geneus for training could help improve the accuracy and generalization of the models.  

# Acknowledgements
Kaggle for providing the dataset.
