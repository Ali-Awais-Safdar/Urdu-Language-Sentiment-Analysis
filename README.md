# Urdu-Language-Sentiment-Analysis

#### Overview
This project focuses on sequence tagging using a Bidirectional Gated Recurrent Unit (GRU) neural network. The primary objective is to build a model that can tag sequences of words with the appropriate labels. This task is common in natural language processing (NLP) applications such as part-of-speech tagging, named entity recognition, and more.

#### Requirements
To run this project, you need the following dependencies:

Python 3.7
TensorFlow
Keras
NumPy
Pandas
Matplotlib
scikit-learn
You can install these dependencies using pip:
```
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```
#### Setup
Clone this repository:
```
git clone https://github.com/your-username/your-repository.git
cd your-repository
```
Ensure you have all the required dependencies installed.

Open the Jupyter Notebook:
```
jupyter notebook Assignment3.ipynb
```
#### Usage
Follow the steps in the notebook to process the data, train the model, and evaluate the results. Below is a brief description of each major section in the notebook:

## Data Processing
Load and preprocess the data.
Split the data into training and test sets.
##  Model Training
Define and compile the model.
Train the model on the training data.
Monitor the training process using training and validation loss.
##  Evaluation
Evaluate the model's performance on the test set.
Generate classification reports and confusion matrices.
##  Results
Plot training and validation loss over epochs.
Display evaluation metrics such as precision, recall, and F1-score.
#### Results
Here are some example outputs from the notebook:

Training and Validation Loss

Classification Report
```
              precision    recall  f1-score   support

         0.0       0.85      0.86      0.86      4795
         1.0       0.87      0.86      0.87      5205

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.86     10000
weighted avg       0.86      0.86      0.86     10000
```
#### License
This project is licensed under the MIT License. See the LICENSE file for details.
