Linear Support Vector Classifier (SVC) Visualization
This repository contains Python code to visualize the decision boundaries created by a linear Support Vector Classifier (SVC) on the Iris dataset.

Dependencies:
*numpy
*matplotlib
*scikit-learn
You can install the dependencies using pip:
pip install numpy matplotlib scikit-learn
Usage:
1.Clone the repository:
git clone https://github.com/your-username/linear-svc-visualization.git
2.Navigate to the repository directory:
cd linear-svc-visualization
3.Run the Python script:
python linear_svc_visualization.py
Description:
The code performs the following tasks:
*Loads the Iris dataset.
*Prepares the data by selecting only the first two features (sepal length and sepal width).
*Initializes and trains a linear Support Vector Classifier (SVC) using scikit-learn.
*Creates a mesh grid to visualize the decision boundaries.
*Plots the decision boundaries and data points using matplotlib.
Result:
The script generates a plot showing the decision boundaries created by the linear SVC to classify the Iris dataset based on sepal length and sepal width.
