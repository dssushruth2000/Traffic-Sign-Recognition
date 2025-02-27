Traffic Sign Classification using CNNs

This project classifies German traffic signs using three deep learning models:

Model 1: MobileNetV2-based Model
Model 2: Custom CNN with Dropout and Batch Normalization
Model 3: Deeper Custom CNN with Batch Normalization and Dropout
The models are trained and evaluated using the GTSRB dataset (German Traffic Sign Recognition Benchmark).

Setup Instructions:

1. Install Dependencies
Ensure the following libraries are installed. You can install them using pip:

pip install tensorflow numpy pandas opencv-python matplotlib seaborn scikit-learn pillow

Run Jupyter Notebook:

To open Jupyter Notebook, run this in your terminal or command prompt:

How to Run the Code in Jupyter Notebook:

1. Make sure the repository contains the dataset (gtsrb).
2. Open the Jupyter Notebook and run Final_Project_Traffic.ipynb.
3. Run the cells sequentially:
4. Preprocess and normalize images.
5. Define, train, and evaluate three models (Model 1, Model 2, Model 3).
6. Generate performance metrics such as:
	Training and Validation Metrics (accuracy/loss plots).
	Confusion Matrices for all models.
	ROC Curves for model comparison.
7. Visualize test predictions with actual and predicted labels.
8. Trained models will be saved in the Model/ directory.

Dependencies:

Ensure the following Python libraries are installed:

TensorFlow (2.x)
NumPy
Pandas
OpenCV (opencv-python)
Matplotlib
Seaborn
Scikit-learn
Pillow (PIL)
